"""Train RandomForest, XGBoost and LightGBM severity classifiers on a
chronological split, track every run in MLflow, and promote the best (by
macro-F1) to the Production stage of the model registry.

Wraps whichever library wins in one mlflow.pyfunc.PythonModel so
backend/app/routers/predict.py has a single, uniform predict_proba-shaped
serving contract regardless of which framework trained the winner.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLUMNS, NUMERIC_COLUMNS, build_features_df  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_NAME = "traffic_severity_classifier"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EXPERIMENT_NAME = "traffic-accident-severity"

TRAIN_QUERY = """
    SELECT fa.accident_index, fa.severity_id, fa.number_of_vehicles,
           d.full_date, d.hour, d.day_of_week, d.month, d.is_weekend,
           r.road_type, r.speed_limit, r.junction_detail,
           l.urban_or_rural_area, l.latitude, l.longitude,
           w.light_conditions, w.weather_conditions, w.road_surface_conditions,
           w.temperature_c, w.precipitation_mm
    FROM fact_accidents fa
    JOIN dim_date d ON d.date_key = fa.date_key
    JOIN dim_road r ON r.road_id = fa.road_id
    JOIN dim_weather w ON w.weather_id = fa.weather_id
    JOIN dim_location l ON l.location_id = fa.location_id
    ORDER BY d.full_date
"""


class SeverityModelWrapper(mlflow.pyfunc.PythonModel):
    """Uniform predict_proba-shaped serving contract over any of the three
    trained model flavours."""

    def __init__(self, model, classes: list[int]):
        self.model = model
        self.classes = list(classes)

    def predict(self, context, model_input: pd.DataFrame):
        return self.model.predict_proba(model_input[FEATURE_COLUMNS])


def load_training_data() -> pd.DataFrame:
    from sqlalchemy import create_engine

    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (check your .env)")
    engine = create_engine(url)
    df = pd.read_sql(TRAIN_QUERY, engine)
    if df.empty:
        raise SystemExit("fact_accidents is empty -- run the ETL pipeline (extract/transform/load) first")
    return df


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    df = df.sort_values("full_date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def decile_profile(series: pd.Series) -> dict | None:
    """Decile edges + the training set's share of rows in each bucket.

    This is the shape /monitoring/drift's PSI computation expects, so anything
    profiled this way -- a model feature or a coordinate -- is comparable
    against live data with the same code path.
    """
    series = series.dropna().astype(float)
    if series.empty:
        return None
    edges = np.quantile(series, np.linspace(0.1, 0.9, 9)).tolist()
    counts, _ = np.histogram(series, bins=[-np.inf, *edges, np.inf])
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "decile_edges": edges,
        "decile_props": (counts / counts.sum()).tolist(),
    }


def numeric_feature_baseline(train_df: pd.DataFrame) -> dict:
    stats = {}
    for col in NUMERIC_COLUMNS:
        profile = decile_profile(train_df[col])
        if profile is not None:
            stats[col] = profile
    return stats


def location_baseline(train_df: pd.DataFrame) -> dict:
    """Latitude/longitude profile of the training set, so location drift is
    measurable: a model fitted mostly on one part of the country and later
    scored on another shows up as a PSI shift on these two columns even when
    every model feature looks stable."""
    stats = {}
    for col in ["latitude", "longitude"]:
        profile = decile_profile(train_df[col])
        if profile is not None:
            stats[col] = profile
    return stats


def evaluate_split(model, X, y_true, classes) -> dict:
    proba = model.predict_proba(X)
    y_pred = [classes[i] for i in np.argmax(proba, axis=1)]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def default_tracking_uri() -> str:
    """An absolute file:// URI, since train.py/evaluate.py/predict.py run
    from different working directories and must share one MLflow store."""
    return (Path(__file__).parent / "mlruns").resolve().as_uri()


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[train] loading training data from the warehouse...")
    df = load_training_data()
    print(f"[train] {len(df)} labeled accidents loaded")

    train_df, val_df, test_df = chronological_split(df)
    print(f"[train] chronological split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    classes = sorted(df["severity_id"].unique().tolist())
    X_train = build_features_df(train_df)
    X_val = build_features_df(val_df)
    X_test = build_features_df(test_df)
    y_train = train_df["severity_id"].tolist()
    y_val = val_df["severity_id"].tolist()
    y_test = test_df["severity_id"].tolist()

    sample_weight = compute_sample_weight("balanced", y_train)

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=14, class_weight="balanced",
            n_jobs=-1, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
            n_jobs=-1, random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1,
            class_weight="balanced", n_jobs=-1, random_state=42, verbosity=-1,
        ),
    }

    results = {}
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = [class_to_idx[c] for c in y_train]

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            print(f"[train] fitting {name}...")
            if name == "XGBoost":
                # XGBoost has no class_weight=; balance via sample_weight instead,
                # and fit on 0..K-1 label indices (its multi:softprob requirement).
                # Its predict_proba then returns columns in that same index order,
                # which is exactly `classes` since class_to_idx = enumerate(classes).
                model.fit(X_train, y_train_idx, sample_weight=sample_weight)
            else:
                # RandomForest and LightGBM already balance via class_weight=.
                model.fit(X_train, y_train)

            val_metrics = evaluate_split(model, X_val, y_val, classes)
            test_metrics = evaluate_split(model, X_test, y_test, classes)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            results[name] = {
                "model": model, "val_metrics": val_metrics, "test_metrics": test_metrics,
                "run_id": mlflow.active_run().info.run_id,
            }
            print(f"[train] {name}: val_macro_f1={val_metrics['macro_f1']:.4f} "
                  f"test_macro_f1={test_metrics['macro_f1']:.4f}")

    best_name = max(results, key=lambda n: results[n]["val_metrics"]["macro_f1"])
    best = results[best_name]
    print(f"[train] best model: {best_name} (val_macro_f1={best['val_metrics']['macro_f1']:.4f})")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{best_name}-production-candidate"):
        wrapper = SeverityModelWrapper(best["model"], classes)
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(3),
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in best["test_metrics"].items()})
        mlflow.set_tag("algorithm", best_name)

        client = mlflow.tracking.MlflowClient()
        version = model_info.registered_model_version
        for mv in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
            client.transition_model_version_stage(MODEL_NAME, mv.version, stage="Archived")
        client.transition_model_version_stage(MODEL_NAME, version, stage="Production")
        print(f"[train] registered {MODEL_NAME} v{version} and promoted to Production")

    baseline = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": best_name,
        "model_version": version,
        "metrics": best["test_metrics"],
        "class_labels": classes,
        "numeric_feature_stats": numeric_feature_baseline(train_df),
        "location_stats": location_baseline(train_df),
        "class_distribution": pd.Series(y_train).value_counts(normalize=True).to_dict(),
    }
    (ARTIFACTS_DIR / "baseline.json").write_text(json.dumps(baseline, indent=2, default=str))
    print(f"[train] wrote baseline to {ARTIFACTS_DIR / 'baseline.json'}")


if __name__ == "__main__":
    main()
