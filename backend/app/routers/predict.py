"""Part 2: severity prediction served from the MLflow Production model, plus
lightweight drift/performance monitoring against the training baseline.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncConnection

from ..db import fetch_all, get_connection
from ..schemas import (
    DataQualityField,
    DataQualityReport,
    DriftReport,
    LatencyReport,
    ModelHealth,
    ModelInfo,
    PerformanceReport,
    PredictRequest,
    PredictResponse,
)

def _find_ml_dir(start: Path) -> Path:
    """Walk up from this file to find the 'ml' directory, rather than
    hardcoding a parent count -- so the import keeps working whether the app
    is launched from backend/, from the repo root, or from anywhere else."""
    for parent in start.parents:
        candidate = parent / "ml"
        if (candidate / "features.py").exists():
            return candidate
    raise RuntimeError("Could not locate ml/features.py from any ancestor directory")


ML_DIR = _find_ml_dir(Path(__file__).resolve())
sys.path.insert(0, str(ML_DIR))
from features import (CATEGORICAL_VOCAB, FEATURE_COLUMNS, NUMERIC_COLUMNS,  # noqa: E402
                      build_features_df, encode_category, input_issues)

router = APIRouter(tags=["predict"])

MODEL_NAME = "traffic_severity_classifier"
BASELINE_PATH = ML_DIR / "artifacts" / "baseline.json"


class LatencyTracker:
    """Rolling in-process record of inference latency and failures.

    Deliberately in-memory and bounded: this is a demonstration of serving
    telemetry, not a metrics backend. It resets when the process restarts, and
    a real deployment would export these to Prometheus/OpenTelemetry instead.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self.samples: deque[float] = deque(maxlen=maxlen)
        self.served = 0
        self.failures = 0

    def record(self, elapsed_ms: float) -> None:
        self.samples.append(elapsed_ms)
        self.served += 1

    def record_failure(self) -> None:
        self.failures += 1

    def percentile(self, pct: float) -> float | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        # nearest-rank: index of the first value at or above the percentile
        idx = min(int(math.ceil(pct / 100 * len(ordered))) - 1, len(ordered) - 1)
        return round(ordered[max(idx, 0)], 2)


latency = LatencyTracker()


class InputQualityTracker:
    """Counts problems in live /predict inputs (missing numerics, categorical
    values outside the model's vocabulary). In-memory like LatencyTracker."""

    def __init__(self) -> None:
        self.requests = 0
        self.with_issues = 0
        self.counts: dict[str, int] = {}

    def record(self, issues: dict[str, list[str]]) -> None:
        self.requests += 1
        flat = [f"missing:{c}" for c in issues["missing_numeric"]] +                [f"unknown:{c}" for c in issues["unknown_category"]]
        if flat:
            self.with_issues += 1
        for key in flat:
            self.counts[key] = self.counts.get(key, 0) + 1


input_quality = InputQualityTracker()


def _model_input(df: pd.DataFrame) -> pd.DataFrame:
    """What the loaded model's predict() expects: raw records for the packaged
    pipeline, pre-encoded features for a legacy registration."""
    return build_features_df(df)[FEATURE_COLUMNS] if state.legacy else df


def _raw_frame(records: list[dict]) -> pd.DataFrame:
    """Raw records -> the DataFrame the packaged pipeline expects. Numerics
    are coerced (the warehouse hands back Decimals, requests may carry None)
    so MLflow's schema enforcement sees plain floats; the pipeline itself does
    all feature engineering."""
    df = pd.DataFrame(records)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


class ModelState:
    model = None
    version: str | None = None
    stage: str | None = None
    baseline: dict | None = None
    # True for a model registered before the packaged-pipeline change: it
    # expects pre-encoded feature columns, so the API encodes for it.
    legacy: bool = False


state = ModelState()


def load_model() -> None:
    """Called from main.py's lifespan handler. Never raises -- a missing
    model just means /predict returns 503 until train.py has registered one.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or (ML_DIR / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            print(f"[predict] no Production version of '{MODEL_NAME}' found")
            return
        mv = versions[0]
        state.model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
        state.version = mv.version
        state.stage = mv.current_stage
        inner = state.model.unwrap_python_model()
        state.legacy = not hasattr(inner, "feature_columns")
        if state.legacy:
            print("[predict] WARNING: legacy model without bundled preprocessing -- "
                  "retrain with ml/train.py to register the packaged pipeline")
        elif inner.vocab != {k: list(v) for k, v in CATEGORICAL_VOCAB.items()}:
            print("[predict] WARNING: lookup vocabularies changed since this model "
                  "was trained -- retrain (see ml/RETRAINING.md)")
        print(f"[predict] loaded {MODEL_NAME} v{mv.version} ({mv.current_stage})")
    except Exception as exc:  # noqa: BLE001
        print(f"[predict] could not load model from MLflow: {exc}")

    if BASELINE_PATH.exists():
        state.baseline = json.loads(BASELINE_PATH.read_text())


def _require_model():
    if state.model is None:
        raise HTTPException(status_code=503, detail="No Production model is loaded yet. Run ml/train.py.")
    return state.model


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    model = _require_model()
    record = req.model_dump()
    input_quality.record(input_issues(record))
    row = _raw_frame([record])

    started = time.perf_counter()
    try:
        proba = model.predict(_model_input(row))
    except Exception as exc:  # noqa: BLE001
        latency.record_failure()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc
    latency.record((time.perf_counter() - started) * 1000)

    proba_arr = np.asarray(proba)
    if proba_arr.ndim == 2 and state.baseline:
        labels = state.baseline["class_labels"]
        probs = {str(lbl): float(p) for lbl, p in zip(labels, proba_arr[0])}
        best_idx = int(np.argmax(proba_arr[0]))
        predicted = labels[best_idx]
    else:
        predicted = int(proba_arr.reshape(-1)[0])
        probs = {str(predicted): 1.0}

    severity_labels = {1: "Fatal", 2: "Serious", 3: "Slight", -1: "Data missing"}
    return PredictResponse(
        predicted_severity_id=int(predicted),
        predicted_severity_label=severity_labels.get(int(predicted), "Unknown"),
        probabilities=probs,
        model_version=str(state.version or "unknown"),
    )


@router.get("/model/info", response_model=ModelInfo)
async def model_info():
    if state.model is None or state.baseline is None:
        raise HTTPException(status_code=503, detail="No model loaded yet")
    b = state.baseline
    return ModelInfo(
        model_name=MODEL_NAME,
        version=str(state.version),
        stage=str(state.stage),
        algorithm=b.get("algorithm"),
        trained_at=b.get("trained_at"),
        metrics=b.get("metrics", {}),
    )


@router.get("/model/health", response_model=ModelHealth)
async def model_health():
    return ModelHealth(model_loaded=state.model is not None,
                        model_version=str(state.version) if state.version else None)


def _psi(baseline_edges: list[float], baseline_props: list[float], current: pd.Series) -> float:
    if not baseline_edges or current.empty:
        return 0.0
    counts, _ = np.histogram(current.dropna(), bins=[-math.inf, *baseline_edges, math.inf])
    current_props = counts / max(counts.sum(), 1)
    psi = 0.0
    for b, c in zip(baseline_props, current_props):
        b = max(b, 1e-4)
        c = max(c, 1e-4)
        psi += (c - b) * math.log(c / b)
    return float(psi)


def _class_psi_terms(baseline_dist: dict, current: pd.Series) -> dict[int, tuple[float, float, float]]:
    """Per-class (baseline share, current share, PSI contribution).

    PSI is a sum of independent per-bucket terms, so reporting them per class
    shows *which* severity moved, not just that something did.
    """
    current_shares = current.value_counts(normalize=True)
    terms: dict[int, tuple[float, float, float]] = {}
    for raw_label, baseline_share in baseline_dist.items():
        label = int(raw_label)
        b = max(float(baseline_share), 1e-4)
        c = max(float(current_shares.get(label, 0.0)), 1e-4)
        terms[label] = (float(baseline_share), float(current_shares.get(label, 0.0)),
                        (c - b) * math.log(c / b))
    return terms


SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight", -1: "Data missing"}


@router.get("/monitoring/drift", response_model=list[DriftReport])
async def monitoring_drift(conn: AsyncConnection = Depends(get_connection)):
    """Feature, location and class drift of the most recent 100,000 accidents (about one year, so seasons cancel out) against
    the training baseline, all measured as PSI (>0.2 = retrain, per
    ml/RETRAINING.md)."""
    if state.baseline is None:
        raise HTTPException(status_code=503, detail="No training baseline available yet")

    rows = await fetch_all(conn, """
        SELECT fa.severity_id, fa.number_of_vehicles, d.hour, d.day_of_week, d.month,
               r.speed_limit, w.temperature_c, w.precipitation_mm,
               l.latitude, l.longitude
        FROM fact_accidents fa
        JOIN dim_date d ON d.date_key = fa.date_key
        JOIN dim_road r ON r.road_id = fa.road_id
        JOIN dim_weather w ON w.weather_id = fa.weather_id
        JOIN dim_location l ON l.location_id = fa.location_id
        -- Most recent accidents by when they happened. loaded_at is
        -- identical across a bulk load, so ordering by it would return an
        -- arbitrary slice of physical row order rather than a time window.
        ORDER BY d.full_date DESC, fa.accident_index DESC
        LIMIT 100000
    """)
    current = pd.DataFrame(rows)
    reports: list[DriftReport] = []
    if current.empty:
        return reports

    # --- model input drift ---
    for feature, stats in state.baseline.get("numeric_feature_stats", {}).items():
        # The window is deliberately ~1 year: weather and month are seasonal,
        # and a shorter slice would read as drift against the multi-year baseline.
        if feature not in current.columns:
            continue
        psi = _psi(stats["decile_edges"], stats["decile_props"], current[feature])
        reports.append(DriftReport(
            kind="feature", feature=feature,
            baseline_value=stats.get("mean"),
            current_value=float(current[feature].mean()),
            psi=round(psi, 4), drifted=psi > 0.2,
        ))

    # --- location drift: are accidents arriving from a different geography? ---
    for feature, stats in state.baseline.get("location_stats", {}).items():
        if feature not in current.columns:
            continue
        series = pd.to_numeric(current[feature], errors="coerce")
        psi = _psi(stats["decile_edges"], stats["decile_props"], series)
        reports.append(DriftReport(
            kind="location", feature=feature,
            baseline_value=stats.get("mean"),
            current_value=float(series.mean()),
            psi=round(psi, 4), drifted=psi > 0.2,
        ))

    # --- class drift: has the severity mix itself shifted? ---
    baseline_dist = state.baseline.get("class_distribution") or {}
    if baseline_dist and "severity_id" in current.columns:
        terms = _class_psi_terms(baseline_dist, current["severity_id"])
        total_psi = sum(t[2] for t in terms.values())
        for label, (baseline_share, current_share, term) in sorted(terms.items()):
            reports.append(DriftReport(
                kind="class",
                feature=f"severity: {SEVERITY_LABELS.get(label, label)}",
                baseline_value=round(baseline_share, 4),
                current_value=round(current_share, 4),
                psi=round(term, 4),
                # the retrain decision is about the distribution as a whole,
                # so every class row carries the same verdict
                drifted=total_psi > 0.2,
            ))

    return reports


SPEED_LIMITS = {20, 30, 40, 50, 60, 70}


@router.get("/monitoring/data-quality", response_model=DataQualityReport)
async def monitoring_data_quality(conn: AsyncConnection = Depends(get_connection)):
    """Input-data quality: missing/out-of-range numerics and categorical values
    unseen by the model, for the newest warehouse accidents, plus the same
    checks tallied over live /predict requests since the backend started."""
    rows = await fetch_all(conn, """
        SELECT d.full_date, d.hour, d.day_of_week, d.month, fa.number_of_vehicles,
               r.speed_limit, r.road_type, r.junction_detail,
               w.light_conditions, w.weather_conditions, w.road_surface_conditions,
               w.temperature_c, w.precipitation_mm, l.urban_or_rural_area
        FROM fact_accidents fa
        JOIN dim_date d ON d.date_key = fa.date_key
        JOIN dim_road r ON r.road_id = fa.road_id
        JOIN dim_weather w ON w.weather_id = fa.weather_id
        JOIN dim_location l ON l.location_id = fa.location_id
        ORDER BY d.full_date DESC, fa.accident_index DESC
        LIMIT 2000
    """)
    if not rows:
        raise HTTPException(status_code=404, detail="No accident data loaded yet")
    df = _raw_frame(rows)
    baseline_stats = (state.baseline or {}).get("numeric_feature_stats", {})
    plausible = {
        "hour": lambda s: ~s.between(0, 23),
        "day_of_week": lambda s: ~s.between(1, 7),
        "month": lambda s: ~s.between(1, 12),
        "speed_limit": lambda s: ~s.isin(SPEED_LIMITS),
        "number_of_vehicles": lambda s: s < 1,
        "temperature_c": lambda s: ~s.between(-30, 45),
        "precipitation_mm": lambda s: s < 0,
    }
    fields: list[DataQualityField] = []
    for col in NUMERIC_COLUMNS:
        s = df[col]
        missing = float(s.isna().mean())
        base = baseline_stats.get(col, {}).get("missing_rate")
        invalid = float((plausible[col](s) & s.notna()).mean())
        fields.append(DataQualityField(
            field=col, kind="numeric", missing_rate=round(missing, 4),
            baseline_missing_rate=None if base is None else round(base, 4),
            invalid_rate=round(invalid, 4),
            flagged=invalid > 0.05 or (base is not None and missing - base > 0.10)
                    or (base is None and missing > 0.25),
        ))
    for field in CATEGORICAL_VOCAB:
        unknown_idx = len(CATEGORICAL_VOCAB[field])
        codes = df[field].map(lambda v: encode_category(field, v))
        unknown = float((codes == unknown_idx).mean())
        missing = float(df[field].isna().mean())
        fields.append(DataQualityField(
            field=field, kind="categorical", missing_rate=round(missing, 4),
            invalid_rate=round(unknown, 4), flagged=unknown > 0.05,
        ))
    return DataQualityReport(
        window_size=len(df),
        window_start=str(df["full_date"].min()), window_end=str(df["full_date"].max()),
        fields=fields,
        live_requests=input_quality.requests,
        live_requests_with_issues=input_quality.with_issues,
        live_issue_counts=input_quality.counts,
    )


@router.get("/monitoring/latency", response_model=LatencyReport)
async def monitoring_latency():
    """Serving telemetry for /predict: inference latency percentiles and the
    failure count, both required by the monitoring brief alongside drift and
    model performance."""
    total_attempts = latency.served + latency.failures
    return LatencyReport(
        sampled_requests=len(latency.samples),
        predictions_served=latency.served,
        failures=latency.failures,
        failure_rate=round(latency.failures / total_attempts, 4) if total_attempts else 0.0,
        p50_ms=latency.percentile(50),
        p95_ms=latency.percentile(95),
        p99_ms=latency.percentile(99),
        max_ms=round(max(latency.samples), 2) if latency.samples else None,
    )


@router.get("/monitoring/performance", response_model=PerformanceReport)
async def monitoring_performance(conn: AsyncConnection = Depends(get_connection)):
    model = _require_model()
    rows = await fetch_all(conn, """
        SELECT fa.severity_id, fa.number_of_vehicles, d.hour, d.day_of_week, d.month,
               d.is_weekend, d.full_date, r.road_type, r.junction_detail, r.speed_limit,
               l.urban_or_rural_area,
               w.light_conditions, w.weather_conditions, w.road_surface_conditions,
               w.temperature_c, w.precipitation_mm
        FROM fact_accidents fa
        JOIN dim_date d ON d.date_key = fa.date_key
        JOIN dim_road r ON r.road_id = fa.road_id
        JOIN dim_weather w ON w.weather_id = fa.weather_id
        JOIN dim_location l ON l.location_id = fa.location_id
        -- Score the newest accidents, which fall in the chronological
        -- test split -- ordering by loaded_at would mix training rows in
        -- and report a rosier macro-F1 than the model actually earns.
        ORDER BY d.full_date DESC, fa.accident_index DESC
        LIMIT 1000
    """)
    if not rows:
        raise HTTPException(status_code=404, detail="No accident data loaded yet")

    df = pd.DataFrame(rows)
    X = _raw_frame(df.to_dict("records"))
    y_true = df["severity_id"].tolist()

    proba = np.asarray(model.predict(_model_input(X)))
    labels = state.baseline["class_labels"] if state.baseline else sorted(set(y_true))
    y_pred = [labels[int(np.argmax(p))] for p in proba] if proba.ndim == 2 else proba.tolist()

    from sklearn.metrics import f1_score, recall_score

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    severe_actual = [i for i, y in enumerate(y_true) if y in (1, 2)]
    fn = sum(1 for i in severe_actual if y_pred[i] not in (1, 2))
    fnr = fn / len(severe_actual) if severe_actual else 0.0

    dates = df["full_date"]
    return PerformanceReport(
        window_start=str(dates.min()),
        window_end=str(dates.max()),
        macro_f1=round(float(macro_f1), 4),
        recall_macro=round(float(recall_macro), 4),
        false_negative_rate_fatal_serious=round(float(fnr), 4),
        sample_size=len(df),
    )
