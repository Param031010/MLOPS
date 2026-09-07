"""Part 2: severity prediction served from the MLflow Production model, plus
lightweight drift/performance monitoring against the training baseline.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncConnection

from ..db import fetch_all, get_connection
from ..schemas import (
    DriftReport,
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
from features import FEATURE_COLUMNS, build_features  # noqa: E402

router = APIRouter(tags=["predict"])

MODEL_NAME = "traffic_severity_classifier"
BASELINE_PATH = ML_DIR / "artifacts" / "baseline.json"


class ModelState:
    model = None
    version: str | None = None
    stage: str | None = None
    baseline: dict | None = None


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
    feat = build_features(req.model_dump())
    row = pd.DataFrame([feat])[FEATURE_COLUMNS]

    try:
        proba = model.predict(row)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

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


@router.get("/monitoring/drift", response_model=list[DriftReport])
async def monitoring_drift(conn: AsyncConnection = Depends(get_connection)):
    if state.baseline is None:
        raise HTTPException(status_code=503, detail="No training baseline available yet")

    rows = await fetch_all(conn, """
        SELECT fa.number_of_vehicles, d.hour, d.day_of_week, d.month, r.speed_limit,
               w.temperature_c, w.precipitation_mm
        FROM fact_accidents fa
        JOIN dim_date d ON d.date_key = fa.date_key
        JOIN dim_road r ON r.road_id = fa.road_id
        JOIN dim_weather w ON w.weather_id = fa.weather_id
        ORDER BY fa.loaded_at DESC
        LIMIT 2000
    """)
    current = pd.DataFrame(rows)
    reports = []
    for feature, stats in state.baseline.get("numeric_feature_stats", {}).items():
        if feature not in current.columns:
            continue
        psi = _psi(stats["decile_edges"], stats["decile_props"], current[feature])
        reports.append(DriftReport(
            feature=feature,
            baseline_mean=stats.get("mean"),
            current_mean=float(current[feature].mean()) if not current.empty else None,
            psi=round(psi, 4),
            drifted=psi > 0.2,
        ))
    return reports


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
        ORDER BY fa.loaded_at DESC
        LIMIT 1000
    """)
    if not rows:
        raise HTTPException(status_code=404, detail="No accident data loaded yet")

    df = pd.DataFrame(rows)
    X = pd.DataFrame([build_features(r) for r in df.to_dict("records")])[FEATURE_COLUMNS]
    y_true = df["severity_id"].tolist()

    proba = np.asarray(model.predict(X))
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
