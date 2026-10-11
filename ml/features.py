"""Shared feature engineering for train.py and backend/app/routers/predict.py.

Importing this from both places is the only thing that guarantees training
and serving can't drift apart: the categorical vocabularies come straight
from etl/stats19_lookups.py, and both the trainer and the API build features
by calling build_features_df() / build_features() below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# In the repo the lookups live in ../etl; inside a logged MLflow model both
# files are bundled side by side (see train.py's code_paths).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "etl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats19_lookups as lk  # noqa: E402

CATEGORICAL_VOCAB: dict[str, list[str]] = {
    "road_type": sorted(set(lk.ROAD_TYPE.values())),
    "junction_detail": sorted(set(lk.JUNCTION_DETAIL.values())),
    "light_conditions": sorted(set(lk.LIGHT_CONDITIONS.values())),
    "weather_conditions": sorted(set(lk.WEATHER_CONDITIONS.values())),
    "road_surface_conditions": sorted(set(lk.ROAD_SURFACE_CONDITIONS.values())),
    "urban_or_rural_area": sorted(set(lk.URBAN_OR_RURAL_AREA.values())),
}

NUMERIC_COLUMNS = [
    "hour", "day_of_week", "month", "speed_limit",
    "number_of_vehicles", "temperature_c", "precipitation_mm",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + ["is_weekend"] + [
    f"{field}_code" for field in CATEGORICAL_VOCAB
]


# Position of each label within its vocabulary, precomputed. list.index() is a
# linear scan, which at ~514k rows x 6 categorical fields is tens of millions
# of string comparisons; this makes the lookup O(1) with identical results.
_VOCAB_INDEX: dict[str, dict[str, int]] = {
    field: {label: i for i, label in enumerate(vocab)}
    for field, vocab in CATEGORICAL_VOCAB.items()
}


def encode_category(field: str, value) -> int:
    index = _VOCAB_INDEX[field]
    unknown_bucket = len(CATEGORICAL_VOCAB[field])
    try:
        return index.get(value, unknown_bucket)
    except TypeError:  # unhashable value
        return unknown_bucket


def _to_float(val) -> float:
    """None, NaN and unparseable values all become NaN -- the same value
    pandas gives a NULL when train.py reads the warehouse, so a missing
    reading looks identical at training and serving time. (The tree models
    treat NaN natively; mapping it to 0.0 would silently turn "unknown
    temperature" into "freezing".)"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def build_features(record: dict) -> dict:
    """record: a raw accident-like dict, or a /predict request body dict."""
    feat: dict = {}
    for col in NUMERIC_COLUMNS:
        feat[col] = _to_float(record.get(col))
    weekend = record.get("is_weekend")
    if weekend is None or weekend != weekend:  # None or NaN: derive it
        # STATS19 day_of_week: 1 = Sunday ... 7 = Saturday (as in transform.py)
        weekend = record.get("day_of_week") in (1, 7)
    feat["is_weekend"] = int(bool(weekend))
    for field in CATEGORICAL_VOCAB:
        feat[f"{field}_code"] = encode_category(field, record.get(field))
    return feat


def build_features_df(df: pd.DataFrame) -> pd.DataFrame:
    records = df.to_dict("records")
    encoded = pd.DataFrame([build_features(r) for r in records], index=df.index)
    return encoded[FEATURE_COLUMNS]


def input_issues(record: dict) -> dict[str, list[str]]:
    """Data-quality problems in one raw record, for serving-time monitoring:
    numeric fields that are missing/unparseable and categorical values the
    model's vocabulary has never seen (they would silently fall into the
    'unknown' bucket)."""
    missing = [c for c in NUMERIC_COLUMNS if _to_float(record.get(c)) != _to_float(record.get(c))]
    unknown = [f for f in CATEGORICAL_VOCAB
               if encode_category(f, record.get(f)) == len(CATEGORICAL_VOCAB[f])]
    return {"missing_numeric": missing, "unknown_category": unknown}
