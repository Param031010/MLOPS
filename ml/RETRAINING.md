# Retraining policy

The severity classifier is retrained manually (`python train.py`) rather
than on a fixed schedule, but a retrain is **required** — not optional —
as soon as any of the following is true.

## Triggers

| Signal | Threshold | Source |
|---|---|---|
| Rolling macro-F1 drop | More than 5 points below the Production model's logged `test_macro_f1` baseline | `GET /monitoring/performance` |
| False-negative rate on Fatal/Serious | Exceeds 15% (a Fatal or Serious accident predicted as Slight is the costliest error this model can make) | `GET /monitoring/performance` |
| Feature drift | Population Stability Index (PSI) exceeds **0.2** on any single feature in `GET /monitoring/drift` | `dim_weather`, `dim_road`, `dim_date` via recent `fact_accidents` |
| New data volume | A `load.py` run adds a full new year of accidents to `fact_accidents` | `etl_audit_log` |
| Upstream schema/code change | STATS19 changes its lookup codes, or `ml/features.py`'s `CATEGORICAL_VOCAB` / `FEATURE_COLUMNS` changes | code review |

PSI interpretation used here (standard bands): `< 0.1` no meaningful shift,
`0.1-0.2` moderate shift worth watching, `> 0.2` significant shift -- retrain.

## Procedure

1. Re-run the ETL pipeline (`extract.py` -> `transform.py` -> `load.py`) so
   `fact_accidents` includes the latest data.
2. `python ml/train.py`. This re-fits RandomForest, XGBoost and LightGBM on
   a fresh chronological split, logs all three to MLflow, and registers the
   macro-F1 winner as a new model version.
3. `python ml/evaluate.py` against the new Production version to get a
   confusion matrix and classification report for the report/submission.
4. Compare the new version's `test_macro_f1` to the previous Production
   version (visible in the MLflow UI, `mlflow ui --port 5000`). Only keep
   the new version promoted if it is not worse than the old one by more
   than 2 points -- `train.py` auto-promotes the winner of its own run, so
   a regression must be caught here and manually rolled back with
   `MlflowClient.transition_model_version_stage(..., stage="Archived")`
   on the new version and re-promoting the previous one.
5. Restart the backend so the FastAPI
   `lifespan` handler picks up the new Production model and baseline.

## What is intentionally *not* automated

Retraining is not triggered automatically by the monitoring endpoints --
`/monitoring/drift` and `/monitoring/performance` are read-only signals for
a human to act on. Automatic retraining on live production-severity
accident data is out of scope for this project (STATS19 records arrive in
DfT's own release cycle, not in real time).
