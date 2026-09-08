# Traffic Accident Severity Analytics & Prediction

Individual project — Data Engineering and MLOps course (100 marks, 50/50
split across Part 1 and Part 2). UK STATS19 road collision records enriched
with historical weather (Open-Meteo) and road context (TomTom reverse
geocoding), loaded into a PostgreSQL star schema, visualized on an 8-page
interactive Streamlit dashboard, and served through a FastAPI backend that
also predicts accident severity with an MLflow-tracked model.

## 1. Objectives

- Integrate accident, weather, and road datasets into one analytical store.
- Build a location- and time-aware database (dimensional warehouse).
- Visualize accident patterns and contributing factors interactively.
- Operationalize a severity-classification model with tracked experiments,
  a registry, an inference API, and basic monitoring (Part 2).

## 2. Data sources

| Source | Provides | Access |
|---|---|---|
| UK Road Safety Open Data (STATS19) | Collision, casualty, vehicle records | Provided locally in `dataset/raw/` |
| Open-Meteo Historical Weather API | Weather at accident time/location | Free, no key, quota-limited per day |
| TomTom Reverse Geocoding | Road/street context | Free-tier API key, quota-limited per day |

## 3. System architecture

```
dataset/raw/ (local CSVs) ─┐
Open-Meteo                 ├─► extract.py ─► raw zone ─► transform.py ─► staging ─► load.py ─► Postgres (+ PostGIS if available)
TomTom geocoding          ─┘   (+ audit log)  (immutable)  (clean/decode/join)   (parquet)    (star schema + mart)
                                                                                                    │
                                                          Streamlit multi-page app ◄─ FastAPI (:9001) ◄─────┘
                                                          (8 pages, one per view)      (mart + /predict)
                                                                                          ▲
                                                                         Part 2: MLflow-tracked model
                                                                         ("traffic_severity_classifier"),
                                                                         served from the same FastAPI app
```

PostgreSQL, the ETL scripts, and the Streamlit dashboard all run natively as
local processes, each in its own virtualenv. The FastAPI backend can run the
same way, or be built into a single-purpose Docker image
(`backend/Dockerfile`) that containerizes only the inference/analytics API —
Postgres and Streamlit are never containerized. An Airflow DAG
(`airflow/dags/traffic_accident_etl_dag.py`) offers a third way to trigger
the same `extract → transform → load` scripts, alongside a manual CLI run
and `POST /admin/run-etl` — see Section 10 for how to run either.

## 4. Tech stack

| Layer | Choice |
|---|---|
| Data source ingestion | Python, pandas, requests |
| Weather enrichment | Open-Meteo historical API, checkpointed grid-based fetching |
| Road enrichment | TomTom reverse geocoding, quota-prioritised and cached |
| Database | PostgreSQL 15+ (built/verified on PostgreSQL 18); PostGIS auto-detected and optional |
| Backend | FastAPI, SQLAlchemy + asyncpg |
| Frontend | Streamlit, multi-page (8 pages), Plotly + pydeck |
| Model training | scikit-learn (Random Forest), XGBoost, LightGBM |
| Experiment tracking / registry | MLflow (local file store) |
| Environment isolation | Separate virtualenv per component: `etl/`, `backend/`, `dashboard/`, `ml/` |

## 5. Data model

Star schema, built by `sql/schema.sql`:

- **`dim_date`** — date + hour, key format `YYYYMMDDHH`.
- **`dim_location`** — latitude/longitude, TomTom `street_name` (nullable —
  coverage is quota-limited), and a generated PostGIS `geography` column +
  GiST index **only when PostGIS is installed**; every table, view, endpoint,
  and dashboard page works without it since no query depends on a PostGIS
  function.
- **`dim_road`**, **`dim_weather`** — junk dimensions over STATS19's
  categorical codes.
- **`dim_severity`** — seeded: `1=Fatal, 2=Serious, 3=Slight, -1=Data missing
  or out of range` (STATS19's own missing-data code, kept as a labeled
  category rather than dropped).
- **`fact_accidents`** — one row per collision.
- **`fact_casualties`**, **`fact_vehicles`** — separate grain from
  `fact_accidents`: one collision can injure several people, each with their
  own severity, and involve several vehicles.
- **`etl_audit_log`** — source, row count, rejected count, status per
  pipeline run; mirrored to `etl/data/audit/audit_log.jsonl`.
- Ten `mart_*` materialized views back every dashboard page, refreshed at the
  end of every `load.py` run.

## 6. Pipeline workflow

1. **`extract.py`** — reads local STATS19 CSVs from `dataset/raw/`, snapshots
   them immutably, and enriches:
   - **Weather**: accidents snap to a spatial grid; one request per
     (grid cell, year) pulls a full year of hourly history. A full GB
     backfill is ~930 cell-year requests, more than one day's free quota, so
     extraction checkpoints every 25 cells and stops cleanly on a 429 —
     simply re-run once the quota resets and it resumes from the checkpoint.
   - **Road context**: TomTom calls are spent busiest-coordinate-first under
     `--road-limit`, caching every resolved coordinate so reruns never pay
     for the same point twice.
   - Every source/stage is appended to the audit log with row counts and
     status.
2. **`transform.py`** — drops duplicates, invalid/out-of-GB coordinates, and
   unparseable dates; decodes STATS19 codes into labels; joins weather (grid
   cell + exact hour) and road context (exact coordinate); restricts
   casualties/vehicles to accidents that survived cleaning; logs every
   rejection and reason.
3. **`load.py`** — bulk-loads staging parquet into the warehouse in one
   transaction (dimensions first, then facts upserted against them), commits,
   refreshes all mart views, and prints final row counts as pipeline-success
   evidence.
4. **Admin-triggered refresh** — `POST /admin/run-etl` (with `X-Admin-Token`)
   runs extract → transform → load as a background task using the `etl/`
   virtualenv's interpreter; `GET /admin/run-etl/status` reports progress.

Accidents whose grid cell or coordinate isn't yet enriched simply carry NULL
weather/street values rather than being dropped — coverage is reported
honestly (`/accidents/overview`, dashboard home page), not hidden.

## 7. Backend API (FastAPI, port 9001)

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness |
| `GET /accidents/overview` | Headline counts + real enrichment coverage % |
| `GET /accidents/severity-distribution` | Collisions per severity |
| `GET /accidents/by-hour` | Hour × weekday × severity |
| `GET /accidents/monthly-trend` | Monthly counts by severity |
| `GET /accidents/weather-impact` | Weather condition vs severity |
| `GET /accidents/road-type-severity` | Road type vs severity |
| `GET /accidents/hotspots` | Points, filterable by severity/date |
| `GET /accidents/top-locations` | Coordinates with most collisions |
| `GET /accidents/casualties` | Per-year casualty/vehicle totals |
| `GET /casualties/by-type` | Casualty type/class vs severity |
| `GET /casualties/demographics` | Age band and sex vs severity |
| `GET /vehicles/by-type` | Vehicle type vs severity |
| `POST /admin/run-etl` | Trigger a pipeline refresh (`X-Admin-Token` required) |
| `GET /admin/run-etl/status` | Last/current run status |
| `POST /predict` | Predicted severity + probabilities |
| `GET /model/info` | Served model version, stage, training metrics |
| `GET /model/health` | Whether a model is currently loaded |
| `GET /monitoring/drift` | Feature/label drift vs training baseline |
| `GET /monitoring/performance` | Rolling recall/macro-F1/false-negative rate |
| `GET /monitoring/latency` | Serving latency (p50/p95) and failure rate |

Data endpoints return `503` with a readable message until `DATABASE_URL` is
configured and reachable; `/predict` and `/model/*` return `503` until a
Production model is loaded.

## 8. Dashboard (Streamlit, 8 pages)

| Page | Contents |
|---|---|
| `Home.py` | KPI strip (from `/accidents/overview`), navigation, honest enrichment-coverage callout |
| `1_Severity_Distribution.py` | Severity split, interactive |
| `2_Time_Patterns.py` | Hour × weekday heatmap, monthly trend |
| `3_Weather_Impact.py` | Weather condition vs severity |
| `4_Road_Type_Severity.py` | Road type vs severity |
| `5_Hotspot_Map.py` | pydeck map, live severity/date filters |
| `6_Casualties_Vehicles.py` | Casualty type, demographics, vehicle type |
| `7_Severity_Predictor.py` | Form → `POST /predict` → predicted class + probabilities |
| `8_Model_Monitoring.py` | Drift and performance charts from `/monitoring/*` |

Every page queries the live API — no bundled or placeholder data — and a
panel with no rows says so rather than rendering empty.

## 9. Part 2 — MLOps

- **Models**: Random Forest, XGBoost, LightGBM, trained with class weighting
  to address severity-class imbalance (fatal collisions are rare).
- **Split**: chronological — earlier years train, most recent years
  validate/test — since the model predicts future accidents, not
  exchangeable historical rows.
- **Tracking**: every run's parameters, metrics, and artifacts logged to
  MLflow; the macro-F1 winner is registered as `traffic_severity_classifier`
  and staged `Production`.
- **Evaluation**: `evaluate.py` produces recall, macro-F1, and a confusion
  matrix as MLflow artifacts.
- **Serving**: the backend loads the Production model at startup via a
  `lifespan` handler; restart the backend after training so it picks up a
  newly promoted model.
- **Monitoring**: `/monitoring/drift` and `/monitoring/performance` compare
  live traffic against the training baseline.
- **Retraining criteria**: documented concretely in `ml/RETRAINING.md`
  (macro-F1 drop, false-negative rate, and feature drift PSI thresholds).

## 10. Setup and run order

```bash
cp .env.example .env          # DATABASE_URL, TOMTOM_API_KEY, ADMIN_TOKEN

# Special characters in the DB password must be URL-encoded in DATABASE_URL
# (an "@" becomes %40, or the driver misreads the hostname).

# 1. Database — schema.sql enables PostGIS itself if it's installed, skips it otherwise
createdb -U postgres traffic_accidents
psql -U postgres -d traffic_accidents -f sql/schema.sql
# Windows: use the full path to psql/createdb under the PostgreSQL install dir if not on PATH.

# 2. Data — dataset/raw/collisions.csv, casualties.csv, vehicles.csv already in place

# 3. Pipeline (own virtualenv)
cd etl
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python extract.py         # --skip-road / --road-limit N as needed
.venv/Scripts/python transform.py
.venv/Scripts/python load.py            # final row counts = pipeline-success screenshot

# 4. API (own virtualenv)
cd ../backend
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m uvicorn app.main:app --reload --port 9001

# 5. Dashboard (own virtualenv)
cd ../dashboard
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/streamlit run Home.py

# 6. Model training (own virtualenv)
cd ../ml
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python train.py
.venv/Scripts/python evaluate.py
.venv/Scripts/mlflow ui --port 5000
```

### Docker (containerized inference API, optional)

`backend/Dockerfile` builds *only* the FastAPI inference/analytics service —
Postgres and Streamlit stay native. Build from the repo root (it needs to
copy `ml/` and `etl/stats19_lookups.py` alongside `backend/app`):

```bash
docker build -f backend/Dockerfile -t traffic-severity-api .
```

The image doesn't bake in the MLflow model store (`ml/mlruns/` is in
`.dockerignore` on purpose, so the image stays small and always serves
whatever is currently registered `Production`). Run it with `DATABASE_URL`
pointed at the host's Postgres via `host.docker.internal` (Docker Desktop's
bridge to the host), and `ml/mlruns` mounted in:

```bash
docker run --rm -p 9000:9000 \
  --env-file .env \
  -e DATABASE_URL=postgresql://user:pass@host.docker.internal:5432/traffic_accidents \
  -e MLFLOW_TRACKING_URI="file:///C:/Users/PARAM/OneDrive/Desktop/MLOPS1/ml/mlruns" \
  --mount type=bind,source="%cd%/ml/mlruns",target=/C:/Users/PARAM/OneDrive/Desktop/MLOPS1/ml/mlruns \
  traffic-severity-api
# http://127.0.0.1:9000/health, /model/info, /docs
```

The mount target and `MLFLOW_TRACKING_URI` above look unusual — they
deliberately repeat this machine's own absolute path. That's not
cosmetic: MLflow's local file store records each run's artifact location
as an **absolute path** in its `meta.yaml` (e.g.
`/C:/Users/PARAM/OneDrive/Desktop/MLOPS1/ml/mlruns/<exp>/<run>/artifacts/model`)
at training time, not a relative one — so mounting the same directory at a
different container path (e.g. `/srv/ml/mlruns`) lets MLflow find the run,
but then fails to load its model artifact, because the path baked into
that run's metadata doesn't exist inside the container. Mounting at the
identical absolute path sidesteps that entirely. This was hit and fixed
while producing the container run-test evidence below — adjust the path to
match wherever you clone this repo.

Verified working (2026-09-08): built `traffic-severity-api:latest`, ran it
against the host Postgres with `ml/mlruns` mounted as above, and confirmed
`/health`, `/model/info` (`RandomForest`, Production v2, macro-F1 0.378),
`/model/health`, a real `/predict` call, and `/accidents/overview` (mart
read through the host DB) all returned correctly from inside the
container.

### Airflow (optional orchestration)

`airflow/dags/traffic_accident_etl_dag.py` schedules the same
`extract.py → transform.py → load.py → quality_check.py` scripts — it
doesn't reimplement the pipeline. Airflow has no native Windows support, so
it needs WSL2 or a container; see `airflow/README.md` for the full setup and
the `traffic_etl_python` / `traffic_project_root` / `traffic_road_limit`
Variables it reads.

Actually run (2026-09-08) inside a fresh WSL2 Ubuntu distro: installed
`apache-airflow` (resolved to **3.3.1** — this Ubuntu image ships only
Python 3.14, and Airflow installed and ran on it cleanly despite that being
a very new interpreter). Symlinked the DAG into `$AIRFLOW_HOME/dags`, set
the three Variables (`traffic_etl_python` pointed at the existing
`etl/.venv/Scripts/python.exe`, invoked through WSL2's Windows-binary
interop rather than reinstalling ETL dependencies inside Linux), started
`airflow standalone`, confirmed `/api/v2/monitor/health` reported the
metastore/scheduler/triggerer/dag-processor all `healthy`, unpaused the DAG,
and triggered a real run — `extract` executed against the live Open-Meteo
API, not a stub.

Two environment gotchas worth keeping if you redo this:
- **`PATH` must be set explicitly.** A non-login `wsl -e bash -c "..."`
  invocation doesn't reliably inherit a usable `PATH` in every calling
  context — `airflow standalone`'s own subprocess spawns (`airflow ...`,
  `bash -c ...`) failed with `FileNotFoundError` until `PATH` was set to an
  explicit, complete value covering both `/root/airflow-venv/bin` and the
  standard `/usr/bin` etc. directories, rather than prepending to `$PATH`
  and trusting it was already populated.
- **From Git Bash specifically**, MSYS's automatic path conversion mangles
  bare Linux paths like `/root/airflow` into a Windows path before `wsl.exe`
  ever sees them — prefix such commands with `MSYS_NO_PATHCONV=1`.

Airflow 3.x also moved `BashOperator` into
`apache-airflow-providers-standard`; the DAG's `from airflow.operators.bash
import BashOperator` still works via a deprecation-warning back-compat
shim, so no DAG code changes were needed.

## 11. Known data-quality notes and limits

- TomTom's free tier can't cover every accident coordinate; coverage is
  reported honestly (`/accidents/overview`, dashboard home page), not hidden.
- STATS19's own `-1 = data missing or out of range` code is preserved as a
  labeled category everywhere, never silently dropped.
- Open-Meteo's free quota requires checkpointed, resumable extraction rather
  than a single run — a normal, expected part of using free-tier enrichment.
- Every rejected row during `transform.py` is logged with a reason to
  `etl/data/audit/audit_log.jsonl` and the `etl_audit_log` table.

## 12. Assignment submission mapping

| Required item | Comes from |
|---|---|
| Source code and ETL scripts | `etl/`, `backend/`, `ml/` |
| Orchestration workflow | `extract.py → transform.py → load.py`, triggerable via `/admin/run-etl` or the Airflow DAG (`airflow/`) |
| Database schema and sample tables | `sql/schema.sql` + a populated-table screenshot |
| Dataset source info | Section 2 |
| Architecture diagram | Section 3, exported as `docs/architecture-diagram.svg` for the report |
| Data dictionary and validation rules | `docs/DATA_DICTIONARY.md` (generated from `sql/schema.sql`) + rejection rules (Section 6) |
| Dashboard (5+ views) | Section 8 — 8 pages |
| Report (8–10 pages) | Written separately, referencing all sections above |
| Screenshots / execution evidence | `load.py`'s final row-count output, a working `/predict` call, a dashboard page |
| README with setup instructions | Section 10 |

## 13. Repository structure

```
.
├── dataset/raw/                # STATS19 CSVs
├── etl/                        # extract.py, transform.py, load.py, quality_check.py, stats19_lookups.py
├── backend/app/                # FastAPI app, routers, db.py, models.py
├── backend/Dockerfile          # containerizes only the inference API
├── dashboard/                  # Home.py + 8-page multi-page app
├── ml/                         # features.py, train.py, evaluate.py, RETRAINING.md
├── sql/schema.sql
├── airflow/dags/                # optional Airflow DAG wrapping the ETL scripts
├── docs/                        # DATA_DICTIONARY.md, architecture-diagram.svg
├── .dockerignore
├── .env.example
├── CLAUDE.md                   # full build spec
└── README.md                   # this file
```
