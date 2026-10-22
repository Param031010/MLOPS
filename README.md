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
  end of every `load.py` run. Seven of them carry a `year` column so the API
  can filter by year range (and severity) server-side; `mart_weather_impact`
  stores sums + non-null counts instead of averages so a multi-year average
  is computed correctly rather than as an average of averages.
- **`vw_model_features`** — a plain (non-materialized) view: one row per
  collision with every field the severity classifier consumes, resolved
  through the dimensions. `ml/train.py` reads exactly this object.

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
   rejection and reason, and writes each dropped collision (with its
   rejection reason) to `etl/data/audit/rejected_records.csv`.
   - **Junction-detail reconciliation**: the DfT "last 5 years" export carries
     both a historic `junction_detail_historic` column and a renumbered
     `junction_detail`. `stats19_lookups.combine_junction_detail` maps the new
     codes onto the historic labels (verified by cross-tabulating the raw
     columns), and `9`/`99` "unknown (self reported)" codes are labeled — this
     fixed ~45% of collisions that previously decoded as "Unknown code N".
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
| `GET /monitoring/data-quality` | Missing / out-of-range / unseen-category rates, warehouse window + live `/predict` inputs |

**Server-side filters.** The mart-backed endpoints (`/accidents/by-hour`,
`monthly-trend`, `weather-impact`, `road-type-severity`, `/casualties/by-type`,
`/casualties/demographics`, `/vehicles/by-type`) accept repeatable
`severity`, plus `year_from` / `year_to`; `road-type-severity` also accepts
repeatable `road_type`. All values are bound parameters. The dashboard's
sidebar filters re-query these, so filtering happens in Postgres, not in the
browser.

Data endpoints return `503` with a readable message until `DATABASE_URL` is
configured and reachable; `/predict` and `/model/*` return `503` until a
Production model is loaded.

**Monitoring windows.** `/monitoring/drift` scores the ~100,000 most recent
accidents (about one year, so seasonal weather/month shifts don't read as
drift against the multi-year baseline); `/monitoring/performance` scores the
1,000 most recent; `/monitoring/data-quality` checks the 2,000 newest
warehouse rows plus every live `/predict` input since the backend started.
Latency, failure and input-quality counters are in-process and reset on
restart.

## 8. Dashboard (Streamlit, 4 pages)

| Page | Contents |
|---|---|
| `Home.py` | Overview: KPI strip, severity mix, hour × weekday heatmap, monthly trend, enrichment-coverage expander |
| `1_Risk_Factors.py` | Tabs: weather vs severity, road type vs severity, casualty age/sex and vehicle type |
| `2_Hotspot_Map.py` | pydeck map with live severity/date filters, top locations in an expander |
| `3_Severity_Model.py` | Tabs: prediction form → `POST /predict` with probabilities; monitoring (rolling performance, PSI drift, latency, input-data quality) |

Home and Risk Factors share one sidebar (`api_client.sidebar_filters`: year
range + severity) that is passed to the API as query parameters. Every page
queries the live API — no bundled or placeholder data — and a panel with no
rows says so rather than rendering empty.

## 9. Part 2 — MLOps

- **Data**: `ml/train.py` reads `vw_model_features` (one row per collision).
- **Models**: Random Forest, XGBoost, LightGBM, trained with class weighting
  to address severity-class imbalance (fatal collisions are rare).
- **Split**: chronological 70/15/15 — earlier years train, most recent years
  validate/test — since the model predicts future accidents, not
  exchangeable historical rows.
- **Tracking**: every run's parameters, metrics, and artifacts are logged to
  the MLflow experiment `traffic-accident-severity`; the validation-macro-F1
  winner is registered as `traffic_severity_classifier`, staged
  `Production`, and the previous Production version is archived.
- **Packaged inference artifact**: the winner is wrapped in
  `ml/pipeline.py`'s `SeverityPipeline`, so the registered model takes *raw*
  accident records and does its own feature engineering via
  `ml/features.py`. Training, `/predict` and the Docker image therefore can't
  be fed differently-preprocessed data. A model registered before this change
  is detected as "legacy" and still served (the API encodes for it), with a
  warning to retrain.
- **Evaluation**: `ml/reporting.py` (shared by `train.py` and `evaluate.py`)
  logs macro-F1, macro and per-class recall, the Fatal/Serious
  false-negative rate, a confusion-matrix PNG + CSV and the classification
  report for every candidate run. `evaluate.py` re-scores the current
  Production model on the held-out test slice as a separate MLflow run.
- **Baseline**: `ml/artifacts/baseline.json` stores feature and location
  decile profiles, the class mix and missing rates from the training split;
  the monitoring endpoints compare live data against it.
- **Serving**: the backend loads the Production model at startup via a
  `lifespan` handler; restart the backend after training so it picks up a
  newly promoted model. Serving warns if the lookup vocabularies changed
  since training.
- **Monitoring**: `/monitoring/drift` (feature, location, class PSI),
  `/monitoring/performance`, `/monitoring/data-quality` (missing,
  out-of-range, unseen-category rates), and `/monitoring/latency`.
- **Retraining criteria**: documented concretely in `ml/RETRAINING.md`
  (macro-F1 drop, false-negative rate, PSI > 0.2, input-data quality, new
  data volume), along with the model lifecycle and rollback procedure.
- **Current model**: Random Forest, registered version 4 (trained
  2026-10-11), test macro-F1 ≈ 0.36, macro recall ≈ 0.46. The classes are
  heavily imbalanced (Slight ≈ 76%, Serious ≈ 22%, Fatal ≈ 1.5% of
  training rows); exact figures are in `ml/artifacts/baseline.json`.

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
.venv/Scripts/python quality_check.py   # post-load gate: exit 0 = warehouse healthy

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

Verified working (2026-09-08, against the model registered at that time —
Production v2, macro-F1 0.378): built `traffic-severity-api:latest`, ran it
against the host Postgres with `ml/mlruns` mounted as above, and confirmed
`/health`, `/model/info`, `/model/health`, a real `/predict` call, and
`/accidents/overview` (mart read through the host DB) all returned correctly
from inside the container. The API, packaged-pipeline model (v4) and
`/monitoring/data-quality` added since have not been re-run in the container;
rebuild the image and repeat the checks before using it as fresh evidence.

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
├── ml/                         # features.py, pipeline.py, reporting.py, train.py, evaluate.py, RETRAINING.md
├── sql/schema.sql              # star schema, 10 marts, vw_model_features
├── airflow/                    # optional Airflow DAG wrapping the ETL scripts + WSL2 README
├── docs/                       # DATA_DICTIONARY.md, REPORT_OUTLINE.md, architecture-diagram.svg, generate_data_dictionary.py
├── .dockerignore
├── .env.example
├── DETAILED_README.md          # function- and column-level technical reference
└── README.md                   # this file
```
