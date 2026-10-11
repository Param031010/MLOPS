# Detailed Technical Reference

Companion to [README.md](README.md) (setup/run instructions) and
[CLAUDE.md](CLAUDE.md) (build spec). This document is a function- and
column-level reference: what every module actually does, every endpoint's
exact request/response shape, and every warehouse column's meaning —
written from the code as it stands, so it can be read instead of opening
each file. Verified against the repository on 2026-09-11.

## Table of contents

1. [Repository map](#1-repository-map)
2. [ETL pipeline (`etl/`)](#2-etl-pipeline-etl)
3. [Database schema (`sql/schema.sql`)](#3-database-schema-sqlschemasql)
4. [Backend (`backend/app/`)](#4-backend-backendapp)
5. [MLOps (`ml/`)](#5-mlops-ml)
6. [Dashboard (`dashboard/`)](#6-dashboard-dashboard)

---

## 1. Repository map

```
etl/
  audit.py             shared JSONL audit logger
  stats19_lookups.py   STATS19 code -> label dictionaries (31 dicts) + decode()/decode_series()
  extract.py           reads dataset/raw/*.csv, snapshots, enriches (weather + road)
  transform.py         cleans, decodes, joins -> staging parquet
  load.py              bulk-loads warehouse, upserts facts, refreshes marts
  quality_check.py     post-load gate (used by CLI, Airflow, CI)
  data/{raw,staging,audit}/   created at runtime

backend/app/
  main.py              FastAPI app, lifespan model load, CORS
  db.py                lazy async SQLAlchemy engine
  models.py            ORM mirror of sql/schema.sql (reference only, not queried at runtime)
  schemas.py           Pydantic response/request models
  routers/
    accidents.py  casualties.py  vehicles.py  admin.py  predict.py

ml/
  features.py          shared feature engineering (train.py + predict.py import this)
  train.py             trains RF/XGBoost/LightGBM, tracks + registers in MLflow
  evaluate.py           scores Production model, logs confusion matrix
  artifacts/           baseline.json, confusion_matrix.png (generated)
  mlruns/              MLflow local file store (generated)
  RETRAINING.md        documented retraining thresholds

dashboard/
  api_client.py        cached GET / uncached POST wrapper around the backend
  Home.py               KPI strip + navigation
  pages/1..8_*.py       one page per analytics view + predictor + monitoring

sql/schema.sql          star schema + 10 mart materialized views + column comments
airflow/dags/traffic_accident_etl_dag.py   orchestrates extract->transform->load->quality_check
docs/generate_data_dictionary.py           emits docs/DATA_DICTIONARY.md from the live schema
backend/Dockerfile      containerizes only the inference/analytics API
```

---

## 2. ETL pipeline (`etl/`)

### 2.1 `audit.py`

`log_event(run_id, source, stage, status, row_count=None, rejected_count=None, message="", started_at=None)`
appends one JSON line to `etl/data/audit/audit_log.jsonl` with
`{run_id, source, stage, row_count, rejected_count, status, message, started_at, finished_at}`.
Every stage in `extract.py`/`transform.py`/`load.py` calls this once per
source it touches, so the file is a complete, replayable trail independent
of the `etl_audit_log` Postgres table (which mirrors the same events).

### 2.2 `stats19_lookups.py`

31 `{code: label}` dictionaries decoding STATS19's numeric fields (severity,
day of week, urban/rural, road type, junction detail/control, pedestrian
crossing, light/weather/road-surface conditions, road class, casualty
class/type, sex, age band, pedestrian location/movement, car/bus passenger
status, vehicle type, towing, manoeuvre, junction location, skidding,
first point of impact, propulsion, journey purpose, left-hand drive, etc.).

Convention, enforced by `decode()`/`decode_series()`:
- `-1` always decodes to `"Data missing"` (or a field-specific missing
  string) rather than being dropped.
- A code with no entry in the dictionary decodes to `"Unknown code {n}"`
  rather than raising — an unexpected/rare STATS19 code never crashes the
  pipeline.

### 2.3 `extract.py`

Constants: `GRID_STEP = 0.5` (degrees), `CHECKPOINT_EVERY = 25` (cells),
`SNAPSHOT_RETENTION = 2` (snapshots kept per source, ~250MB each).

| Function | Behavior |
|---|---|
| `validate_and_read(filename)` | Reads `dataset/raw/{filename}`; `SystemExit` with a clear message if the file is missing or missing a required column (`REQUIRED_COLUMNS` per file). |
| `snapshot(df, name)` / `prune_snapshots(name, keep=2)` | Writes a timestamped immutable CSV to `etl/data/raw/`, then deletes all but the newest `keep` per source. |
| `snap_to_grid(value)` | `round(value / 0.5) * 0.5` — same function transform.py reuses for the weather join. |
| `fetch_weather_year(grid_lat, grid_lon, year)` | One Open-Meteo `archive` API call for a full year of hourly `temperature_2m, precipitation, windspeed_10m, weathercode`, timezone `Europe/London`. Retried up to 4 times with exponential backoff (2–30s) via `tenacity`, **except** on a 429 (raises `RateLimited`, which is never retried — hammering an already-closed quota window wastes nothing but time). |
| `extract_weather(collisions, run_id)` | Dedupes accidents to `(grid_lat, grid_lon, year)` cells, skips cells already in `weather_grid.parquet`, fetches the rest, checkpoints to parquet every 25 cells (and once more at the end) so a rate-limited or interrupted run resumes exactly where it left off. Logs a `partial` audit status when rate-limited, `success` otherwise. |
| `reverse_geocode(lat, lon, api_key)` | One TomTom reverse-geocode call; returns `streetName` or falls back to `municipality`; retried up to 3 times (1–10s backoff); a 403 raises immediately (quota/key issue, not transient). |
| `extract_road_context(collisions, run_id, limit)` | Ranks distinct coordinates by accident count (busiest first), skips coordinates already resolved in `road_context.parquet`, resolves up to `limit` new ones, stops early if a failure looks quota-related. No-ops with a `success`/`0 rows` audit entry if `TOMTOM_API_KEY` is unset. |
| `main()` | CLI: `--skip-road`, `--road-limit N` (default 300). Generates a 12-hex-char `run_id`, snapshots all three CSVs, then runs weather and (unless skipped) road-context extraction. |

### 2.4 `transform.py`

Constants: `GB_LAT_RANGE = (49.8, 60.9)`, `GB_LON_RANGE = (-8.6, 1.8)`.

| Function | Behavior |
|---|---|
| `latest_snapshot(name)` | Picks the most recent timestamped snapshot for a source; `SystemExit` if none exists. |
| `clean_collisions(df, run_id)` | Drops, in order: duplicate `collision_index` (keep first), missing lat/lon, coordinates outside the GB bounding box, rows whose `date`/`time` don't parse (`dayfirst=True`, `time` format `%H:%M`). Every rejection reason and count is logged to the audit trail as a single dict. |
| `decode_collisions(df)` | Builds the full staged-accident row set: `date_key = YYYYMMDD*100 + hour`, `day_of_week`/`day_name`/`is_weekend` (STATS19 numbering, 1=Sunday, weekend = {1,7}), decodes every categorical STATS19 field via `stats19_lookups`, computes `grid_lat`/`grid_lon` for the weather join (dropped again before staging is written). |
| `join_weather(staged, run_id)` | Left-merges `weather_grid.parquet` on **grid cell and exact hour** (`grid_lat, grid_lon, full_date=date, hour`) — not just date. If the parquet doesn't exist yet, weather columns are all `NULL` rather than failing. |
| `join_road_context(staged, run_id)` | Left-merges `road_context.parquet` on exact `(latitude, longitude)`. |
| `finalize_weather_flags(staged)` | Derives `is_raining = precipitation_mm.fillna(0) > 0`, `is_freezing = temperature_c <= 0`. |
| `transform_casualties(raw, valid_accident_index, run_id)` / `transform_vehicles(...)` | Restrict to `accident_index` values that survived `clean_collisions`, decode every categorical field, log rows dropped as "belonged to a cleaned-out accident". |
| `main()` | Reads the latest snapshot of each source, runs the pipeline above, writes `etl/data/staging/{accidents,casualties,vehicles}_staged.parquet`. |

### 2.5 `load.py`

Two lookup tables of note:
- `NUMERIC_DIM_SCALES` — columns whose dimension-join comparison must be
  rounded to the same decimal scale the dimension table stores
  (`speed_limit`→0dp, `weather_code_meteo`→0dp, `temperature_c`→2dp,
  `precipitation_mm`→2dp, `wind_speed_kmh`→2dp), otherwise an unrounded
  staged value fails to match its own freshly-inserted dimension row and the
  accident is silently dropped by the inner join.
- `dimension_join(staged_alias, dim_alias, cols)` builds an `ON` clause using
  `COALESCE(x, sentinel) = COALESCE(y, sentinel)` per column rather than
  `IS NOT DISTINCT FROM` — the latter isn't a hashable equality operator, so
  Postgres would nested-loop the whole dimension instead of hash-joining it
  (measured as the difference between finishing and not finishing at ~514k
  rows).

Load sequence, all in **one transaction**:
1. `SET LOCAL work_mem = '256MB'`, `SET LOCAL maintenance_work_mem = '512MB'`
   (session-local; without this the transaction spills to disk — see
   [README.md §"Known constraints"] and the load-time notes below).
2. `bulk_load()` COPYs each staging parquet into a `tmp_*_staged` table via
   `cursor.copy_expert` (row-by-row `to_sql` would take minutes on the
   ~937k-row vehicles table; COPY takes seconds).
3. Populate `dim_date`, `dim_road`, `dim_weather` via
   `INSERT ... SELECT DISTINCT ... ON CONFLICT DO NOTHING`.
4. Populate `dim_location` via `ON CONFLICT ... DO UPDATE` (not `DO NOTHING`)
   so a later run's newly-resolved TomTom street names actually land —
   `DO NOTHING` would freeze road-context coverage at whatever the first
   load happened to resolve. `COALESCE(EXCLUDED.street_name, dim_location.street_name)`
   keeps an existing name when a rerun has nothing new for that coordinate.
5. Upsert `fact_accidents` (`ON CONFLICT (accident_index) DO UPDATE`),
   resolving `location_id`/`road_id`/`weather_id` via inner joins to the
   dimensions. Because these are inner joins, a staged row that fails to
   resolve a dimension would silently vanish — so the script re-counts
   staged vs. resolved rows immediately after and **raises (rolling back the
   whole transaction) if they don't match**.
6. Delete-and-reinsert `fact_casualties`/`fact_vehicles` for every
   `accident_index` present in this run's staging data.
7. Drop the temp tables, record final counts into `etl_audit_log`.
8. **After** commit: `ANALYZE` every dimension/fact table (a freshly
   COPY'd table has no planner statistics) and `REFRESH MATERIALIZED VIEW`
   each of the 10 marts **individually**, each in its own autocommit
   connection — refreshing all ten inside one transaction would hold
   `ACCESS EXCLUSIVE` on every mart until the whole batch finished, blocking
   every API read for that entire window; per-view commits release each
   lock as soon as that view is done.

Measured performance on this project's data (513,748 accidents): full load
~3m30s, mart refresh ~9–14s, versus **hours** at Postgres' stock `work_mem`
(4MB) — a single run at the defaults was seen spilling ~1.4–1.6GB across
~380–520 temp files.

### 2.6 `quality_check.py`

Exit code 0/1 gate, run after `load.py` (by hand, by the Airflow DAG's
`data_quality_check` task, or in CI). Fails if any of:
- `fact_accidents`, `fact_casualties`, `fact_vehicles`, or any dimension
  table is empty.
- No row exists in `etl_audit_log` for `stage='load'`, or the most recent
  one isn't `status='success'`.
- Any `fact_accidents` row has no matching `dim_date` row (orphan check —
  should be structurally impossible given the load's inner joins, so a hit
  here means the warehouse was modified outside the pipeline).
- Any `mart_*` materialized view is empty.

---

## 3. Database schema (`sql/schema.sql`)

PostgreSQL 15+; PostGIS is auto-detected (`CREATE EXTENSION IF NOT EXISTS
postgis` wrapped in a `DO` block that swallows the error if unavailable) —
`dim_location.geog` (a generated, stored `GEOGRAPHY(Point,4326)` column with
a GiST index) is only added when the extension is present, and nothing else
in the schema, marts, or API depends on it. `dim_road`/`dim_weather`'s
dedupe `UNIQUE` constraints are `NULLS NOT DISTINCT` (Postgres 15+) — needed
because most weather/road fields are `NULL` for unenriched accidents, and
the SQL-standard "NULLs are never equal" behavior would let `load.py`
re-insert an identical-looking NULL-heavy dimension row on every rerun,
which then breaks the fact upsert (`ON CONFLICT DO UPDATE command cannot
affect row a second time`) once two such duplicate rows both match the same
staged row through `dimension_join`'s `COALESCE`.

### Dimensions

**`dim_severity`** — seeded, 4 rows.

| Column | Meaning |
|---|---|
| `severity_id` (PK) | STATS19 code: `1`=Fatal, `2`=Serious, `3`=Slight, `-1`=Data missing. |
| `severity_label` | Human-readable label. |

**`dim_date`** — one row per distinct hour a collision occurred.

| Column | Meaning |
|---|---|
| `date_key` (PK) | `YYYYMMDDHH`, e.g. `2025063017` = 17:00 on 30 June 2025. |
| `full_date` | Calendar date. |
| `hour` | 0–23. |
| `day_of_week` | STATS19 numbering: 1=Sunday … 7=Saturday. |
| `day_name`, `day_of_month`, `month`, `month_name`, `quarter`, `year` | Decoded/derived calendar fields. |
| `is_weekend` | True for Saturday/Sunday (codes 7 and 1). |

**`dim_location`** — one row per distinct `(latitude, longitude)`.

| Column | Meaning |
|---|---|
| `location_id` (PK) | Surrogate key. |
| `latitude`, `longitude` | `NUMERIC(9,6)`, WGS84, validated against the GB bounding box. |
| `location_easting_osgr`, `location_northing_osgr` | OS grid reference, as supplied. |
| `lsoa_of_accident_location` | LSOA code. |
| `local_authority_district`, `local_authority_ons_district`, `local_authority_highway`, `police_force` | Administrative context. |
| `urban_or_rural_area` | Decoded: Urban / Rural / Unallocated / Data missing. |
| `street_name` | TomTom reverse-geocoded street; `NULL` until quota reaches this coordinate. |
| `road_context_resolved_at` | When `street_name` was last set. |
| `geog` *(PostGIS only)* | Generated `GEOGRAPHY(Point,4326)`, GiST-indexed. |

**`dim_road`** — junk dimension, deduped on the full column tuple.

| Column | Meaning |
|---|---|
| `road_id` (PK) | Surrogate key. |
| `first_road_class`, `second_road_class` | Decoded road class (Motorway, A(M), A, B, C, Unclassified). |
| `road_type` | Decoded carriageway layout. |
| `speed_limit` | Posted mph limit. |
| `junction_detail`, `junction_control` | Decoded junction type/control. |
| `pedestrian_crossing` | Decoded crossing facility within 50m. |
| `special_conditions_at_site`, `carriageway_hazards` | Decoded site defects/hazards. |
| `trunk_road_flag` | Trunk road or not. |

**`dim_weather`** — junk dimension combining STATS19's police-observed
fields with Open-Meteo measurements, deduped on the full tuple.

| Column | Meaning |
|---|---|
| `weather_id` (PK) | Surrogate key. |
| `weather_conditions`, `road_surface_conditions`, `light_conditions` | Decoded STATS19 fields. |
| `temperature_c`, `precipitation_mm`, `wind_speed_kmh`, `weather_code_meteo` | Open-Meteo values for the collision hour/grid cell; `NULL` where not yet backfilled. |
| `is_raining`, `is_freezing` | Derived booleans. |

### Facts

**`fact_accidents`** — one row per collision.

| Column | Meaning |
|---|---|
| `accident_key` (PK) | Surrogate key. |
| `accident_index` (unique) | Business key (source `collision_index`); the upsert's conflict target. |
| `date_key`, `location_id`, `road_id`, `weather_id`, `severity_id` | FKs to the dimensions above. |
| `number_of_vehicles`, `number_of_casualties` | Collision-level counts. |
| `police_force`, `did_police_officer_attend` | Reporting context. |
| `source_year` | Year of the source STATS19 export. |
| `loaded_at` | When `load.py` wrote the row. |

**`fact_casualties`** — one row per injured person (separate grain from
`fact_accidents`: one collision can injure several people, each with their
own severity).

| Column | Meaning |
|---|---|
| `casualty_key` (PK), `accident_index` (FK) | Keys. |
| `vehicle_reference`, `casualty_reference` | Position within the collision. |
| `casualty_class` | Driver or rider / Passenger / Pedestrian. |
| `sex_of_casualty`, `age_of_casualty`, `age_band_of_casualty` | Demographics. |
| `casualty_severity_id` (FK to `dim_severity`) | This person's injury severity — may differ from the collision's overall severity. |
| `pedestrian_location`, `pedestrian_movement` | Decoded, pedestrians only. |
| `car_passenger`, `bus_or_coach_passenger` | Decoded passenger position. |
| `casualty_type` | Decoded road-user type (Car occupant, Pedestrian, Cyclist, ...). |
| `casualty_imd_decile` | Index of Multiple Deprivation decile of the casualty's home area. |

**`fact_vehicles`** — one row per vehicle involved.

| Column | Meaning |
|---|---|
| `vehicle_key` (PK), `accident_index` (FK) | Keys. |
| `vehicle_reference` | Position within the collision. |
| `vehicle_type` | Decoded (Car, Van, Bus/coach, Motorcycle by engine size, ...). |
| `towing_and_articulation`, `vehicle_manoeuvre`, `junction_location`, `skidding_and_overturning`, `first_point_of_impact` | Decoded collision dynamics. |
| `sex_of_driver`, `age_of_driver`, `age_band_of_driver` | Driver demographics. |
| `engine_capacity_cc`, `propulsion_code`, `age_of_vehicle` | Vehicle attributes. |
| `journey_purpose` | Decoded journey purpose. |

**`etl_audit_log`** — one row per source+stage of every pipeline run
(mirrors `etl/data/audit/audit_log.jsonl`).

| Column | Meaning |
|---|---|
| `audit_id` (PK), `run_id` | Keys; every row from one run shares `run_id`. |
| `source` | `collisions`\|`casualties`\|`vehicles`\|`weather`\|`road_context`\|`load`. |
| `stage` | `extract`\|`transform`\|`load`. |
| `row_count`, `rejected_count` | Rows accepted / rejected by the stage. |
| `status` | `success`\|`partial`\|`failed`. |
| `message`, `started_at`, `finished_at` | Detail (including per-reason rejection breakdown) and timing. |

### Marts (`mart_*`, materialized views, refreshed at the end of every load)

| View | Grain / purpose |
|---|---|
| `mart_severity_distribution` | Accidents + total casualties per severity. |
| `mart_by_hour_weekday` | Accidents per (hour, day_of_week, severity). |
| `mart_monthly_trend` | Accidents/casualties/vehicles per (year, month, severity). |
| `mart_weather_impact` | Accidents + avg temperature/precipitation per (weather_conditions, severity). |
| `mart_road_type_severity` | Accidents per (road_type, severity). |
| `mart_hotspots` | Row-per-accident view (index, lat/lon, street, date, severity) — backs the map and `/accidents/hotspots`. |
| `mart_top_locations` | Accidents/casualties/fatal count per distinct coordinate, ranked. |
| `mart_casualties_by_type` | Casualties per (casualty_type, casualty_class, severity). |
| `mart_casualty_demographics` | Casualties per (age_band, sex, severity). |
| `mart_vehicles_by_type` | Vehicles per (vehicle_type, severity). |

`refresh_marts()` (plpgsql function) refreshes all ten in sequence — used
as the reference implementation `load.py` mirrors with per-view autocommit
instead of calling this function directly (so a slow refresh doesn't hold
every view's lock at once).

---

## 4. Backend (`backend/app/`)

### 4.1 `main.py`

Loads `.env`, imports routers (which import `predict.py`, which loads
`ml/features.py` — see §4.5), assembles `FastAPI(lifespan=lifespan)` where
`lifespan` calls `predict.load_model()` once at startup. CORS is
permissive (`allow_origins=["*"]`) for local dev. `GET /health` returns
`{"status": "ok"}` unconditionally (liveness only — it does not check the
database or model).

### 4.2 `db.py`

`get_engine()` builds the async engine lazily, once, from `DATABASE_URL`
(rewriting `postgresql://` to `postgresql+asyncpg://`); returns `None` if
unset, so the module never fails at import time. `get_connection()` is the
FastAPI dependency every data router uses — it raises `HTTPException(503,
"DATABASE_URL is not configured on the server.")` if there's no engine,
otherwise yields a connection. `fetch_all()`/`fetch_one()` wrap
`conn.execute(text(sql), params)` and return plain dicts.

### 4.3 `models.py`

SQLAlchemy ORM classes mirroring every table in §3 exactly (column names,
types, FKs). **Not used by any router at runtime** — the data endpoints
read the `mart_*` views with raw SQL (a better fit for read-only aggregates
than an ORM), and `ml/train.py` pulls training data with one hand-written
join. Kept as the authoritative typed reference for column names/types, and
as the starting point if a write-path or per-row entity API is ever added.

### 4.4 `schemas.py`

One Pydantic model per endpoint response shape (see the field lists in
§4.5's tables), plus `PredictRequest`/`PredictResponse`, `ModelInfo`,
`ModelHealth`, `DriftReport`, `LatencyReport`, `PerformanceReport`, and
`EtlRunStatus`.

### 4.5 Routers

#### `accidents.py` (prefix `/accidents`)

| Endpoint | Params | Response fields |
|---|---|---|
| `GET /overview` | — | `total_accidents, total_casualties, total_vehicles, fatal_count, serious_count, slight_count, years_covered, road_context_coverage_pct, weather_coverage_pct`. Coverage is computed live (`AVG(CASE WHEN ... IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100`), not cached — always the real, current enrichment percentage. |
| `GET /severity-distribution` | — | List of `{severity_id, severity_label, accident_count, total_casualties}` from `mart_severity_distribution`. |
| `GET /by-hour` | — | List of `{hour, day_of_week, day_name, severity_id, severity_label, accident_count}` from `mart_by_hour_weekday`. |
| `GET /monthly-trend` | — | List of `{year, month, month_name, severity_id, severity_label, accident_count, total_casualties, total_vehicles}` from `mart_monthly_trend`. |
| `GET /weather-impact` | — | List of `{weather_conditions, severity_id, severity_label, accident_count, avg_temperature_c, avg_precipitation_mm}` from `mart_weather_impact`. |
| `GET /road-type-severity` | — | List of `{road_type, severity_id, severity_label, accident_count}` from `mart_road_type_severity`. |
| `GET /hotspots` | `severity?: int`, `date_from?: date`, `date_to?: date`, `limit: int = 5000 (≤20000)` | List of `{accident_index, latitude, longitude, street_name, full_date, year, severity_id, severity_label, number_of_casualties}` from `mart_hotspots`, filtered server-side, ordered by date descending. |
| `GET /top-locations` | `limit: int = 50 (≤500)` | List of `{latitude, longitude, street_name, local_authority_district, accident_count, total_casualties, fatal_count}` from `mart_top_locations`. |
| `GET /casualties` | — | Per-year `{year, total_accidents, total_casualties, total_vehicles}`, computed with `COUNT(DISTINCT ...)` over left joins (not a mart — this one is a live aggregate query). |

#### `casualties.py` (prefix `/casualties`)

| Endpoint | Response |
|---|---|
| `GET /by-type` | `{casualty_type, casualty_class, severity_id, severity_label, casualty_count}` from `mart_casualties_by_type`. |
| `GET /demographics` | `{age_band_of_casualty, sex_of_casualty, severity_id, severity_label, casualty_count}` from `mart_casualty_demographics`. |

#### `vehicles.py` (prefix `/vehicles`)

| Endpoint | Response |
|---|---|
| `GET /by-type` | `{vehicle_type, severity_id, severity_label, vehicle_count}` from `mart_vehicles_by_type`. |

#### `admin.py` (prefix `/admin`)

In-memory `_status` dict (`status: idle|running|success|failed`, `run_id`,
`started_at`, `finished_at`, `message`) — **resets on backend restart**,
not persisted.

- `POST /run-etl` — requires header `X-Admin-Token` matching env
  `ADMIN_TOKEN` (`401` otherwise); `409` if a run is already `running`;
  `503` if the `etl/` directory isn't present in this deployment (e.g. the
  Docker image, which ships only `backend/` + `ml/`). On success, schedules
  `_run_pipeline()` as a FastAPI `BackgroundTasks` job and returns the new
  status immediately.
- `_run_pipeline(run_id)` shells out to `extract.py` → `transform.py` →
  `load.py` in order via `subprocess.run`, using the interpreter resolved by
  `_etl_python()` (env override `ETL_PYTHON`, else `etl/.venv/Scripts/python.exe`
  on Windows / `etl/.venv/bin/python` elsewhere, else falls back to the
  backend's own interpreter). Stops at the first non-zero exit and records
  its stderr (last 2000 chars) in `_status.message`.
- `GET /run-etl/status` — returns the current `_status` as-is.

#### `predict.py` (no prefix; Part 2)

Locates `ml/` by walking up from its own file until it finds
`ml/features.py` (`_find_ml_dir`), so the import works regardless of the
working directory the backend is launched from — then imports
`FEATURE_COLUMNS`/`build_features` from `ml/features.py` (see §5.1).

**Module state** (`ModelState`, process-lifetime, reset on restart):
`model` (loaded MLflow pyfunc), `version`, `stage`, `baseline` (parsed
`ml/artifacts/baseline.json`).

- `load_model()` — called once from `main.py`'s lifespan. Sets
  `MLFLOW_TRACKING_URI` (env override, else `ml/mlruns` as an absolute
  `file://` URI — must be absolute since ETL/ML/backend run from different
  working directories), looks up the `Production`-staged version of
  `traffic_severity_classifier` via `MlflowClient`, loads it with
  `mlflow.pyfunc.load_model`. **Never raises** — if no Production version
  exists yet, `state.model` simply stays `None` and every model-dependent
  endpoint below returns `503` until `ml/train.py` has been run.
- `LatencyTracker` — an in-process `deque(maxlen=500)` of per-request
  latencies plus served/failure counters; `percentile()` uses nearest-rank
  on the sorted sample. Explicitly a demonstration of serving telemetry, not
  a metrics backend — resets on restart, would be Prometheus/OTel in a real
  deployment.

| Endpoint | Request | Response |
|---|---|---|
| `POST /predict` | `PredictRequest`: `hour, day_of_week, month, road_type, speed_limit, junction_detail, light_conditions, weather_conditions, road_surface_conditions, urban_or_rural_area, number_of_vehicles` (required) + `temperature_c, precipitation_mm, is_weekend` (optional) | `PredictResponse`: `predicted_severity_id, predicted_severity_label, probabilities (dict[str,float]), model_version`. `503` if no model is loaded; `500` if inference itself throws. |
| `GET /model/info` | — | `ModelInfo`: `model_name, version, stage, algorithm, trained_at, metrics` — read straight from `baseline.json`. `503` if no model/baseline yet. |
| `GET /model/health` | — | `ModelHealth`: `model_loaded, model_version`. Always `200`. |
| `GET /monitoring/drift` | — | List of `DriftReport` (`kind: feature\|location\|class`, `feature, baseline_value, current_value, psi, drifted`). Computes PSI (Population Stability Index) for: **feature drift** (each numeric model input, vs. training deciles), **location drift** (lat/lon distribution, independent of the model's own features — catches the model being scored on a different part of the country than it trained on), **class drift** (has the severity mix itself shifted). Threshold: `psi > 0.2` per `ml/RETRAINING.md`. Scored against the 2,000 most recently loaded accidents by `full_date` (not `loaded_at`, which is identical across a bulk load and so would return an arbitrary physical slice, not a time window). `503` if no baseline yet. |
| `GET /monitoring/performance` | — | `PerformanceReport`: `window_start, window_end, macro_f1, recall_macro, false_negative_rate_fatal_serious, sample_size`. Scores the Production model against the 1,000 most recent accidents (again by `full_date`, approximating the chronological test split rather than mixing in training rows). `404` if no accidents loaded. |
| `GET /monitoring/data-quality` | — | `DataQualityReport`: per-field `missing_rate`, `invalid_rate` (out-of-range numerics / categorical values outside the model vocabulary), `baseline_missing_rate`, `flagged`; plus `live_requests`, `live_requests_with_issues`, `live_issue_counts` from the in-process `InputQualityTracker`. |
| `GET /monitoring/latency` | — | `LatencyReport`: `sampled_requests, predictions_served, failures, failure_rate, p50_ms, p95_ms, p99_ms, max_ms`, from the in-process `LatencyTracker`. |

---

## 5. MLOps (`ml/`)

### 5.1 `features.py`

`CATEGORICAL_VOCAB` is built directly from `stats19_lookups`' value sets
(`sorted(set(lk.ROAD_TYPE.values()))`, etc.) for six fields: `road_type`,
`junction_detail`, `light_conditions`, `weather_conditions`,
`road_surface_conditions`, `urban_or_rural_area` — so the model's
categorical vocabulary can never drift from the ETL's decoded labels.
`NUMERIC_COLUMNS = [hour, day_of_week, month, speed_limit,
number_of_vehicles, temperature_c, precipitation_mm]`. `FEATURE_COLUMNS =
NUMERIC_COLUMNS + [is_weekend] + [f"{field}_code" for field in
CATEGORICAL_VOCAB]` — this exact order is what both `train.py` and
`predict.py` feed the model.

`encode_category(field, value)` does an O(1) dict lookup into a
precomputed `label -> index` map (`_VOCAB_INDEX`) rather than
`list.index()` — at ~514k rows × 6 categorical columns, a linear scan would
be tens of millions of string comparisons. An unseen/unhashable value maps
to `len(vocab)` (one "unknown" bucket per field, past the end of the known
categories).

`build_features(record: dict) -> dict` — the single conversion function
imported by both `train.py` (row-wise, via `build_features_df`) and
`backend/app/routers/predict.py` (once per `/predict` call), so training
and serving literally cannot diverge. Missing numeric values become `NaN` (exactly what training sees for a NULL; the tree models handle it natively) and an omitted `is_weekend` is derived from `day_of_week`;
missing/unset `is_weekend` becomes `False`.

### 5.2 `train.py`

`TRAIN_QUERY` joins `fact_accidents` to all four dimensions, ordered by
`d.full_date` — the ordering matters because `chronological_split()`
(70% train / 15% val / 15% test, no shuffling) relies on it: **earlier
years train, most recent years test**, because this model predicts future
accidents, not exchangeable historical rows.

Three candidates are trained per run, each wrapped in one `mlflow.start_run`:

| Model | Config |
|---|---|
| RandomForest | 300 trees, max_depth 14, `class_weight="balanced"` |
| XGBoost | 300 trees, max_depth 6, lr 0.1, `multi:softprob`; balanced via `compute_sample_weight("balanced", y_train)` since XGBoost has no native `class_weight=`; fit on 0..K-1 label indices matching `classes = sorted(severity_id.unique())`, so its `predict_proba` columns line up with `classes` by construction |
| LightGBM | 300 trees, no max depth limit, lr 0.1, `class_weight="balanced"` |

Each run logs its full hyperparameter set (`model.get_params()`) and
`val_*`/`test_*` metrics (`accuracy`, `macro_f1`, `recall_macro`). The
val-macro-F1 winner is re-wrapped in `SeverityPipeline` (`ml/pipeline.py`, raw record in -> feature engineering -> model -> probabilities) (an
`mlflow.pyfunc.PythonModel` exposing a uniform `predict()` = `predict_proba`
contract regardless of which of the three libraries actually won — this is
what lets `predict.py` treat every possible winner identically), logged as
a fresh registered version of `traffic_severity_classifier`, and promoted:
every existing `Production` version is moved to `Archived` first, then the
new version is set `Production` (so there's exactly one Production version
at a time).

Finally, `baseline.json` is written to `ml/artifacts/` —
`{trained_at, algorithm, model_version, metrics, class_labels,
numeric_feature_stats, location_stats, class_distribution}`. The two
`*_stats` blocks are per-column **decile profiles** (`decile_profile()`:
mean, std, 9 decile edges, per-bucket proportions) computed on the
*training* split — this is the exact shape `/monitoring/drift`'s PSI
calculation expects, so any later profiled column (a model feature or
`latitude`/`longitude`) is comparable against live data through the same
code path.

### 5.3 `evaluate.py`

Standalone: loads whatever is currently `Production`, re-derives the same
chronological split from a fresh warehouse pull, scores the test slice, and
logs `macro_f1`, `recall_macro`, the full `classification_report` text, and
a confusion-matrix PNG (`matplotlib`, `Agg` backend) — all as a new MLflow
run tagged with the evaluated model's version, separate from the training
runs. Doesn't touch the registry; purely a scoring/reporting pass, runnable
any time without retraining.

---

## 6. Dashboard (`dashboard/`)

### 6.1 `api_client.py`

`get(path, params=None)` — `@st.cache_data(ttl=60s)`, so identical calls
within 60 seconds don't re-hit the API; raises `ApiError` on any
`requests.RequestException`. `post(path, json=None, headers=None)` —
deliberately **not** cached (a prediction request has side effects on the
in-memory latency tracker and must always be a real call), returns the raw
`requests.Response` so callers can inspect the status code themselves.
`API_BASE_URL` from env, default `http://127.0.0.1:9001`.

### 6.2 `Home.py`

Calls `/accidents/overview`; stops with a warning if `total_accidents == 0`
(no bundled/placeholder KPIs). Renders two rows of `st.metric` (totals;
Fatal/Serious/Slight breakdown), then the two enrichment-coverage bars
(`st.progress`) with the exact live percentages from the API — not a static
claim — and `st.page_link` navigation to all 8 pages.

### 6.3 Analytics pages (1–4, 6)

All five follow the same shape: fetch one or more endpoints, stop early
with `st.info` if empty, apply a sidebar filter, render Plotly.

| Page | Endpoint(s) | Sidebar filter(s) | Chart(s) |
|---|---|---|---|
| `1_Severity_Distribution.py` | `/accidents/severity-distribution` | Chart type: Bar / Donut | Bar or donut by severity, plus the raw table. |
| `2_Time_Patterns.py` | `/accidents/by-hour`, `/accidents/monthly-trend` | Multiselect severity | Tab 1: hour×weekday heatmap (`px.imshow`). Tab 2: monthly line trend by severity. |
| `3_Weather_Impact.py` | `/accidents/weather-impact` | Multiselect severity | Stacked bar (weather condition × severity), plus grouped bar of avg temperature/precipitation per condition. |
| `4_Road_Type_Severity.py` | `/accidents/road-type-severity` | Multiselect severity, "Show as % within road type" checkbox | Stacked bar, counts or row-normalized percentages; raw table below. |
| `6_Casualties_Vehicles.py` | `/casualties/by-type`, `/casualties/demographics`, `/vehicles/by-type`, `/accidents/casualties` | Multiselect severity (applies to tabs 1–3) | 4 tabs: casualty type (stacked bar), demographics (age×sex heatmap), vehicle type (stacked bar), yearly totals (grouped bar + table). |

### 6.4 `5_Hotspot_Map.py`

Sidebar: severity (`All`/Fatal/Serious/Slight, mapped to `severity_id`),
date range (defaulted from `/accidents/overview`'s `years_covered`). Calls
`/accidents/hotspots` with those as query params (`limit=8000`, filtering
happens server-side, not client-side) and renders an `st.pydeck_chart`
`ScatterplotLayer` colored by severity (`Fatal` red, `Serious` orange,
`Slight` blue), tooltip showing severity/date/street. Below the map, a
separate table from `/accidents/top-locations?limit=20`.

### 6.5 `7_Severity_Predictor.py` (Part 2)

Imports `CATEGORICAL_VOCAB` directly from `ml/features.py`, so every
dropdown's options are guaranteed to match what the model was trained on.
Checks `/model/health` first and stops with a warning if no model is
loaded. The form collects all `PredictRequest` fields (hour slider, day
selectbox, month slider, speed-limit select-slider, vehicle-count slider,
weekend checkbox, six categorical selectboxes defaulted to a "typical"
condition, temperature/precipitation number inputs) and `POST`s to
`/predict` on submit, then renders the predicted label (color-coded) and a
Plotly bar of the full probability breakdown.

### 6.6 `8_Model_Monitoring.py` (Part 2)

Checks `/model/health` first. Three sections, each independently
error-tolerant (one endpoint failing doesn't blank the others):
- **Rolling performance** (`/monitoring/performance`): metric tiles +
  window dates; flags in red if macro-F1 has dropped >0.05 vs. the training
  baseline (`/model/info`'s metrics) or the Fatal/Serious false-negative
  rate exceeds 15% — both thresholds match `ml/RETRAINING.md`.
- **Drift** (`/monitoring/drift`): one tab per `kind` (feature/location/
  class), a PSI bar chart per tab with a dashed line at the 0.2 retrain
  threshold, colored red/green by the `drifted` flag, plus the raw table.
- **Serving telemetry** (`/monitoring/latency`): served/failed counts and
  p50/p95 latency tiles; explains itself as empty until `/predict` has been
  called at least once since the backend started (the tracker is
  in-process and resets on restart).

---

For environment variables, the run order, Docker/Airflow instructions, and
the assignment-submission mapping, see [README.md](README.md).
