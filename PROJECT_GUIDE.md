# Traffic Accident Severity Analytics & Prediction — Project Guide

A plain-language walkthrough of the whole project: what it is, why it exists,
how data moves through it, and what every part does. For exact setup commands
see [README.md](README.md); for column-level and endpoint-level reference see
[DETAILED_README.md](DETAILED_README.md).

## Contents

1. [What the project is about](#1-what-the-project-is-about)
2. [What problem it solves](#2-what-problem-it-solves)
3. [The workflow, end to end](#3-the-workflow-end-to-end)
4. [The data flow](#4-the-data-flow)
5. [PostgreSQL: what happens in the database](#5-postgresql-what-happens-in-the-database)
6. [The ETL pipeline in detail](#6-the-etl-pipeline-in-detail)
7. [The backend (FastAPI)](#7-the-backend-fastapi)
8. [The Streamlit dashboard, graph by graph](#8-the-streamlit-dashboard-graph-by-graph)
9. [What the model predicts](#9-what-the-model-predicts)
10. [How the model was trained](#10-how-the-model-was-trained)
11. [Why these three models](#11-why-these-three-models)
12. [Model results and how to read them honestly](#12-model-results-and-how-to-read-them-honestly)
13. [Monitoring and retraining](#13-monitoring-and-retraining)
14. [Why Docker](#14-why-docker)
15. [Why Airflow](#15-why-airflow)
16. [Known limitations](#16-known-limitations)

---

## 1. What the project is about

This project takes the UK government's official road-accident records
(**STATS19**, the "last 5 years" export from the Department for Transport) and
turns them into three things:

1. **A clean, queryable warehouse.** The raw files are three big CSVs
   (collisions, casualties, vehicles) full of numeric codes. The project cleans
   them, decodes the codes into readable labels, and adds two kinds of outside
   information: the **weather** at the time and place of each accident
   (Open-Meteo) and the **street name** (TomTom). Everything is stored in
   PostgreSQL with the PostGIS extension.
2. **An interactive dashboard.** A Streamlit app that shows when, where and
   under what conditions accidents happen, and which of them are the most
   severe.
3. **A prediction service.** A machine-learning model that, given conditions
   such as time of day, road type, speed limit, light and weather, estimates
   whether an accident would be **Fatal, Serious or Slight**. It is tracked
   with MLflow, served through the same API as the dashboard's data, and
   monitored for drift.

It is an individual project for the Data Engineering and MLOps course.
**Part 1** is the data pipeline and dashboard. **Part 2** is the MLOps layer:
training, model registry, serving, monitoring and Docker.

## 2. What problem it solves

| Problem | How the project addresses it |
|---|---|
| The raw STATS19 files are unreadable as-is: severity is `3`, weather is `1`, road type is `6`. | `etl/stats19_lookups.py` holds a dictionary per field, and the transform step decodes every code into a label. |
| Accident data says nothing about the weather or the street. | The extract step pulls hourly weather for every accident's location and time, and reverse-geocodes busy coordinates to street names. |
| Questions like "are wet roads deadlier?" need joins over hundreds of thousands of rows. | A star-schema warehouse plus pre-computed materialized views make those questions fast. |
| Raw data has bad rows (duplicates, missing or impossible coordinates). | The transform step rejects them and records every rejection reason and count, so nothing disappears silently. |
| Results are hard to share when they live in scripts. | A FastAPI service exposes everything over HTTP, and the dashboard consumes only that API. |
| A model that is trained once and forgotten goes stale. | MLflow tracks every run, a registry holds the Production version, and monitoring endpoints watch drift and performance. |
| "Works on my machine" deployment. | The inference API is packaged in a Docker image. |

The practical value is a tool for **road-safety analysis**: seeing where and
when severe accidents cluster, and getting a rough risk estimate for a given
set of conditions.

## 3. The workflow, end to end

```
1. Create the database       sql/schema.sql  (tables, views, indexes)
2. Run the ETL               extract.py -> transform.py -> load.py
3. Verify                    quality_check.py  (fails loudly if the warehouse looks wrong)
4. Start the API             uvicorn app.main:app --port 9001
5. Open the dashboard        streamlit run Home.py
6. Train the model           python ml/train.py   (registers a Production model in MLflow)
7. Restart the API           it loads the new Production model at startup
8. Predict and monitor       dashboard "Severity Model" page -> /predict and /monitoring/*
9. Retrain when needed       per ml/RETRAINING.md thresholds, then repeat from step 2 or 6
```

The ETL can be triggered three ways, all running the same scripts:
a manual command line run, `POST /admin/run-etl`, or the Airflow DAG.

## 4. The data flow

```
 dataset/raw/                      Open-Meteo API            TomTom API
 collisions.csv                    (hourly weather)          (street names)
 casualties.csv                          │                        │
 vehicles.csv                            │                        │
        │                                │                        │
        └──────────────┬─────────────────┴────────────────────────┘
                       ▼
               ┌──────────────┐
               │  extract.py  │  validate columns, snapshot raw CSVs (immutable),
               └──────┬───────┘  fetch weather + street names, write audit log
                      ▼
               etl/data/raw/        timestamped CSV snapshots, weather_grid.parquet,
                      │             road_context.parquet
                      ▼
               ┌──────────────┐
               │ transform.py │  reject bad rows, decode codes, build date keys,
               └──────┬───────┘  join weather + street, restrict child tables
                      ▼
               etl/data/staging/    accidents / casualties / vehicles .parquet
                      │
                      ▼
               ┌──────────────┐
               │   load.py    │  one transaction, set-based SQL, then refresh marts
               └──────┬───────┘
                      ▼
        PostgreSQL + PostGIS  ── dimensions, facts, 10 mart_* views, vw_model_features
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
   FastAPI (mart reads,      ml/train.py (reads vw_model_features)
   /predict, /monitoring)           │
          ▲                         ▼
          │                  MLflow registry ── Production model
          │                         │
          └──── loads at startup ◄──┘
          ▲
          │ HTTP (API_BASE_URL)
   Streamlit dashboard
```

Three design rules hold the flow together:

- **The raw zone is immutable.** Each run copies the source CSVs to a
  timestamped snapshot before any cleaning. A bad transform can always be
  re-run from the snapshot.
- **The dashboard never touches the database.** It only calls the API, so there
  is one place where business logic and SQL live.
- **Training and serving share one feature function.** `ml/features.py` is
  imported by both `train.py` and the API, so the model cannot be fed
  differently-prepared data in production than it saw in training.

## 5. PostgreSQL: what happens in the database

PostgreSQL 15+ runs as a native local install (not in Docker). The PostGIS
extension adds a geography type so locations are real map points, not just two
numbers.

### The star schema

A star schema keeps measurable events in the middle (**fact** tables) and the
descriptive context around them (**dimension** tables).

| Table | Kind | One row is… | Holds |
|---|---|---|---|
| `fact_accidents` | fact | one collision | severity, number of vehicles/casualties, and keys into the dimensions |
| `fact_casualties` | fact | one injured person | their own severity, age band, sex, casualty class and type |
| `fact_vehicles` | fact | one vehicle in a collision | vehicle type and related attributes |
| `dim_date` | dimension | one distinct hour | date, hour, day of week, month, quarter, year, weekend flag. Key format is `YYYYMMDDHH` |
| `dim_location` | dimension | one distinct coordinate | latitude/longitude, a PostGIS `geography` column with a GiST index, police force, local authority, urban/rural, TomTom street name (nullable) |
| `dim_road` | dimension | one distinct road configuration | road class, road type, speed limit, junction detail and control |
| `dim_weather` | dimension | one distinct weather situation | light, weather and road-surface conditions plus temperature and precipitation from Open-Meteo |
| `dim_severity` | dimension | one severity code | `1=Fatal, 2=Serious, 3=Slight, -1=Data missing` |
| `etl_audit_log` | audit | one pipeline stage per run | source, row count, rejected count, status |

Casualties and vehicles are separate fact tables because they have a different
**grain** from the accident. One collision can injure several people, each with
their own severity, so folding them into `fact_accidents` would lose that.

STATS19's own `-1` ("data missing or out of range") is kept as a labelled
category rather than being dropped.

### Materialized views (the "mart")

A **materialized view** stores the result of a query on disk. The dashboard's
questions (counts by hour and weekday, by weather, by road type…) are
aggregations over the whole fact table; computing them once at load time makes
every dashboard click fast.

| View | Answers |
|---|---|
| `mart_severity_distribution` | how many collisions per severity |
| `mart_by_hour_weekday` | accidents by hour × weekday × severity |
| `mart_monthly_trend` | monthly counts and casualties by severity |
| `mart_weather_impact` | weather condition vs severity |
| `mart_road_type_severity` | road type vs severity |
| `mart_hotspots` | individual accident points for the map (indexed on severity and date) |
| `mart_top_locations` | coordinates with the most collisions |
| `mart_casualties_by_type` | casualty type and class vs severity |
| `mart_casualty_demographics` | age band and sex vs severity |
| `mart_vehicles_by_type` | vehicle type vs severity |

All ten are refreshed at the end of every `load.py` run.

There is also a plain view, **`vw_model_features`**: one row per collision with
exactly the columns the model uses (date, hour, road, weather, location, and the
severity label). `train.py` and the monitoring endpoints read from it so the
model never queries the raw tables.

### What a load does to the database

1. Bulk-loads each staging parquet file into a temporary table.
2. Fills the dimensions with `INSERT … SELECT DISTINCT … ON CONFLICT DO NOTHING`,
   so reruns never duplicate a dimension row.
3. Upserts `fact_accidents`, joining back to the dimensions to resolve the
   surrogate keys, with `ON CONFLICT (accident_index) DO UPDATE`, so reruns
   update instead of duplicating.
4. Inserts casualties and vehicles.
5. Commits once. If anything fails, the whole load rolls back and the database
   is left as it was.
6. Refreshes the ten materialized views and writes a summary row to
   `etl_audit_log`.

The load is set-based (whole tables at a time), not row-by-row. That is the
difference between minutes and hours on roughly half a million collisions.

## 6. The ETL pipeline in detail

### `extract.py`

- Reads the three CSVs from `dataset/raw/` and checks the required columns
  exist, failing with a clear message if not.
- Copies each to a timestamped snapshot under `etl/data/raw/` and prunes old
  snapshots.
- **Weather.** Fetching weather for every accident one by one would be hundreds
  of thousands of API calls. Instead each accident's coordinates are snapped to a
  **0.5° grid** (`round(x / 0.5) * 0.5`), and the script makes one Open-Meteo
  request per *(grid cell, year)* to pull that cell's whole year of hourly
  history. Timestamps are requested in `Europe/London` to match STATS19's local
  wall-clock time. Requests retry with exponential backoff, and progress is
  checkpointed so a retry resumes rather than refetching. Output:
  `weather_grid.parquet`.
- **Street names.** TomTom reverse geocoding is quota-limited on the free tier,
  so `--road-limit N` spends the quota on the busiest coordinates first and skips
  any already resolved in an earlier run. Output: `road_context.parquet`.
- Appends a JSON line to `etl/data/audit/audit_log.jsonl` for every
  source and stage.
- Flags: `--skip-road`, `--road-limit`.

### `transform.py`

- **Rejects** (and logs counts plus a `rejected_records.csv`): duplicate
  collision IDs, missing coordinates, coordinates outside Great Britain's
  bounding box (about lat 49.8–60.9, lon −8.6–1.8), and unparseable date/time.
- **Decodes** numeric codes to labels through `stats19_lookups.py`.
- **Derives** `date_key`, `day_of_week`, `day_name`, `is_weekend`.
- **Joins weather** using the same grid-rounding as extract, matched on grid cell
  *and* exact hour.
- **Left-joins street names** on exact latitude/longitude, so accidents without a
  resolved street keep a NULL instead of vanishing.
- **Restricts** casualties and vehicles to collisions that survived cleaning, so
  the load cannot violate a foreign key.
- Writes the three staging parquet files.

### `load.py` and `quality_check.py`

`load.py` is described in section 5. `quality_check.py` runs afterwards and exits
non-zero if any core table is empty or the last load did not succeed, so an
orchestrator turns red instead of leaving a half-loaded warehouse behind a green
tick.

## 7. The backend (FastAPI)

The backend (`backend/app/`) is the only thing that talks to PostgreSQL on behalf
of the dashboard, and the only thing that runs the model.

| File | Role |
|---|---|
| `main.py` | Creates the app, adds permissive CORS (local development), includes all routers, and loads the model in a **lifespan** handler at startup. |
| `db.py` | A **lazy** async SQLAlchemy engine. It does not fail at import time if `DATABASE_URL` is missing; a dependency raises a friendly HTTP 503 instead. |
| `schemas.py` | Pydantic models that validate every request and shape every response. |
| `models.py` | ORM mirror of the schema, kept for reference. |
| `routers/accidents.py`, `casualties.py`, `vehicles.py` | Thin reads over the `mart_*` views. Most accept year and severity filters; `/accidents/hotspots` also takes `date_from` / `date_to`. |
| `routers/admin.py` | `POST /admin/run-etl` (needs the `X-Admin-Token` header) starts extract → transform → load as a background task; `GET /admin/run-etl/status` reports `idle / running / success / failed` from an in-memory dict. A second request while one is running gets HTTP 409. |
| `routers/predict.py` | Everything model-related, below. |

### What `predict.py` does

- **At startup** it points MLflow at the shared tracking store, finds the
  `Production` version of `traffic_severity_classifier`, loads it, and loads the
  training baseline (`ml/artifacts/baseline.json`). If no model exists it does
  not crash; `/predict` simply returns 503 until one is trained.
- **`POST /predict`** validates the request, records any input problems (missing
  numbers, categories the model has never seen), runs the model, times the call,
  and returns the predicted severity plus the probability of each class.
- **`GET /model/info`** and **`/model/health`** report the served version,
  algorithm, test metrics, and whether a model is loaded.
- **`GET /monitoring/drift`** compares recent warehouse data to the training
  baseline using **PSI** (Population Stability Index) for the numeric features,
  for location (latitude/longitude) and for the severity class mix.
- **`GET /monitoring/performance`** re-scores the 1,000 newest accidents against
  their true severity and reports macro-F1, macro recall and the
  Fatal/Serious false-negative rate.
- **`GET /monitoring/latency`** reports p50/p95 latency and the failure count
  from a bounded in-memory tracker.
- **`GET /monitoring/data-quality`** reports missing, out-of-range and
  unseen-category rates, for both the warehouse window and live `/predict`
  inputs.

The latency and input-quality trackers are deliberately in-memory and reset when
the process restarts. They demonstrate serving telemetry; a real deployment would
export them to Prometheus or OpenTelemetry.

### Endpoint summary

`/health`, `/accidents/{overview, severity-distribution, by-hour, monthly-trend,
weather-impact, road-type-severity, hotspots, top-locations, casualties}`,
`/casualties/{by-type, demographics}`, `/vehicles/by-type`,
`/admin/run-etl[/status]`, `/predict`, `/model/{info, health}`,
`/monitoring/{drift, performance, latency, data-quality}`.
Interactive docs are at `http://127.0.0.1:9001/docs`.

## 8. The Streamlit dashboard, graph by graph

The dashboard (`dashboard/`) is a native Streamlit multi-page app. Three
principles apply to every page:

- **No bundled or placeholder data.** Every number comes from the API through
  `api_client.py`, which caches GET results for 60 seconds.
- **Filters are server-side.** Changing a sidebar filter re-queries the API with
  new parameters; the browser does not filter a cached table.
- **Empty states are explicit.** If a filter matches nothing, the page says so
  instead of rendering an empty chart.

The **shared sidebar** (from `api_client.sidebar_filters`) offers a **year-range
slider** and a **severity multiselect** (Fatal / Serious / Slight). The hotspot
map has its own filters (below).

Severity colours are consistent everywhere: **Fatal = red, Serious = orange,
Slight = blue, Data missing = grey**, so a colour means the same thing on every
chart.

### Home page — the overview

| Element | What it shows | Why it exists |
|---|---|---|
| **KPI strip** (Accidents, Casualties, Fatal, Serious) | Totals for the current filter. | The headline numbers a reader looks for first. They respond to the filters, so they double as a sanity check on what is selected. |
| **Severity mix** (donut chart) | Share of Fatal / Serious / Slight / Data missing. | Shows the central fact of the dataset: most accidents are Slight and Fatal is rare. That imbalance explains how the model is built and why it is hard (sections 10 and 12). |
| **When accidents happen** (heatmap, weekday × hour) | Darker cells mean more accidents at that hour on that weekday. | Reveals the rhythm of risk: weekday commuter peaks versus weekend patterns. It answers "when should safety effort be concentrated?" |
| **Monthly trend** (line chart, one line per severity) | Accidents per month over the selected years. | Shows seasonality and long-term change across 2021–2025, and whether severe accidents move with the total or separately. |
| **Data coverage** (expander) | Percentage of accidents with weather, and with a TomTom street name. | Honesty about data quality. TomTom's free tier cannot cover every coordinate, so the real coverage figure is displayed instead of hidden. |
| **Navigation links** | `st.page_link` to each other page. | Easy movement between pages. |

### Page 1 — Risk Factors

Four API calls feed three tabs. This page asks: *which conditions, roads and
people are associated with more severe accidents?*

**Weather tab — stacked bar chart.** One bar per weather condition (fine, rain,
fog, snow, high winds…), stacked by severity.
- Default: raw counts, sorted by total accidents. This shows where most accidents
  happen. Most happen in fine weather simply because most driving does.
- Toggle **"Show as % within weather condition"**: each bar becomes 100% split by
  severity. This is the useful view. Raw counts are dominated by common
  conditions, while percentages let you compare the *severity mix* between
  conditions and see whether, for example, fog or snow shifts accidents toward
  Serious.

**Road type tab — stacked bar chart.** Same idea for road layout (single
carriageway, dual carriageway, roundabout, slip road, one-way street…). It
defaults to the percentage view, because the point is to compare how dangerous
each layout is *per accident*, not how common it is. It supports the road-type
feature in the model.

**People & vehicles tab — two charts side by side.**
- **Casualties by age and sex (heatmap).** Age band on one axis, sex on the
  other, colour by casualty count. It shows which groups are injured most, for
  example young adults, and how that differs by sex.
- **Vehicles involved (horizontal bar, stacked by severity).** Vehicle types
  ranked by how many were involved, with the severity breakdown. It shows which
  vehicle types appear in the worst outcomes, such as motorcycles or goods
  vehicles versus cars.

### Page 2 — Hotspot Map

This page asks: *where do accidents happen?* It has its own sidebar filters:
**severity** (All / Fatal / Serious / Slight) and a **From / To date range**,
defaulting to the full span of the data.

- **Scatter map (pydeck).** Each dot is one accident, coloured by severity
  (red / orange / blue). Hovering shows the severity, date and street name. The
  query runs on the server and is capped at **8,000 points**, because sending
  hundreds of thousands of points to a browser would be unusably slow. The caption
  states the count so the cap is not hidden.
- **Top 20 locations (table, in an expander).** The exact coordinates with the
  most collisions. A scatter map shows density visually, but a ranked table gives
  precise, citable answers: "these are the 20 worst spots."

It exists because accident risk is spatial. Filtering to Fatal only usually
shows a very different pattern from Slight, with motorways and rural A-roads
standing out.

### Page 3 — Severity Model

A header strip shows the **algorithm, model version, test macro-F1 and training
date** of the model currently in Production, then two tabs. If no model is
loaded the page says to run `ml/train.py` rather than showing an empty form.

**Predict tab — form → `/predict` → probability chart.** The user sets hour, day
of week, month, speed limit, number of vehicles, road type, junction detail,
light, weather, road surface, urban/rural, temperature and precipitation. On
submit the page shows:
- the **predicted class**, colour-coded;
- a **bar chart of the probability of each class** (Fatal / Serious / Slight).

The probabilities matter as much as the label. A prediction of "Slight" at 48%
with Serious at 41% tells the reader the model is unsure; showing only the label
would hide that.

**Monitoring tab — is the model still healthy?**
- **Rolling performance** (four metric tiles): macro-F1, macro recall, the share
  of Fatal/Serious accidents the model missed, and the sample size. The page turns
  red if macro-F1 falls more than 5 points below the training baseline, or if the
  Fatal/Serious miss rate exceeds 15%. These are the retraining triggers from
  `RETRAINING.md`.
- **Drift vs training baseline** (bar chart of PSI, panelled by feature, location
  and class). Each bar is how far recent data has moved from the training data. A
  dashed line at 0.2 marks the retrain threshold, and bars are red when past it.
  Drift is the early warning: it can appear before performance visibly drops.
- **Serving telemetry and input-data quality** (expander): predictions served,
  p50/p95 latency, failures, and a grouped bar chart of missing and invalid rates
  per field. It shows whether the *service* is healthy and whether the *inputs*
  are sensible, which is separate from whether the *model* is accurate.

## 9. What the model predicts

**It predicts the severity of a road collision: Fatal (1), Serious (2) or
Slight (3).**

- It is a **three-class classification** problem.
- The **input** is the circumstances of the accident: when (hour, day of week,
  month, weekend), where and on what (road type, speed limit, junction detail,
  urban or rural), what the environment was (light, weather, road surface,
  temperature, precipitation) and how many vehicles were involved.
- The **output** is the predicted class and a probability for each class.
- The **label** is the collision's recorded severity in STATS19, which is the
  severity of its worst injury.

The features are deliberately known-in-advance or known-at-the-scene conditions.
Nothing about the injuries or outcome is fed in. That is what makes the question
meaningful: *given these conditions, how bad is the accident likely to be?*

Practical uses are risk screening (which combinations of conditions skew severe),
helping decide where interventions such as lower speed limits or lighting would
matter, and, for this course, demonstrating a full prediction service with
monitoring.

**Feature encoding** (`ml/features.py`): seven numeric features (hour, day of
week, month, speed limit, number of vehicles, temperature, precipitation), a
weekend flag, and six categorical features turned into integer codes using the
STATS19 vocabularies (road type, junction detail, light, weather, road surface,
urban/rural). A category the model has never seen goes into an explicit "unknown"
bucket, and a missing number stays missing (NaN) instead of becoming 0, because
the tree models handle missing values natively and a missing temperature should
not be read as "freezing".

## 10. How the model was trained

Everything is in `ml/train.py`.

1. **Load the data.** Reads `vw_model_features` from PostgreSQL via
   `DATABASE_URL`: roughly half a million labelled collisions, ordered by date.
2. **Split chronologically: 70% train / 15% validation / 15% test.** The earliest
   accidents train the model, the next block tunes and selects, and the most recent
   block tests it. A random shuffle would be wrong here. Real use is predicting
   *future* accidents from *past* ones, and a random split would let the model see
   neighbouring days from the test period during training, which flatters the
   score. The chronological split gives an honest estimate.
3. **Handle class imbalance.** The training labels are about **76% Slight, 22%
   Serious, 1.5% Fatal**. A model that always answered "Slight" would be 76%
   accurate and useless. So Random Forest and LightGBM use `class_weight="balanced"`
   and XGBoost uses balanced sample weights, making a mistake on a rare Fatal case
   cost far more than one on a common Slight case.
4. **Train three candidates** (Random Forest, XGBoost, LightGBM; see the next
   section), each with 300 trees and fixed random seeds.
5. **Evaluate each on validation and test** and log to **MLflow**, with one run
   per model:
   - parameters and split sizes;
   - accuracy, **macro-F1** and macro recall on validation and test;
   - per-class recall, the Fatal/Serious false-negative rate, and confusion
     matrices (via `reporting.py`).
6. **Pick the winner by validation macro-F1.** Selection uses *validation*, not
   test, so the test set stays untouched by the choice and remains an honest final
   score.
7. **Package it.** The winner is wrapped in a `SeverityPipeline`
   (`ml/pipeline.py`), a single MLflow model that accepts **raw** records and does
   its own feature encoding. The feature code and STATS19 lookups are bundled with
   it (`code_paths`), so it loads anywhere, including inside Docker, with no path
   tricks.
8. **Register and promote.** It is registered as `traffic_severity_classifier`,
   moved to the `Production` stage, and the previous Production version is
   archived.
9. **Write the monitoring baseline.** `ml/artifacts/baseline.json` stores the
   training set's decile profile of every numeric feature, of latitude/longitude,
   and the class mix. It is also logged next to the model in MLflow, so each model
   version is versioned with the exact baseline its drift is measured against.

**Why macro-F1 is the selection metric.** Accuracy rewards ignoring the rare
classes. Macro-F1 computes F1 for each class separately and averages them equally,
so doing badly on Fatal costs as much as doing badly on Slight. For a safety
problem that is the right pressure.

`ml/evaluate.py` re-scores the Production model and saves a confusion matrix and
classification report as artifacts.

## 11. Why these three models

All three are **tree-based ensembles**. That fits this data for the same reasons
across the board:

- The data is **tabular**, with mixed numeric and categorical features. Tree
  ensembles are the standard strong choice for this shape of data.
- They **capture interactions** without hand-engineering. For example,
  "70 mph road × darkness × wet surface" is the kind of combination that matters
  for severity, and trees find such splits automatically.
- They need **no feature scaling** and cope with the integer-coded categories.
- They are **robust** to noisy features and handle missing values (XGBoost and
  LightGBM natively).
- They produce **class probabilities**, which the dashboard displays.

Within that family the three differ in useful ways, so comparing them is
informative instead of redundant:

| Model | Method | Why include it |
|---|---|---|
| **Random Forest** | Many deep trees trained independently on random subsets, then averaged ("bagging"). | A strong, low-tuning baseline. Averaging reduces variance and resists overfitting on noisy data. `class_weight="balanced"` handles imbalance directly. |
| **XGBoost** | Trees built one after another, each correcting the previous trees' errors ("boosting"), with regularization. | Usually among the most accurate on tabular data. Boosting can fit subtler patterns than bagging. It lacks a `class_weight` option, so balanced sample weights are used instead. |
| **LightGBM** | A faster boosting implementation that grows trees leaf-wise and uses histogram binning. | A second boosting variant. It is fast on large tables (about half a million rows) and handles imbalance with `class_weight`. |

Bagging versus two flavours of boosting spans the main approaches to ensembles,
and MLflow makes the comparison fair, since every candidate has the same data
split, features and metric. The winner is chosen by the numbers, not by assumption.

Logistic regression or a neural network were not used: a linear model would miss
the feature interactions, and a neural network adds complexity with no reliable
benefit on a tabular table of this size.

## 12. Model results and how to read them honestly

The model currently registered in Production is **Random Forest, version 4**
(from `ml/artifacts/baseline.json`), with these held-out **test** results on the
most recent 15% of accidents:

| Metric | Value |
|---|---|
| Accuracy | 0.553 |
| Macro-F1 | 0.364 |
| Macro recall | 0.460 |

How to read this:

- **This is a modest model, and that is expected.** Severity depends heavily on
  things the dataset cannot know in advance: how fast people were actually
  driving, whether seatbelts were worn, who was at fault, what the vehicles
  weighed. Conditions alone only partly predict severity.
- **Accuracy is lower than the 76% "always Slight" baseline.** That is not a
  malfunction. Because the model is deliberately weighted toward the rare severe
  classes, it labels many Slight accidents as more serious, which lowers raw
  accuracy and raises recall on Fatal and Serious. That trade-off is the intended
  one for a safety use case, where missing a severe accident is worse than a false
  alarm. The metrics to watch are macro-F1, macro recall and the Fatal/Serious
  false-negative rate, not accuracy.
- The output is best used as a **relative risk indicator** (do these conditions
  skew toward severe outcomes?), not as an authoritative forecast for one
  collision.
- The monitoring page measures against this baseline, so any future drop is
  visible.

## 13. Monitoring and retraining

Thresholds are documented in `ml/RETRAINING.md`. A retrain is **required** when
any of these holds:

| Signal | Threshold |
|---|---|
| Rolling macro-F1 | more than 5 points below the Production model's logged test macro-F1 |
| Fatal/Serious false-negative rate | above 15% |
| Feature drift | PSI above 0.2 on any feature |
| New data | a load adds a full new year of accidents |
| Input data quality | a field's missing rate rises more than 10 points, or over 5% implausible or unseen values |
| Code/vocabulary change | STATS19 lookups or `ml/features.py` vocabularies change |

**PSI** measures how much a feature's distribution has shifted from training:
below 0.1 is stable, 0.1–0.2 is worth watching, above 0.2 is a significant
shift.

Retraining is a deliberate human decision. The monitoring endpoints are read-only
signals, because STATS19 data arrives in the government's own release cycle, not
in real time. The procedure is: re-run the ETL, run `python ml/train.py`, compare
the new version in the MLflow UI, keep it or roll back, and restart the API.

## 14. Why Docker

The assignment brief grades **"Containerize the inference service using Docker"**
as its own line item, so this is a deliberate, limited use of Docker.

**What is containerized:** only the FastAPI service (`backend/Dockerfile`). The
image holds the app, the `ml/` code, the STATS19 lookups and all Python
dependencies. It can load the registered model, which is already packaged with
its feature code.

**What is not:** PostgreSQL and Streamlit run natively, and the ETL runs natively.
Containerizing everything would add moving parts without helping the core goals.

**Why it helps the service specifically:**
- **Reproducibility.** The image pins the Python version and dependencies, so the
  API behaves the same on any machine. The model stack (scikit-learn, XGBoost,
  LightGBM, MLflow) is sensitive to library versions.
- **System dependency handled once.** XGBoost and LightGBM need the OpenMP runtime
  (`libgomp1`), absent from a slim base image. The Dockerfile installs it, so
  nobody hits an import error at deploy time.
- **Portability.** One image can run on a laptop, a server or a cloud service.
- **Isolation.** The API's pinned FastAPI stack is kept apart from Streamlit's and
  MLflow's dependencies, which would otherwise conflict. This is also why the
  project uses separate virtualenvs locally.

**How to build and run** (from the repository root; the build context must be the
root so it can copy `ml/` and `etl/stats19_lookups.py`):

```bash
docker build -f backend/Dockerfile -t traffic-severity-api .
```

The container listens on port **9000**, while the local `uvicorn` command in this
project uses **9001**. Running the image requires the database URL and the MLflow
store to be reachable from inside the container; the exact `docker run` command
with its mounts is in [README.md](README.md).

## 15. Why Airflow

The brief also grades an **"Airflow DAG or approved orchestration workflow."**
`airflow/dags/traffic_accident_etl_dag.py` schedules the same `extract.py →
transform.py → load.py → quality_check.py` scripts as separate tasks, with two
retries five minutes apart, because both enrichment APIs are free-tier and
rate-limited, so transient failures are expected. It adds scheduling, retries and
a visual run history on top of the same code. It does not reimplement the
pipeline, so there is one definition of the ETL however it is triggered.
Airflow has no native Windows support, so [airflow/README.md](airflow/README.md)
explains running it through WSL.

## 16. Known limitations

- **Weather is gridded.** One 0.5° cell (roughly 35 × 55 km) stands in for every
  accident inside it, so local weather differences are smoothed over.
- **Street names are partial.** TomTom's free tier covers only the busiest
  coordinates, and the dashboard shows the real coverage percentage.
- **The model is modest** (section 12). It is built on conditions only and should
  be treated as a risk indicator.
- **Monitoring is intentionally simple.** Latency and input-quality trackers live
  in memory and reset on restart; there is no automatic retraining.
- **Model-vs-warehouse drift windows are approximate.** Drift and performance are
  computed over recent slices of the warehouse (the newest accidents), not a live
  production stream, because the data is historical.
- **Dashboard scope.** The dashboard has a Home page plus three consolidated pages
  (Risk Factors, Hotspot Map, Severity Model), not the eight single-chart pages in
  the original build plan. Check this against the rubric's "5+ views" requirement
  before submission; the pages contain multiple charts each.
