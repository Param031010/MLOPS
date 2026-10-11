# Report Outline — Traffic Accident Severity Analytics & Prediction

Scaffold only: section structure + facts pulled straight from the repo,
mapped to CLAUDE.md §12's submission checklist. Every `[WRITE: ...]` marker
is something only you can supply — narrative, judgment, or a screenshot.
Target length 8–10 pages; delete this instruction line before submitting.

---

## 1. Introduction & Objectives

- Individual project, Data Engineering and MLOps course. Part 1 (data
  pipeline + dashboard) and Part 2 (MLOps), 100 marks, due 2026-09-09.
- Combines UK STATS19 road-collision records with historical weather
  (Open-Meteo) and road context (TomTom) into a queryable warehouse,
  visualizes patterns on an 8-page dashboard, and (Part 2) trains/serves a
  severity-prediction model.
- [WRITE: 2-3 sentences of personal motivation/framing — why severity
  prediction, what question the project answers.]

## 2. Data Sources

| Source | Provides | Access |
|---|---|---|
| UK Road Safety Open Data (STATS19) | Collision, casualty, vehicle records | Local CSVs, DfT "last 5 years" export |
| Open-Meteo Historical Weather API | Weather at accident time/location | Free, no key, quota-limited/day |
| TomTom Reverse Geocoding | Road/street context | Free-tier key, quota-limited/day |

- Final volumes loaded: **513,748 accidents**, **652,763 casualties**,
  **937,164 vehicles**, spanning years **2021–2025**.
- [WRITE: why STATS19 was chosen, any licensing note required by your
  course (STATS19 is Open Government Licence).]

## 3. System Architecture

- Insert `docs/architecture-diagram.svg` here as a figure.
- Native local processes: PostgreSQL, ETL scripts, and the Streamlit
  dashboard each run in their own virtualenv. The FastAPI backend can also
  run inside `backend/Dockerfile` — a single-purpose container for just the
  inference/analytics API (Postgres and Streamlit are never containerized).
- Three ways to trigger the same `extract.py → transform.py → load.py →
  quality_check.py` pipeline: manual CLI, `POST /admin/run-etl`, or the
  Airflow DAG (`airflow/dags/traffic_accident_etl_dag.py`).
- [WRITE: one paragraph walking the diagram left-to-right for a reader who
  hasn't seen the code.]

## 4. Data Model

Star schema (`sql/schema.sql`), documented column-by-column in
`docs/DATA_DICTIONARY.md` (auto-generated from the live schema's
`COMMENT ON` text — cite/embed relevant tables rather than retyping them):

- **5 dimensions**: `dim_date` (42,170 rows — one per distinct collision
  hour), `dim_location` (512,309 rows, generated PostGIS geography column +
  optional TomTom `street_name`), `dim_road`, `dim_weather` (junk
  dimensions over STATS19 codes), `dim_severity` (4 rows: 1=Fatal,
  2=Serious, 3=Slight, -1=Data missing — STATS19's own missing-data code
  kept as a labeled category, never dropped).
- **3 facts**: `fact_accidents` (one row per collision), `fact_casualties`
  and `fact_vehicles` (separate grain — one collision injures several
  people/involves several vehicles, each with its own attributes).
- **`etl_audit_log`**: source, row count, rejected count, status per run.
- **10 `mart_*` materialized views**, one per dashboard page/analytics
  endpoint, refreshed at the end of every `load.py` run.
- [WRITE: a `psql \d` or pgAdmin screenshot of a populated table, per
  CLAUDE.md §12.]

## 5. ETL Pipeline & Validation Rules

- **`extract.py`**: immutable timestamped snapshot before any cleaning;
  weather snapped to a 0.5° grid, one Open-Meteo request per (cell, year);
  TomTom spent busiest-coordinate-first under a quota; every stage appended
  to the audit log.
- **`transform.py`** rejection rules (each logged with count + reason):
  duplicate `accident_index`, missing/null coordinates, coordinates outside
  GB's bounding box (lat 49.8–60.9, lon -8.6–1.8), unparseable date/time.
  STATS19 codes decoded via `stats19_lookups.py`; weather joined on grid
  cell **and** exact hour; road context joined on exact coordinate.
- **`load.py`**: one transaction, dimensions first (`ON CONFLICT DO
  NOTHING`), then facts upserted against resolved surrogate keys, then all
  10 mart views refreshed.
- **`quality_check.py`** (post-load gate, also the Airflow DAG's final
  task): fails if any table/mart is empty, any fact row fails to resolve a
  dimension, or the last audit-log row isn't `success`.
- Known, honestly-surfaced coverage gaps (free-tier quotas, not a bug):
  weather coverage started at **35.0%** and reached **100%** (513,748 of
  513,748 accidents) after repeated resumable `extract` runs pulled the
  previously unfetched Open-Meteo grid-year cells — direct evidence the
  pipeline is genuinely resumable/rerunnable, not a one-shot script.
  Road-context (TomTom) coverage is **0.29%** (1,510 accidents) because the
  free-tier quota is spent busiest-coordinate-first; both are reported live
  via `/accidents/overview` and the dashboard home page rather than hidden.
- **Rejected-record log**: `transform.py` writes
  `etl/data/audit/rejected_records.csv` — one row per dropped collision with
  its reason (latest run: 53 rejected, all `missing_coordinates`) — alongside
  the per-reason counts in `audit_log.jsonl` / `etl_audit_log`.
- **Model-ready table**: view `vw_model_features` (one row per collision,
  features + label), which `ml/train.py` reads.
- **A real data-quality finding** (worth a paragraph in the report): the
  input-data-quality monitor flagged `junction_detail` — 45% of collisions
  decoded as "Unknown code N". Root cause: the DfT export carries both a
  historic junction column (codes 0–9) and a renumbered new-scheme column
  (13, 16, 17, 18, 19). Cross-tabulating them in the raw data proved the
  mapping (13↔3, 16↔6, 17↔7, 18↔8, 19↔9), so `stats19_lookups.py` now merges
  both onto one vocabulary and `9`/`99` decode as "Unknown (self reported)".
  The warehouse was reloaded and the model retrained; the monitor now flags
  nothing.
- [WRITE: `load.py`'s final row-count console output as a screenshot —
  this is the assignment's "successful end-to-end pipeline execution"
  evidence.]

## 6. Orchestration

- Same three trigger paths as §3: CLI, `POST /admin/run-etl` (token-gated
  background task, status polled via `GET /admin/run-etl/status`), and the
  Airflow DAG.
- Airflow DAG: `extract >> transform >> load >> data_quality_check`, daily
  schedule, idempotent reruns (facts upsert on business key). Needs WSL2 or
  Docker on Windows — see `airflow/README.md`.
- **Actually run end-to-end, 2026-09-08** (WSL2 Ubuntu, Airflow 3.3.1): the
  scheduler auto-picked up the `0 3 * * *` run once unpaused; all four tasks
  completed **success** against the real pipeline — `extract` 08:30:11 to
  09:20:30 (~50 min, genuine Open-Meteo backfill of the previously-unfetched
  grid-year cells), `transform` 09:20:31–09:21:33, `load` 09:21:35–09:28:24,
  `data_quality_check` 09:28:25–09:28:29. Not a dry run: this re-populated
  the live warehouse.
- [WRITE: confirm with the instructor whether the script-based pipeline
  (with Airflow as one optional trigger among three) satisfies "Airflow DAG
  or approved equivalent," and note the answer here.]
- [WRITE: screenshot the Airflow UI (`http://localhost:8080`) showing this
  green run — the graph/grid view for `traffic_accident_etl`'s
  `scheduled__2026-09-08T03:00:00+00:00` run is the strongest evidence.]

## 7. Backend API & Dashboard

- FastAPI backend, 21 endpoints total (accidents/casualties/vehicles reads
  over the mart views, `/admin/run-etl*`, and Part 2's `/predict`,
  `/model/*`, `/monitoring/*`). Full list in README.md §7.
- 8-page Streamlit dashboard, each page independently querying the live
  API with its own filters — no bundled/placeholder data anywhere.
  `5_Hotspot_Map.py` uses `st.pydeck_chart`; the rest use Plotly.
- [WRITE: 2-3 dashboard screenshots — Home KPI strip, Hotspot Map, and
  Severity Predictor are the most visually convincing choices.]
- [WRITE: a terminal/Swagger screenshot of a real `POST /predict` call and
  response, per CLAUDE.md §12's execution-evidence requirement.]

## 8. Part 2 — MLOps

- **Models**: RandomForest, XGBoost, LightGBM, all three trained with class
  weighting for severity-class imbalance.
- **Split**: chronological (earlier years train, most recent years
  validate/test) — a deliberate choice since this is a time series of
  future accidents, not exchangeable rows.
- **Tracking**: every run's params/metrics/artifacts logged to MLflow under
  experiment `traffic-accident-severity`; the macro-F1 winner registered as
  `traffic_severity_classifier` and staged `Production`, archiving the
  prior version on promotion.
- **Packaging**: preprocessing and model are one registered artifact
  (`ml/pipeline.py`, `SeverityPipeline`): raw record in -> feature
  engineering -> model -> class probabilities, with the feature code bundled
  in the MLflow model (`code_paths`) so it loads in the Docker image without
  the repo on the path. Missing weather is `NaN` at both training and
  serving time.
- **Current Production model** (v4, 2026-10-11): **RandomForest** won on
  validation macro-F1 (RandomForest 0.371, LightGBM 0.349, XGBoost 0.348).
  Test split: accuracy **0.55**, macro-F1 **0.364**, recall (macro)
  **0.461**; recall by class Fatal **0.43**, Serious **0.32**, Slight
  **0.64**; false-negative rate on Fatal/Serious **0.48**. It predicts all
  three classes and trades precision for recall on the rare severities, but
  roughly half of Fatal/Serious accidents are still predicted as Slight —
  a limit of the available features (no driver, speed or vehicle-condition
  information is known beforehand), and it exceeds the 15% retraining
  trigger, which the monitoring page shows deliberately.
- **Evaluation**: `evaluate.py` — recall, macro-F1, confusion matrix logged
  as MLflow artifacts. [WRITE: embed `ml/artifacts/confusion_matrix.png`
  and a sentence interpreting it — which severities get confused with
  which.]
- **Serving**: loaded at FastAPI startup via a `lifespan` handler; the
  backend must be restarted after training to pick up a newly promoted
  version.
- **Monitoring**: `/monitoring/drift` (PSI across feature/location/class
  vs. a training baseline), `/monitoring/performance` (rolling macro-F1/
  recall/false-negative rate on the most recent accidents),
  `/monitoring/data-quality` (missing / out-of-range / unseen-category
  rates, warehouse window + live requests), `/monitoring/latency` (serving
  p50/p95 + failure rate) — all real computed metrics, not placeholders.
  Drift is measured on the most recent ~1 year (100k accidents) so
  seasonal weather doesn't read as drift. `8_Model_Monitoring.py`
  visualizes all of them with the thresholds below drawn in.
- **Retraining policy** (`ml/RETRAINING.md`): retrain if rolling macro-F1
  drops **>5 points**, false-negative rate on Fatal/Serious exceeds
  **15%**, PSI exceeds **0.2** on any feature, a full new year of data
  lands, or `ml/features.py`'s vocabulary changes. Retraining is
  intentionally manual, not automated — STATS19 arrives on DfT's release
  cycle, not in real time.
- **Docker**: `backend/Dockerfile` containerizes only the inference API
  (Postgres/dashboard stay native), built to satisfy the assignment's
  Docker requirement specifically. Built and run-tested 2026-09-08 against
  the host Postgres and the mounted MLflow registry — `/health`,
  `/model/info`, `/model/health`, a real `/predict` call, and
  `/accidents/overview` all verified working from inside the container
  (see README.md's Docker section for the exact command and the
  absolute-path mount gotcha that had to be worked around).
- [WRITE: `docker images` / `docker ps` screenshot, per CLAUDE.md §12.]

## 9. Results & Discussion

- [WRITE: interpret the model's metrics in plain language — is 0.364
  macro-F1 "good" for this problem? Compare to a naive baseline (e.g.
  always predicting the majority class "Slight") if you want a concrete
  anchor.]
- [WRITE: what the enrichment-coverage gaps (weather now complete, 0.29% road
  context) mean for the model and dashboard — does the model even use
  `street_name`? If not, road-context coverage matters more for the
  dashboard than for prediction quality; say so.]

## 10. Limitations & Future Work

- [WRITE: free-tier API quotas as a structural limit, not a bug — what
  would change with a paid tier.]
- [WRITE: anything you'd do differently with more time — e.g. automated
  retraining, a paid weather/geocoding tier, more model families,
  hyperparameter search.]

## 11. Conclusion

- [WRITE: 3-4 sentences tying objectives (§1) back to what was delivered.]

## Appendix — Screenshots / Execution Evidence Checklist

Per CLAUDE.md §12, collect these before assembling the final document:

- [ ] `load.py`'s final row-count console output
- [ ] `psql \d` (or pgAdmin) of a populated table
- [ ] A working `POST /predict` call and its response
- [ ] At least one dashboard page in the browser
- [ ] `docker images` / `docker ps` showing `traffic-severity-api`
- [ ] (if run) an Airflow UI screenshot of a successful DAG run
