# Airflow DAG — `traffic_accident_etl`

This DAG (`airflow/dags/traffic_accident_etl_dag.py`) doesn't reimplement
the pipeline — it schedules the same `etl/extract.py`, `etl/transform.py`,
`etl/load.py`, and `etl/quality_check.py` scripts that also run standalone
from the CLI or via `POST /admin/run-etl`. One definition of the pipeline,
three ways to trigger it.

## Why this needs WSL2 or Docker on Windows

Apache Airflow has no native Windows support (its scheduler relies on POSIX
`fork()`). Since the rest of this project runs natively on Windows, Airflow
is the one piece that needs either:

- **WSL2** (recommended) — install Airflow inside a WSL2 Ubuntu instance and
  point it at the Windows filesystem's copy of this repo (`/mnt/c/Users/...`
  or wherever it's mounted), or
- **Docker** — run Airflow via its own official image (`apache/airflow`),
  separate from and unrelated to `backend/Dockerfile` (which only
  containerizes the FastAPI inference service, not Airflow).

Either way, Airflow orchestrates the pipeline from outside — it never needs
to run *inside* the same container/environment as the ETL scripts
themselves, since each task just shells out to the `etl/` virtualenv's
interpreter.

## Setup (WSL2 path) — verified 2026-09-08

```bash
# inside WSL2
python3 -m venv ~/airflow-venv
~/airflow-venv/bin/pip install apache-airflow   # resolved to 3.3.1; installs fine even on Python 3.14
export AIRFLOW_HOME=~/airflow
export PATH=~/airflow-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
airflow db migrate     # Airflow 3.x renamed `db init` to `db migrate`

# point Airflow at this DAG without copying it
mkdir -p $AIRFLOW_HOME/dags
ln -s /mnt/c/Users/PARAM/OneDrive/Desktop/MLOPS1/airflow/dags/traffic_accident_etl_dag.py \
  $AIRFLOW_HOME/dags/traffic_accident_etl_dag.py
```

No `airflow users create` step is needed: `airflow standalone` (below) uses
Airflow 3's `simple_auth_manager` and auto-generates an admin login on first
start, writing the password to
`$AIRFLOW_HOME/simple_auth_manager_passwords.json.generated`.

**Two gotchas that will bite you if skipped:**
- **Set `PATH` explicitly**, as above — a non-login `wsl -e bash -c "..."`
  shell doesn't reliably inherit a full `PATH` in every calling context.
  `airflow standalone` spawns its own subprocesses (`airflow scheduler`,
  `bash -c ...` for each `BashOperator` task) and needs *both*
  `~/airflow-venv/bin` (to find `airflow`) *and* the standard system dirs
  (to find `bash`) on `PATH` — prepending to a possibly-empty `$PATH` isn't
  enough; set the whole thing explicitly.
- **From Windows Git Bash**, MSYS auto-converts bare Linux paths like
  `/root/airflow` into a Windows path before `wsl.exe` sees them — prefix
  those commands with `MSYS_NO_PATHCONV=1`, or run them from PowerShell
  instead.

## Required Airflow Variables (or environment variables)

The DAG reads each of these as an Airflow Variable first, falling back to
the uppercased environment variable, falling back to a default:

| Variable | Purpose | Example |
|---|---|---|
| `traffic_project_root` | Absolute path to the repo root, as seen from where Airflow runs | `/mnt/c/Users/PARAM/OneDrive/Desktop/MLOPS1` |
| `traffic_etl_python` | Absolute path to the `etl/` virtualenv's interpreter — deliberately separate from Airflow's own pinned dependencies | `/mnt/c/Users/PARAM/OneDrive/Desktop/MLOPS1/etl/.venv/Scripts/python.exe` |
| `traffic_road_limit` | Max new TomTom coordinates to resolve per run | `300` |

Set them via the Airflow UI (Admin → Variables) or the CLI:

```bash
airflow variables set traffic_project_root /mnt/c/Users/PARAM/OneDrive/Desktop/MLOPS1
airflow variables set traffic_etl_python /mnt/c/Users/PARAM/OneDrive/Desktop/MLOPS1/etl/.venv/Scripts/python.exe
airflow variables set traffic_road_limit 300
```

Also make sure `DATABASE_URL` and the other `.env` values are available to
that interpreter — the ETL scripts read them via `python-dotenv` from the
repo's `.env`, same as when run by hand.

## Running it

```bash
export PATH=~/airflow-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__LOAD_EXAMPLES=false   # keep the DAG list free of Airflow's sample DAGs
airflow standalone       # dev-only: scheduler + webserver + a generated admin login
# UI at http://localhost:8080 (WSL2 forwards localhost to Windows automatically)
```

Airflow 3.x moved `BashOperator` into `apache-airflow-providers-standard`;
this DAG's `from airflow.operators.bash import BashOperator` still resolves
via a deprecation-warning back-compat shim, so it runs unmodified on both
Airflow 2.x and 3.x.

In the UI, unpause `traffic_accident_etl` and trigger it manually
(the ▶ button) to produce a run for screenshot evidence — no need to wait
for the daily `0 3 * * *` schedule. The four tasks run in order:
`extract → transform → load → data_quality_check`; `quality_check.py` is
the final gate and fails the DAG run if any warehouse table or mart view is
empty, a fact row failed to resolve its dimension, or the last load's audit
row isn't `success`.

## "Approved equivalent" note

The assignment brief asks for "an Apache Airflow DAG, or an approved
equivalent ETL workflow." This repo's pipeline is fully runnable and
orchestrable without Airflow (CLI, or `POST /admin/run-etl`), with this DAG
as an additional, optional trigger path on top. Whether that combination
counts as the required "approved equivalent" on its own, or whether the
Airflow DAG itself needs to actually be run and screenshotted, is worth
confirming with the instructor before submission.
