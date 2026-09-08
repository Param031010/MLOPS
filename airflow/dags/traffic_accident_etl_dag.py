"""Airflow DAG orchestrating the traffic-accident ETL.

The pipeline stages are ordinary scripts (``etl/extract.py``,
``etl/transform.py``, ``etl/load.py``) that are runnable on their own, and this
DAG schedules them rather than reimplementing them. That keeps one definition of
the pipeline: the same code runs whether it is triggered by Airflow, by
``POST /admin/run-etl``, or by hand from a terminal.

Each stage runs through the ``etl/`` virtualenv's interpreter, which is
deliberately kept separate from Airflow's own dependencies (Airflow pins its
own pandas/SQLAlchemy versions and would otherwise fight the pipeline's).

Configuration (Airflow Variables, or the matching environment variables):
    traffic_project_root  absolute path to the repository root
    traffic_etl_python    absolute path to the interpreter for the etl/ venv
    traffic_road_limit    max new TomTom coordinates to resolve per run

See airflow/README.md for how to run this on Windows, where Airflow has no
native support.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator


def _cfg(name: str, default: str) -> str:
    """Airflow Variable if one is set, else the environment, else the default.

    Reading through Variable.get with a default keeps the DAG importable on a
    fresh Airflow install that has no Variables configured yet -- an import-time
    failure would make the DAG invisible in the UI rather than merely
    misconfigured.
    """
    try:
        return Variable.get(name, default_var=os.getenv(name.upper(), default))
    except Exception:  # noqa: BLE001 - metadata DB unavailable at parse time
        return os.getenv(name.upper(), default)


PROJECT_ROOT = _cfg("traffic_project_root", "/opt/airflow/project")
ETL_DIR = f"{PROJECT_ROOT}/etl"
ETL_PYTHON = _cfg("traffic_etl_python", "python")
ROAD_LIMIT = _cfg("traffic_road_limit", "300")

default_args = {
    "owner": "param",
    "depends_on_past": False,
    # Both enrichment APIs are free-tier and rate-limited, so a transient 429 or
    # network blip is expected rather than exceptional. extract.py checkpoints
    # its progress, so a retry resumes instead of refetching.
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="traffic_accident_etl",
    description="STATS19 + Open-Meteo + TomTom -> PostgreSQL star schema and marts",
    default_args=default_args,
    start_date=datetime(2026, 9, 1),
    # Daily, but the DAG is safe to trigger manually at any time: load.py
    # upserts on the accident business key and refreshes the marts, so a rerun
    # over unchanged data is a no-op rather than a duplicate load.
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["etl", "traffic", "stats19"],
) as dag:

    extract = BashOperator(
        task_id="extract",
        bash_command=(
            f"cd {ETL_DIR} && {ETL_PYTHON} extract.py --road-limit {ROAD_LIMIT}"
        ),
        doc_md=(
            "Reads the STATS19 CSVs, writes an immutable timestamped snapshot to "
            "the raw zone, then enriches: one Open-Meteo request per (0.5 degree "
            "grid cell, year), and TomTom reverse geocoding spent "
            "busiest-coordinate-first. Checkpoints as it goes and stops cleanly "
            "on a quota refusal, so the next run resumes."
        ),
    )

    transform = BashOperator(
        task_id="transform",
        bash_command=f"cd {ETL_DIR} && {ETL_PYTHON} transform.py",
        doc_md=(
            "Cleans and decodes the snapshot into the staging layer: drops "
            "duplicate keys, missing/out-of-GB coordinates and unparseable "
            "timestamps, decodes STATS19 numeric codes to labels, joins weather "
            "on grid cell and exact hour, and logs every rejection with its "
            "reason to the audit log."
        ),
    )

    load = BashOperator(
        task_id="load",
        bash_command=f"cd {ETL_DIR} && {ETL_PYTHON} load.py",
        doc_md=(
            "Bulk-loads staging into the warehouse in one transaction: "
            "dimensions first, then facts upserted against them, then refreshes "
            "the ten mart materialized views and writes a summary row to "
            "etl_audit_log."
        ),
    )

    # A load that "succeeds" while leaving the warehouse empty, the marts
    # unrefreshed or the audit log reporting a failure should fail the DAG run,
    # not pass silently. quality_check.py is a normal script, so the same gate
    # is runnable by hand or from CI.
    data_quality_check = BashOperator(
        task_id="data_quality_check",
        bash_command=f"cd {ETL_DIR} && {ETL_PYTHON} quality_check.py",
        doc_md=(
            "Gate on the loaded warehouse: fails the run if any table or mart is "
            "empty, if a fact row fails to resolve its dimension, or if the most "
            "recent load audit row is not 'success'."
        ),
    )

    extract >> transform >> load >> data_quality_check
