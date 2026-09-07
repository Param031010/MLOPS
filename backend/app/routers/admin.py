"""POST /admin/run-etl triggers extract -> transform -> load as a background
task by shelling out to the etl/ scripts (they live in their own virtualenv,
deliberately kept separate from the backend's pinned FastAPI stack).
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException

from ..schemas import EtlRunStatus

router = APIRouter(prefix="/admin", tags=["admin"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ETL_DIR = PROJECT_ROOT / "etl"

_status: dict = {"status": "idle", "run_id": None, "started_at": None,
                  "finished_at": None, "message": None}


def _etl_python() -> str:
    override = os.getenv("ETL_PYTHON")
    if override:
        return override
    venv_python = ETL_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv_python) if venv_python.exists() else sys.executable


def _run_pipeline(run_id: str) -> None:
    python = _etl_python()
    steps = ["extract.py", "transform.py", "load.py"]
    for step in steps:
        result = subprocess.run(
            [python, str(ETL_DIR / step)],
            cwd=str(ETL_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _status.update(
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                message=f"{step} failed (exit {result.returncode}): {result.stderr[-2000:]}",
            )
            return
    _status.update(
        status="success",
        finished_at=datetime.now(timezone.utc).isoformat(),
        message="extract -> transform -> load completed",
    )


@router.post("/run-etl", response_model=EtlRunStatus)
async def run_etl(background_tasks: BackgroundTasks, x_admin_token: str | None = Header(None)):
    expected = os.getenv("ADMIN_TOKEN")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Token")
    if _status["status"] == "running":
        raise HTTPException(status_code=409, detail="An ETL run is already in progress")
    if not ETL_DIR.exists():
        raise HTTPException(status_code=503, detail="etl/ directory not available in this deployment")

    run_id = uuid.uuid4().hex[:12]
    _status.update(status="running", run_id=run_id,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    finished_at=None, message=None)
    background_tasks.add_task(_run_pipeline, run_id)
    return EtlRunStatus(**_status)


@router.get("/run-etl/status", response_model=EtlRunStatus)
async def run_etl_status():
    return EtlRunStatus(**_status)
