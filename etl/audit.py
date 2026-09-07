"""Shared JSONL audit logging for extract/transform/load."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = Path(__file__).parent / "data" / "audit" / "audit_log.jsonl"


def log_event(
    run_id: str,
    source: str,
    stage: str,
    status: str,
    row_count: int | None = None,
    rejected_count: int | None = None,
    message: str = "",
    started_at: datetime | None = None,
) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    finished_at = datetime.now(timezone.utc)
    entry = {
        "run_id": run_id,
        "source": source,
        "stage": stage,
        "row_count": int(row_count) if row_count is not None else None,
        "rejected_count": int(rejected_count) if rejected_count is not None else None,
        "status": status,
        "message": message,
        "started_at": (started_at or finished_at).isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
