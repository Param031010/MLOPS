"""Post-load data-quality gate.

Runs after load.py and exits non-zero if the warehouse is not in a state worth
serving, so an orchestrator (the Airflow DAG, CI, or a manual run) fails loudly
instead of leaving a half-loaded database behind a green tick.

Usage:
    python quality_check.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> int:
    url = os.getenv("DATABASE_URL")
    if not url:
        print("FAIL  DATABASE_URL is not set (check your .env)")
        return 1

    engine = create_engine(url)
    failures: list[str] = []

    with engine.connect() as conn:
        counts = {
            table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
            for table in ["fact_accidents", "fact_casualties", "fact_vehicles",
                          "dim_date", "dim_location", "dim_road", "dim_weather"]
        }
        for table, n in counts.items():
            print(f"  {table:18s} {n:>10,}")
            if n == 0:
                failures.append(f"{table} is empty")

        last_load = conn.execute(text("""
            SELECT status, row_count FROM etl_audit_log
            WHERE stage = 'load' ORDER BY audit_id DESC LIMIT 1
        """)).first()
        if last_load is None:
            failures.append("no load run recorded in etl_audit_log")
        else:
            print(f"  last load status   {last_load.status} ({last_load.row_count:,} rows)")
            if last_load.status != "success":
                failures.append(f"most recent load audit row is '{last_load.status}'")

        # Every fact row must resolve to real dimension rows. The load's own
        # inner joins should guarantee this, so a hit here means the warehouse
        # was modified outside the pipeline.
        orphans = conn.execute(text("""
            SELECT COUNT(*) FROM fact_accidents fa
            WHERE NOT EXISTS (
                SELECT 1 FROM dim_date d WHERE d.date_key = fa.date_key
            )
        """)).scalar_one()
        print(f"  orphaned accidents {orphans:>10,}")
        if orphans:
            failures.append(f"{orphans} fact_accidents rows have no dim_date row")

        # The marts drive every dashboard page; an empty one means the refresh
        # did not run even though the load reported success.
        empty_marts = conn.execute(text("""
            SELECT matviewname FROM pg_matviews WHERE matviewname LIKE 'mart\\_%'
            ORDER BY matviewname
        """)).scalars().all()
        for mart in empty_marts:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {mart}")).scalar_one()
            if n == 0:
                failures.append(f"{mart} is empty")

    if failures:
        print("\nFAIL  data-quality check:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPASS  warehouse and marts look healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
