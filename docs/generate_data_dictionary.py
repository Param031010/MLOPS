"""Generate docs/DATA_DICTIONARY.md from the live database schema.

The descriptions come from the COMMENT ON statements at the bottom of
sql/schema.sql, so the dictionary is derived from the database rather than
hand-maintained alongside it -- there is no second copy to drift out of date.

Usage (from the repository root, using the etl virtualenv):
    etl/.venv/Scripts/python docs/generate_data_dictionary.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
OUT_PATH = ROOT / "docs" / "DATA_DICTIONARY.md"

TABLE_ORDER = [
    "dim_severity", "dim_date", "dim_location", "dim_road", "dim_weather",
    "fact_accidents", "fact_casualties", "fact_vehicles", "etl_audit_log",
]

COLUMNS_SQL = """
    SELECT c.column_name,
           c.data_type,
           c.character_maximum_length,
           c.numeric_precision,
           c.numeric_scale,
           c.is_nullable,
           c.column_default,
           col_description(CAST(:table AS regclass), c.ordinal_position) AS description
    FROM information_schema.columns c
    WHERE c.table_name = :table AND c.table_schema = 'public'
    ORDER BY c.ordinal_position
"""

KEYS_SQL = """
    SELECT kcu.column_name, tc.constraint_type,
           ccu.table_name AS foreign_table, ccu.column_name AS foreign_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
    LEFT JOIN information_schema.constraint_column_usage ccu
      ON ccu.constraint_name = tc.constraint_name AND tc.constraint_type = 'FOREIGN KEY'
    WHERE tc.table_name = :table AND tc.table_schema = 'public'
      AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE')
"""


def render_type(row) -> str:
    """Readable SQL type, including precision where it carries meaning."""
    base = row.data_type.replace("character varying", "varchar").replace(
        "timestamp with time zone", "timestamptz")
    if base == "numeric" and row.numeric_precision:
        return f"numeric({row.numeric_precision},{row.numeric_scale})"
    if base == "varchar" and row.character_maximum_length:
        return f"varchar({row.character_maximum_length})"
    return base


def key_flags(keys, column: str) -> str:
    flags = []
    for k in keys:
        if k.column_name != column:
            continue
        if k.constraint_type == "PRIMARY KEY":
            flags.append("PK")
        elif k.constraint_type == "FOREIGN KEY":
            flags.append(f"FK -> {k.foreign_table}.{k.foreign_column}")
        elif k.constraint_type == "UNIQUE":
            flags.append("UQ")
    # a column can appear in several constraints; keep each label once
    return ", ".join(dict.fromkeys(flags))


def main() -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (check your .env)")
    engine = create_engine(url)

    lines: list[str] = [
        "# Data Dictionary",
        "",
        "Every table and column of the traffic-accident warehouse, generated from",
        "the live PostgreSQL schema by `docs/generate_data_dictionary.py`.",
        "Descriptions are the `COMMENT ON` values defined in `sql/schema.sql`.",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        "Validation rules applied before a row reaches these tables are listed in",
        "[Validation rules](#validation-rules) at the end.",
        "",
    ]

    with engine.connect() as conn:
        present = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)).scalars().all()
        marts = conn.execute(text("""
            SELECT matviewname FROM pg_matviews WHERE matviewname LIKE 'mart%'
            ORDER BY matviewname
        """)).scalars().all()

        lines.append("## Tables\n")
        for table in TABLE_ORDER:
            if table not in present:
                continue
            table_desc = conn.execute(
                text("SELECT obj_description(CAST(:t AS regclass))"), {"t": table}).scalar_one()
            rows = conn.execute(text(COLUMNS_SQL), {"table": table}).fetchall()
            keys = conn.execute(text(KEYS_SQL), {"table": table}).fetchall()
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()

            lines.append(f"### `{table}`\n")
            if table_desc:
                lines.append(f"{table_desc}\n")
            lines.append(f"*{n:,} rows at generation time.*\n")
            lines.append("| Column | Type | Null | Key | Description |")
            lines.append("|---|---|---|---|---|")
            for r in rows:
                lines.append(
                    f"| `{r.column_name}` | {render_type(r)} | "
                    f"{'yes' if r.is_nullable == 'YES' else 'no'} | "
                    f"{key_flags(keys, r.column_name) or ''} | "
                    f"{r.description or ''} |"
                )
            lines.append("")

        lines.append("## Analytical layer (materialized views)\n")
        lines.append("Each view backs one dashboard page and is refreshed at the end of "
                     "every `load.py` run.\n")
        lines.append("| View | Columns | Rows |")
        lines.append("|---|---|---|")
        for mart in marts:
            cols = conn.execute(text("""
                SELECT string_agg(column_name, ', ' ORDER BY ordinal_position)
                FROM information_schema.columns
                WHERE table_name = :t AND table_schema = 'public'
            """), {"t": mart}).scalar_one()
            n = conn.execute(text(f"SELECT COUNT(*) FROM {mart}")).scalar_one()
            lines.append(f"| `{mart}` | {cols} | {n:,} |")
        lines.append("")

    lines += [
        "## Validation rules",
        "",
        "Applied by `etl/transform.py`. Every rejection is counted by reason and",
        "written to both `etl/data/audit/audit_log.jsonl` and `etl_audit_log`.",
        "",
        "| Rule | Action on failure |",
        "|---|---|",
        "| `collision_index` must be unique | Duplicate rows dropped, first kept |",
        "| `latitude` and `longitude` must be present | Row rejected |",
        "| Coordinates must fall in Great Britain's bounding box (lat 49.8-60.9, lon -8.6-1.8) | Row rejected |",
        "| `date` and `time` must parse (day-first date, `HH:MM` time) | Row rejected |",
        "| Casualty and vehicle rows must reference a surviving collision | Row dropped before load, so the load cannot violate a foreign key |",
        "| STATS19 numeric codes must decode to a known label | Unmapped codes become `Unknown code N` rather than raising, so a new code is visible instead of fatal |",
        "| `-1` (\"data missing or out of range\") | Preserved as its own labelled category, never silently dropped |",
        "",
        "Post-load, `etl/quality_check.py` gates the run: it fails if any table or",
        "mart is empty, if a fact row cannot resolve its dimension, or if the most",
        "recent `etl_audit_log` load row is not `success`.",
        "",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
