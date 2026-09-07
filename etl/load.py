"""Load stage: bulk-load staging parquet into temp tables, populate
dimensions, upsert facts, refresh the mart, all in one transaction.
"""

from __future__ import annotations

import io
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent))
from audit import log_event  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

STAGING_DIR = Path(__file__).parent / "data" / "staging"

ROAD_COLS = [
    "first_road_class", "road_type", "speed_limit", "junction_detail",
    "junction_control", "second_road_class", "pedestrian_crossing",
    "special_conditions_at_site", "carriageway_hazards", "trunk_road_flag",
]
WEATHER_COLS = [
    "weather_conditions", "road_surface_conditions", "light_conditions",
    "temperature_c", "precipitation_mm", "wind_speed_kmh", "weather_code_meteo",
]


# Columns of the road/weather dimensions that arrive as numbers rather than
# decoded labels. They are the only ones that can be NULL: transform.py's
# decode_series() always emits a label ("Data missing", "Unknown code N")
# for the text columns.
# Numeric dimension columns, mapped to the decimal scale their column uses
# in sql/schema.sql. The staged side must be rounded to the same scale before
# comparing: the dimension rounds on insert (NUMERIC(5,2) turns 5.404 into
# 5.40), so an unrounded staged value would fail to match its own dimension
# row and the accident would be silently dropped by the INNER JOIN.
NUMERIC_DIM_SCALES = {
    "speed_limit": 0,
    "weather_code_meteo": 0,
    "temperature_c": 2,
    "precipitation_mm": 2,
    "wind_speed_kmh": 2,
}
TEXT_NULL_SENTINEL = "\x01__null__"
NUMERIC_NULL_SENTINEL = -999999


def dimension_join(staged: str, dim: str, cols: list[str]) -> str:
    """Build the ON clause that resolves a staged row to its dimension row.

    Uses COALESCE(...) = COALESCE(...) rather than IS NOT DISTINCT FROM:
    the latter is not a hashable equality operator, so Postgres falls back
    to a nested loop over the whole dimension -- which on ~514k staged rows
    against a ~180k-row dim_weather does not finish in any useful time.
    Equality on a COALESCEd expression hash-joins instead.
    """
    parts = []
    for c in cols:
        if c in NUMERIC_DIM_SCALES:
            scale = NUMERIC_DIM_SCALES[c]
            left = f"COALESCE(ROUND({dim}.{c}::numeric, {scale}), {NUMERIC_NULL_SENTINEL})"
            right = f"COALESCE(ROUND({staged}.{c}::numeric, {scale}), {NUMERIC_NULL_SENTINEL})"
        else:
            left = f"COALESCE({dim}.{c}, '{TEXT_NULL_SENTINEL}')"
            right = f"COALESCE({staged}.{c}, '{TEXT_NULL_SENTINEL}')"
        parts.append(f"{left} = {right}")
    return " AND ".join(parts)


def bulk_load(conn, df: pd.DataFrame, table: str) -> None:
    """Create the staging table from the DataFrame's schema, then stream the
    rows in with COPY.

    pandas' to_sql() issues row-by-row INSERTs, which takes many minutes for
    the ~937k-row vehicles table; COPY does the same work in seconds.
    """
    df.head(0).to_sql(table, conn, if_exists="replace", index=False)

    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)

    raw = getattr(conn.connection, "driver_connection", conn.connection)
    with raw.cursor() as cur:
        cur.copy_expert(
            f'COPY "{table}" FROM STDIN WITH (FORMAT CSV, NULL \'\\N\')', buf
        )
    print(f"  copied {len(df):,} rows into {table}")


def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (check your .env)")
    return create_engine(url)


def main() -> None:
    run_id = uuid.uuid4().hex[:12]
    started = datetime.now(timezone.utc)
    print(f"[load] run_id={run_id}")

    accidents = pd.read_parquet(STAGING_DIR / "accidents_staged.parquet")
    casualties = pd.read_parquet(STAGING_DIR / "casualties_staged.parquet")
    vehicles = pd.read_parquet(STAGING_DIR / "vehicles_staged.parquet")

    engine = get_engine()
    with engine.begin() as conn:
        # Postgres' default work_mem (4MB) makes this entire transaction spill
        # to disk. The fact upsert hash-joins ~514k staged rows against a
        # ~512k-row dim_location and a ~139k-row dim_weather, and each
        # dimension's INSERT ... SELECT DISTINCT sorts the same volume: a load
        # at the defaults was measured writing ~1.4GB across 379 temp files,
        # and the mart refresh that follows was hit even harder. Both settings
        # are session-local (SET LOCAL) and revert when this transaction ends.
        conn.execute(text("SET LOCAL work_mem = '256MB'"))
        conn.execute(text("SET LOCAL maintenance_work_mem = '512MB'"))

        bulk_load(conn, accidents, "tmp_accidents_staged")
        bulk_load(conn, casualties, "tmp_casualties_staged")
        bulk_load(conn, vehicles, "tmp_vehicles_staged")

        # ---- dimensions ----
        conn.execute(text("""
            INSERT INTO dim_date (date_key, full_date, hour, day_of_week, day_name,
                                   day_of_month, month, month_name, quarter, year, is_weekend)
            SELECT DISTINCT date_key, full_date::date, hour, day_of_week, day_name,
                   day_of_month, month, month_name, quarter, year, is_weekend::boolean
            FROM tmp_accidents_staged
            ON CONFLICT (date_key) DO NOTHING
        """))

        conn.execute(text("""
            INSERT INTO dim_location (latitude, longitude, location_easting_osgr,
                                       location_northing_osgr, lsoa_of_accident_location,
                                       local_authority_district, local_authority_ons_district,
                                       local_authority_highway, police_force,
                                       urban_or_rural_area, street_name)
            SELECT DISTINCT ON (latitude, longitude)
                   latitude, longitude, location_easting_osgr, location_northing_osgr,
                   lsoa_of_accident_location, local_authority_district,
                   local_authority_ons_district, local_authority_highway, police_force,
                   urban_or_rural_area, street_name
            FROM tmp_accidents_staged
            -- Not DO NOTHING: TomTom's quota is spent a few hundred coordinates
            -- per run, so later runs legitimately carry street names for
            -- coordinates that are already in the dimension. DO NOTHING silently
            -- discarded every one of those, freezing road-context coverage at
            -- whatever the first load happened to resolve. COALESCE keeps an
            -- existing name when a rerun has no enrichment for that coordinate,
            -- so coverage only ever moves up.
            ON CONFLICT (latitude, longitude) DO UPDATE SET
                street_name = COALESCE(EXCLUDED.street_name, dim_location.street_name),
                road_context_resolved_at = CASE
                    WHEN EXCLUDED.street_name IS NOT NULL THEN now()
                    ELSE dim_location.road_context_resolved_at
                END
        """))

        road_cols_sql = ", ".join(ROAD_COLS)
        conn.execute(text(f"""
            INSERT INTO dim_road ({road_cols_sql})
            SELECT DISTINCT {road_cols_sql}
            FROM tmp_accidents_staged
            ON CONFLICT ({road_cols_sql}) DO NOTHING
        """))

        weather_cols_sql = ", ".join(WEATHER_COLS)
        # is_raining/is_freezing are derived from the columns above, so adding
        # them to the projection can't split a group the DISTINCT would have
        # merged -- they just stop being NULL in the dimension.
        conn.execute(text(f"""
            INSERT INTO dim_weather ({weather_cols_sql}, is_raining, is_freezing)
            SELECT DISTINCT {weather_cols_sql},
                   is_raining::boolean, is_freezing::boolean
            FROM tmp_accidents_staged
            ON CONFLICT ({weather_cols_sql}) DO NOTHING
        """))

        # ---- fact_accidents (upsert, resolving surrogate keys) ----
        road_join = dimension_join("t", "r", ROAD_COLS)
        weather_join = dimension_join("t", "w", WEATHER_COLS)
        conn.execute(text(f"""
            INSERT INTO fact_accidents (
                accident_index, date_key, location_id, road_id, weather_id, severity_id,
                number_of_vehicles, number_of_casualties, police_force,
                did_police_officer_attend, source_year
            )
            SELECT
                t.accident_index, t.date_key, l.location_id, r.road_id, w.weather_id,
                t.severity_id, t.number_of_vehicles, t.number_of_casualties,
                t.police_force, t.did_police_officer_attend_scene_of_accident, t.year
            FROM tmp_accidents_staged t
            JOIN dim_location l ON l.latitude = t.latitude AND l.longitude = t.longitude
            JOIN dim_road r ON {road_join}
            JOIN dim_weather w ON {weather_join}
            ON CONFLICT (accident_index) DO UPDATE SET
                date_key = EXCLUDED.date_key,
                location_id = EXCLUDED.location_id,
                road_id = EXCLUDED.road_id,
                weather_id = EXCLUDED.weather_id,
                severity_id = EXCLUDED.severity_id,
                number_of_vehicles = EXCLUDED.number_of_vehicles,
                number_of_casualties = EXCLUDED.number_of_casualties,
                police_force = EXCLUDED.police_force,
                did_police_officer_attend = EXCLUDED.did_police_officer_attend,
                source_year = EXCLUDED.source_year
        """))

        # Those are INNER JOINs, so a staged accident that fails to resolve one
        # of its dimension rows would vanish without a word. Fail loudly instead.
        staged_count = conn.execute(
            text("SELECT COUNT(*) FROM tmp_accidents_staged")).scalar_one()
        resolved_count = conn.execute(text("""
            SELECT COUNT(*) FROM fact_accidents fa
            WHERE fa.accident_index IN (SELECT accident_index FROM tmp_accidents_staged)
        """)).scalar_one()
        if resolved_count != staged_count:
            raise RuntimeError(
                f"Dimension resolution lost rows: {staged_count} staged accidents "
                f"but {resolved_count} in fact_accidents. Rolling back."
            )
        print(f"  resolved all {resolved_count:,} staged accidents to dimension keys")

        # ---- fact_casualties / fact_vehicles: replace rows for touched accidents ----
        conn.execute(text("""
            DELETE FROM fact_casualties
            WHERE accident_index IN (SELECT DISTINCT accident_index FROM tmp_casualties_staged)
        """))
        conn.execute(text("""
            INSERT INTO fact_casualties (
                accident_index, vehicle_reference, casualty_reference, casualty_class,
                sex_of_casualty, age_of_casualty, age_band_of_casualty, casualty_severity_id,
                pedestrian_location, pedestrian_movement, car_passenger,
                bus_or_coach_passenger, casualty_type, casualty_imd_decile
            )
            SELECT accident_index, vehicle_reference, casualty_reference, casualty_class,
                   sex_of_casualty, age_of_casualty, age_band_of_casualty, casualty_severity_id,
                   pedestrian_location, pedestrian_movement, car_passenger,
                   bus_or_coach_passenger, casualty_type, casualty_imd_decile
            FROM tmp_casualties_staged
        """))

        conn.execute(text("""
            DELETE FROM fact_vehicles
            WHERE accident_index IN (SELECT DISTINCT accident_index FROM tmp_vehicles_staged)
        """))
        conn.execute(text("""
            INSERT INTO fact_vehicles (
                accident_index, vehicle_reference, vehicle_type, towing_and_articulation,
                vehicle_manoeuvre, junction_location, skidding_and_overturning,
                first_point_of_impact, sex_of_driver, age_of_driver, age_band_of_driver,
                engine_capacity_cc, propulsion_code, age_of_vehicle, journey_purpose
            )
            SELECT accident_index, vehicle_reference, vehicle_type, towing_and_articulation,
                   vehicle_manoeuvre, junction_location, skidding_and_overturning,
                   first_point_of_impact, sex_of_driver, age_of_driver, age_band_of_driver,
                   engine_capacity_cc, propulsion_code, age_of_vehicle, journey_purpose
            FROM tmp_vehicles_staged
        """))

        conn.execute(text("DROP TABLE tmp_accidents_staged, tmp_casualties_staged, tmp_vehicles_staged"))

        counts = {}
        for table in ["fact_accidents", "fact_casualties", "fact_vehicles",
                      "dim_date", "dim_location", "dim_road", "dim_weather"]:
            counts[table] = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()

        conn.execute(
            text("""
                INSERT INTO etl_audit_log (run_id, source, stage, row_count, rejected_count,
                                            status, message, started_at)
                VALUES (:run_id, 'load', 'load', :row_count, 0, 'success', :message, :started_at)
            """),
            {
                "run_id": run_id,
                "row_count": counts["fact_accidents"],
                "message": str(counts),
                "started_at": started,
            },
        )

    # A freshly bulk-loaded table has no statistics, so the planner guesses --
    # which makes both the mart refresh below and every later API query pick
    # bad plans. ANALYZE runs outside the load transaction, on committed data.
    print("[load] analyzing tables...")
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        for table in ["fact_accidents", "fact_casualties", "fact_vehicles",
                      "dim_date", "dim_location", "dim_road", "dim_weather"]:
            conn.execute(text(f"ANALYZE {table}"))

    # Refresh each mart in its own autocommit transaction rather than in one
    # call to refresh_marts(): a single transaction holds ACCESS EXCLUSIVE on
    # all ten views until it commits, so every API read blocks for the whole
    # batch. Per-view commits release each lock as soon as that view is done.
    #
    # The memory settings are session-local and matter enormously here.
    # Postgres' defaults (work_mem 4MB, maintenance_work_mem 64MB) leave the
    # first refresh after a bulk load spilling to disk -- the same ten views
    # measured in tens of minutes at the defaults and ~14s with these.
    print("[load] refreshing materialized views...")
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("SET maintenance_work_mem = '512MB'"))
        conn.execute(text("SET work_mem = '128MB'"))
        marts = conn.execute(text(
            r"SELECT matviewname FROM pg_matviews WHERE matviewname LIKE 'mart\_%' "
            r"ORDER BY matviewname"
        )).scalars().all()
        for mart in marts:
            t0 = time.time()
            conn.execute(text(f"REFRESH MATERIALIZED VIEW {mart}"))
            print(f"  {mart:32s} {time.time() - t0:6.1f}s")

    log_event(run_id, "load", "load", status="success", row_count=counts["fact_accidents"],
              message=str(counts), started_at=started)

    print("[load] complete. Final row counts:")
    for table, n in counts.items():
        print(f"  {table:16s} {n:>10,}")
    print("[load] materialized views refreshed.")


if __name__ == "__main__":
    main()
