-- Traffic Accident Severity Analytics warehouse
-- Star schema: dim_date, dim_location, dim_road, dim_weather, dim_severity
--              fact_accidents, fact_casualties, fact_vehicles
--              etl_audit_log + mart_* materialized views
--
-- Source CSVs (dataset/raw/) use the current STATS19 export naming
-- (collision_index / collision_severity, not the older accident_index /
-- accident_severity). "accident_index" below is our warehouse's business
-- key, populated from the source's collision_index during transform.

-- PostGIS is used for dim_location's generated geography column and its
-- spatial index. It is an optional add-on that does not ship with
-- PostgreSQL, so this script degrades gracefully: without it every table,
-- view and query below still works (nothing in the API calls a PostGIS
-- function), and installing PostGIS later + re-running this script adds
-- the geography column and index in place.
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS postgis;
    RAISE NOTICE 'PostGIS enabled.';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'PostGIS not available (%) - dim_location.geog will be skipped.', SQLERRM;
END$$;

-- ============================================================
-- DIMENSIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS dim_severity (
    severity_id     SMALLINT PRIMARY KEY,
    severity_label  TEXT NOT NULL
);

INSERT INTO dim_severity (severity_id, severity_label) VALUES
    (1, 'Fatal'),
    (2, 'Serious'),
    (3, 'Slight'),
    (-1, 'Data missing')
ON CONFLICT (severity_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS dim_date (
    date_key        BIGINT PRIMARY KEY,        -- YYYYMMDDHH
    full_date       DATE NOT NULL,
    hour            SMALLINT NOT NULL,
    day_of_week     SMALLINT NOT NULL,          -- STATS19 numbering, 1=Sunday
    day_name        TEXT NOT NULL,
    day_of_month    SMALLINT NOT NULL,
    month           SMALLINT NOT NULL,
    month_name      TEXT NOT NULL,
    quarter         SMALLINT NOT NULL,
    year            SMALLINT NOT NULL,
    is_weekend      BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dim_date_full_date ON dim_date (full_date);
CREATE INDEX IF NOT EXISTS idx_dim_date_year_month ON dim_date (year, month);

CREATE TABLE IF NOT EXISTS dim_location (
    location_id                    BIGSERIAL PRIMARY KEY,
    latitude                       NUMERIC(9,6) NOT NULL,
    longitude                      NUMERIC(9,6) NOT NULL,
    location_easting_osgr          INTEGER,
    location_northing_osgr         INTEGER,
    lsoa_of_accident_location      TEXT,
    local_authority_district       TEXT,
    local_authority_ons_district   TEXT,
    local_authority_highway        TEXT,
    police_force                   TEXT,
    urban_or_rural_area            TEXT,
    street_name                    TEXT,          -- from TomTom reverse geocoding, nullable
    road_context_resolved_at       TIMESTAMPTZ,
    UNIQUE (latitude, longitude)
);

-- Add the PostGIS geography column + spatial index only where the
-- extension is installed. Idempotent: safe to re-run after installing it.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis')
       AND NOT EXISTS (
           SELECT 1 FROM information_schema.columns
           WHERE table_name = 'dim_location' AND column_name = 'geog'
       )
    THEN
        EXECUTE 'ALTER TABLE dim_location ADD COLUMN geog GEOGRAPHY(Point, 4326)
                 GENERATED ALWAYS AS (
                     ST_SetSRID(ST_MakePoint(longitude::float8, latitude::float8), 4326)::geography
                 ) STORED';
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_dim_location_geog ON dim_location USING GIST (geog)';
        RAISE NOTICE 'Added dim_location.geog (PostGIS geography) + GiST index.';
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS dim_road (
    road_id                     BIGSERIAL PRIMARY KEY,
    first_road_class            TEXT,
    road_type                   TEXT,
    speed_limit                 SMALLINT,
    junction_detail             TEXT,
    junction_control            TEXT,
    second_road_class           TEXT,
    pedestrian_crossing         TEXT,
    special_conditions_at_site  TEXT,
    carriageway_hazards         TEXT,
    trunk_road_flag             TEXT,
    UNIQUE (
        first_road_class, road_type, speed_limit, junction_detail,
        junction_control, second_road_class, pedestrian_crossing,
        special_conditions_at_site, carriageway_hazards, trunk_road_flag
    )
);

CREATE TABLE IF NOT EXISTS dim_weather (
    weather_id                  BIGSERIAL PRIMARY KEY,
    weather_conditions          TEXT,           -- STATS19-recorded, police-observed
    road_surface_conditions     TEXT,
    light_conditions            TEXT,
    temperature_c               NUMERIC(5,2),   -- Open-Meteo enrichment
    precipitation_mm            NUMERIC(6,2),
    wind_speed_kmh              NUMERIC(6,2),
    weather_code_meteo          SMALLINT,
    is_raining                  BOOLEAN,
    is_freezing                 BOOLEAN,
    UNIQUE (
        weather_conditions, road_surface_conditions, light_conditions,
        temperature_c, precipitation_mm, wind_speed_kmh, weather_code_meteo
    )
);

-- Dimension dedupe keys must treat NULLs as equal.
--
-- Weather enrichment is quota-limited, so a large share of accidents carry
-- NULL temperature/precipitation/wind/weather-code. Under the SQL default
-- (NULLS DISTINCT) two such rows never conflict, so load.py's
-- "INSERT ... ON CONFLICT DO NOTHING" re-inserts a duplicate dimension row on
-- every rerun. The fact upsert then matches both copies through its
-- COALESCE join, emits the same accident_index twice in one statement, and
-- dies with "ON CONFLICT DO UPDATE command cannot affect row a second time",
-- rolling back the whole load. NULLS NOT DISTINCT (PostgreSQL 15+) makes the
-- rerun the intended no-op and keeps the pipeline idempotent.
--
-- Idempotent: only rebuilds a key that is still NULLS DISTINCT.
DO $$
DECLARE
    con_name text;
BEGIN
    SELECT c.conname INTO con_name
    FROM pg_constraint c
    JOIN pg_index i ON i.indexrelid = c.conindid
    WHERE c.conrelid = 'dim_weather'::regclass AND c.contype = 'u'
      AND NOT i.indnullsnotdistinct;
    IF con_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE dim_weather DROP CONSTRAINT %I', con_name);
        ALTER TABLE dim_weather ADD CONSTRAINT dim_weather_dedupe_key
            UNIQUE NULLS NOT DISTINCT (
                weather_conditions, road_surface_conditions, light_conditions,
                temperature_c, precipitation_mm, wind_speed_kmh, weather_code_meteo
            );
        RAISE NOTICE 'dim_weather dedupe key rebuilt as NULLS NOT DISTINCT.';
    END IF;

    SELECT c.conname INTO con_name
    FROM pg_constraint c
    JOIN pg_index i ON i.indexrelid = c.conindid
    WHERE c.conrelid = 'dim_road'::regclass AND c.contype = 'u'
      AND NOT i.indnullsnotdistinct;
    IF con_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE dim_road DROP CONSTRAINT %I', con_name);
        ALTER TABLE dim_road ADD CONSTRAINT dim_road_dedupe_key
            UNIQUE NULLS NOT DISTINCT (
                first_road_class, road_type, speed_limit, junction_detail,
                junction_control, second_road_class, pedestrian_crossing,
                special_conditions_at_site, carriageway_hazards, trunk_road_flag
            );
        RAISE NOTICE 'dim_road dedupe key rebuilt as NULLS NOT DISTINCT.';
    END IF;
END$$;

-- ============================================================
-- FACTS
-- ============================================================

CREATE TABLE IF NOT EXISTS fact_accidents (
    accident_key            BIGSERIAL PRIMARY KEY,
    accident_index          TEXT NOT NULL UNIQUE,      -- source collision_index
    date_key                BIGINT NOT NULL REFERENCES dim_date (date_key),
    location_id             BIGINT NOT NULL REFERENCES dim_location (location_id),
    road_id                 BIGINT NOT NULL REFERENCES dim_road (road_id),
    weather_id              BIGINT NOT NULL REFERENCES dim_weather (weather_id),
    severity_id             SMALLINT NOT NULL REFERENCES dim_severity (severity_id),
    number_of_vehicles       SMALLINT,
    number_of_casualties     SMALLINT,
    police_force             TEXT,
    did_police_officer_attend TEXT,
    source_year              SMALLINT,
    loaded_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fact_accidents_date ON fact_accidents (date_key);
CREATE INDEX IF NOT EXISTS idx_fact_accidents_severity ON fact_accidents (severity_id);
CREATE INDEX IF NOT EXISTS idx_fact_accidents_location ON fact_accidents (location_id);
CREATE INDEX IF NOT EXISTS idx_fact_accidents_road ON fact_accidents (road_id);

CREATE TABLE IF NOT EXISTS fact_casualties (
    casualty_key            BIGSERIAL PRIMARY KEY,
    accident_index          TEXT NOT NULL REFERENCES fact_accidents (accident_index),
    vehicle_reference        SMALLINT,
    casualty_reference       SMALLINT,
    casualty_class           TEXT,
    sex_of_casualty          TEXT,
    age_of_casualty          SMALLINT,
    age_band_of_casualty     TEXT,
    casualty_severity_id     SMALLINT REFERENCES dim_severity (severity_id),
    pedestrian_location      TEXT,
    pedestrian_movement      TEXT,
    car_passenger            TEXT,
    bus_or_coach_passenger   TEXT,
    casualty_type            TEXT,
    casualty_imd_decile      SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_fact_casualties_accident ON fact_casualties (accident_index);
CREATE INDEX IF NOT EXISTS idx_fact_casualties_severity ON fact_casualties (casualty_severity_id);
CREATE INDEX IF NOT EXISTS idx_fact_casualties_type ON fact_casualties (casualty_type);

CREATE TABLE IF NOT EXISTS fact_vehicles (
    vehicle_key              BIGSERIAL PRIMARY KEY,
    accident_index           TEXT NOT NULL REFERENCES fact_accidents (accident_index),
    vehicle_reference        SMALLINT,
    vehicle_type             TEXT,
    towing_and_articulation  TEXT,
    vehicle_manoeuvre        TEXT,
    junction_location        TEXT,
    skidding_and_overturning TEXT,
    first_point_of_impact    TEXT,
    sex_of_driver            TEXT,
    age_of_driver            SMALLINT,
    age_band_of_driver       TEXT,
    engine_capacity_cc       INTEGER,
    propulsion_code          TEXT,
    age_of_vehicle           SMALLINT,
    journey_purpose          TEXT
);

CREATE INDEX IF NOT EXISTS idx_fact_vehicles_accident ON fact_vehicles (accident_index);
CREATE INDEX IF NOT EXISTS idx_fact_vehicles_type ON fact_vehicles (vehicle_type);

-- ============================================================
-- AUDIT LOG
-- ============================================================

CREATE TABLE IF NOT EXISTS etl_audit_log (
    audit_id        BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    source          TEXT NOT NULL,             -- collisions | casualties | vehicles | weather | road_context | load
    stage           TEXT NOT NULL,             -- extract | transform | load
    row_count       INTEGER,
    rejected_count  INTEGER,
    status          TEXT NOT NULL,             -- success | failed | partial
    message         TEXT,
    started_at      TIMESTAMPTZ NOT NULL,
    finished_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_etl_audit_log_run ON etl_audit_log (run_id);

-- ============================================================
-- MART: materialized views, refreshed at the end of every load
-- ============================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_severity_distribution AS
SELECT
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count,
    SUM(fa.number_of_casualties) AS total_casualties
FROM fact_accidents fa
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_by_hour_weekday AS
SELECT
    d.hour,
    d.day_of_week,
    d.day_name,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count
FROM fact_accidents fa
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY d.hour, d.day_of_week, d.day_name, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_monthly_trend AS
SELECT
    d.year,
    d.month,
    d.month_name,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count,
    SUM(fa.number_of_casualties) AS total_casualties,
    SUM(fa.number_of_vehicles) AS total_vehicles
FROM fact_accidents fa
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY d.year, d.month, d.month_name, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_weather_impact AS
SELECT
    w.weather_conditions,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count,
    AVG(w.temperature_c) AS avg_temperature_c,
    AVG(w.precipitation_mm) AS avg_precipitation_mm
FROM fact_accidents fa
JOIN dim_weather w ON w.weather_id = fa.weather_id
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY w.weather_conditions, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_road_type_severity AS
SELECT
    r.road_type,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count
FROM fact_accidents fa
JOIN dim_road r ON r.road_id = fa.road_id
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY r.road_type, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_hotspots AS
SELECT
    fa.accident_index,
    l.latitude,
    l.longitude,
    l.street_name,
    d.full_date,
    d.year,
    s.severity_id,
    s.severity_label,
    fa.number_of_casualties
FROM fact_accidents fa
JOIN dim_location l ON l.location_id = fa.location_id
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_severity s ON s.severity_id = fa.severity_id;

CREATE INDEX IF NOT EXISTS idx_mart_hotspots_severity ON mart_hotspots (severity_id);
CREATE INDEX IF NOT EXISTS idx_mart_hotspots_date ON mart_hotspots (full_date);

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_top_locations AS
SELECT
    l.latitude,
    l.longitude,
    l.street_name,
    l.local_authority_district,
    COUNT(*) AS accident_count,
    SUM(fa.number_of_casualties) AS total_casualties,
    SUM(CASE WHEN fa.severity_id = 1 THEN 1 ELSE 0 END) AS fatal_count
FROM fact_accidents fa
JOIN dim_location l ON l.location_id = fa.location_id
GROUP BY l.latitude, l.longitude, l.street_name, l.local_authority_district
ORDER BY accident_count DESC;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_casualties_by_type AS
SELECT
    fc.casualty_type,
    fc.casualty_class,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS casualty_count
FROM fact_casualties fc
JOIN dim_severity s ON s.severity_id = fc.casualty_severity_id
GROUP BY fc.casualty_type, fc.casualty_class, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_casualty_demographics AS
SELECT
    fc.age_band_of_casualty,
    fc.sex_of_casualty,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS casualty_count
FROM fact_casualties fc
JOIN dim_severity s ON s.severity_id = fc.casualty_severity_id
GROUP BY fc.age_band_of_casualty, fc.sex_of_casualty, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_vehicles_by_type AS
SELECT
    fv.vehicle_type,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS vehicle_count
FROM fact_vehicles fv
JOIN fact_accidents fa ON fa.accident_index = fv.accident_index
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY fv.vehicle_type, s.severity_id, s.severity_label;

-- Convenience function: refresh every mart in one call (used by etl/load.py
-- and available for a manual `SELECT refresh_marts();`)
CREATE OR REPLACE FUNCTION refresh_marts() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mart_severity_distribution;
    REFRESH MATERIALIZED VIEW mart_by_hour_weekday;
    REFRESH MATERIALIZED VIEW mart_monthly_trend;
    REFRESH MATERIALIZED VIEW mart_weather_impact;
    REFRESH MATERIALIZED VIEW mart_road_type_severity;
    REFRESH MATERIALIZED VIEW mart_hotspots;
    REFRESH MATERIALIZED VIEW mart_top_locations;
    REFRESH MATERIALIZED VIEW mart_casualties_by_type;
    REFRESH MATERIALIZED VIEW mart_casualty_demographics;
    REFRESH MATERIALIZED VIEW mart_vehicles_by_type;
END;
$$ LANGUAGE plpgsql;
