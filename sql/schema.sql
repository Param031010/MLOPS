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
    d.year,
    d.hour,
    d.day_of_week,
    d.day_name,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count
FROM fact_accidents fa
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY d.year, d.hour, d.day_of_week, d.day_name, s.severity_id, s.severity_label;

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
    d.year,
    w.weather_conditions,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count,
    -- sums + non-null counts (not AVG) so the API can re-average correctly
    -- over any year range: an average of averages would be wrong.
    SUM(w.temperature_c) AS sum_temperature_c,
    COUNT(w.temperature_c) AS n_temperature,
    SUM(w.precipitation_mm) AS sum_precipitation_mm,
    COUNT(w.precipitation_mm) AS n_precipitation
FROM fact_accidents fa
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_weather w ON w.weather_id = fa.weather_id
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY d.year, w.weather_conditions, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_road_type_severity AS
SELECT
    d.year,
    r.road_type,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS accident_count
FROM fact_accidents fa
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_road r ON r.road_id = fa.road_id
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY d.year, r.road_type, s.severity_id, s.severity_label;

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
    d.year,
    fc.casualty_type,
    fc.casualty_class,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS casualty_count
FROM fact_casualties fc
JOIN fact_accidents fa ON fa.accident_index = fc.accident_index
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_severity s ON s.severity_id = fc.casualty_severity_id
GROUP BY d.year, fc.casualty_type, fc.casualty_class, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_casualty_demographics AS
SELECT
    d.year,
    fc.age_band_of_casualty,
    fc.sex_of_casualty,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS casualty_count
FROM fact_casualties fc
JOIN fact_accidents fa ON fa.accident_index = fc.accident_index
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_severity s ON s.severity_id = fc.casualty_severity_id
GROUP BY d.year, fc.age_band_of_casualty, fc.sex_of_casualty, s.severity_id, s.severity_label;

CREATE MATERIALIZED VIEW IF NOT EXISTS mart_vehicles_by_type AS
SELECT
    d.year,
    fv.vehicle_type,
    s.severity_id,
    s.severity_label,
    COUNT(*) AS vehicle_count
FROM fact_vehicles fv
JOIN fact_accidents fa ON fa.accident_index = fv.accident_index
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_severity s ON s.severity_id = fa.severity_id
GROUP BY d.year, fv.vehicle_type, s.severity_id, s.severity_label;

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

-- ============================================================
-- DATA DICTIONARY
-- ============================================================
-- Column descriptions live in the database itself, so `\d+ <table>` in psql
-- shows them and docs/generate_data_dictionary.py can emit the submitted data
-- dictionary straight from the live schema, instead of a hand-maintained copy
-- that silently drifts out of date.

-- Model-ready table: one row per accident with every field the severity
-- classifier consumes, resolved through the dimensions. A plain view (not a
-- mart) so it is always current and ml/train.py reads exactly one object.
CREATE OR REPLACE VIEW vw_model_features AS
SELECT fa.accident_index, fa.severity_id, fa.number_of_vehicles,
       d.full_date, d.hour, d.day_of_week, d.month, d.is_weekend,
       r.road_type, r.speed_limit, r.junction_detail,
       l.urban_or_rural_area, l.latitude, l.longitude,
       w.light_conditions, w.weather_conditions, w.road_surface_conditions,
       w.temperature_c, w.precipitation_mm
FROM fact_accidents fa
JOIN dim_date d ON d.date_key = fa.date_key
JOIN dim_road r ON r.road_id = fa.road_id
JOIN dim_weather w ON w.weather_id = fa.weather_id
JOIN dim_location l ON l.location_id = fa.location_id;

COMMENT ON VIEW vw_model_features IS 'Model-ready table: one row per collision with the features and label used by ml/train.py.';

COMMENT ON TABLE  dim_severity IS 'Severity lookup, seeded from STATS19 codes. -1 is STATS19''s own "data missing or out of range" value, kept as a labelled category rather than dropped.';
COMMENT ON COLUMN dim_severity.severity_id IS 'STATS19 severity code: 1=Fatal, 2=Serious, 3=Slight, -1=Data missing.';
COMMENT ON COLUMN dim_severity.severity_label IS 'Human-readable severity label.';

COMMENT ON TABLE  dim_date IS 'Date + hour grain. One row per distinct hour in which a collision occurred.';
COMMENT ON COLUMN dim_date.date_key IS 'Surrogate key, format YYYYMMDDHH (e.g. 2025063017 = 17:00 on 30 June 2025).';
COMMENT ON COLUMN dim_date.full_date IS 'Calendar date of the collision.';
COMMENT ON COLUMN dim_date.hour IS 'Hour of day, 0-23, from the STATS19 time field.';
COMMENT ON COLUMN dim_date.day_of_week IS 'STATS19 day numbering: 1=Sunday through 7=Saturday.';
COMMENT ON COLUMN dim_date.day_name IS 'Decoded day name.';
COMMENT ON COLUMN dim_date.day_of_month IS 'Day of month, 1-31.';
COMMENT ON COLUMN dim_date.month IS 'Month number, 1-12.';
COMMENT ON COLUMN dim_date.month_name IS 'Month name.';
COMMENT ON COLUMN dim_date.quarter IS 'Calendar quarter, 1-4.';
COMMENT ON COLUMN dim_date.year IS 'Calendar year.';
COMMENT ON COLUMN dim_date.is_weekend IS 'True for Saturday and Sunday (STATS19 codes 7 and 1).';

COMMENT ON TABLE  dim_location IS 'One row per distinct collision coordinate, enriched with TomTom street context where quota allowed.';
COMMENT ON COLUMN dim_location.location_id IS 'Surrogate key.';
COMMENT ON COLUMN dim_location.latitude IS 'WGS84 latitude. Validated against Great Britain''s bounding box (49.8 to 60.9).';
COMMENT ON COLUMN dim_location.longitude IS 'WGS84 longitude. Validated against Great Britain''s bounding box (-8.6 to 1.8).';
COMMENT ON COLUMN dim_location.location_easting_osgr IS 'OS grid reference easting, as supplied by STATS19.';
COMMENT ON COLUMN dim_location.location_northing_osgr IS 'OS grid reference northing, as supplied by STATS19.';
COMMENT ON COLUMN dim_location.lsoa_of_accident_location IS 'Lower Layer Super Output Area code of the collision location.';
COMMENT ON COLUMN dim_location.local_authority_district IS 'Local authority district name.';
COMMENT ON COLUMN dim_location.local_authority_ons_district IS 'ONS code for the local authority district.';
COMMENT ON COLUMN dim_location.local_authority_highway IS 'Highway authority responsible for the road.';
COMMENT ON COLUMN dim_location.police_force IS 'Police force area reporting the collision.';
COMMENT ON COLUMN dim_location.urban_or_rural_area IS 'Decoded STATS19 classification: Urban, Rural, Unallocated or Data missing.';
COMMENT ON COLUMN dim_location.street_name IS 'Street name from TomTom reverse geocoding. NULL where the free-tier quota has not yet reached this coordinate; coverage is reported by /accidents/overview rather than hidden.';
COMMENT ON COLUMN dim_location.road_context_resolved_at IS 'When the street name was last resolved, stamped by the load that first learned it.';

COMMENT ON TABLE  dim_road IS 'Junk dimension over the road-configuration attributes of a collision.';
COMMENT ON COLUMN dim_road.road_id IS 'Surrogate key.';
COMMENT ON COLUMN dim_road.first_road_class IS 'Decoded class of the primary road (Motorway, A(M), A, B, C, Unclassified).';
COMMENT ON COLUMN dim_road.road_type IS 'Decoded carriageway layout (Single carriageway, Dual carriageway, Roundabout, Slip road, ...).';
COMMENT ON COLUMN dim_road.speed_limit IS 'Posted speed limit in mph at the collision location.';
COMMENT ON COLUMN dim_road.junction_detail IS 'Decoded junction type, or "Not at junction or within 20 metres".';
COMMENT ON COLUMN dim_road.junction_control IS 'Decoded control at the junction (auto traffic signal, give way, stop sign, ...).';
COMMENT ON COLUMN dim_road.second_road_class IS 'Decoded class of the second road where the collision was at a junction.';
COMMENT ON COLUMN dim_road.pedestrian_crossing IS 'Decoded pedestrian crossing facility within 50 metres.';
COMMENT ON COLUMN dim_road.special_conditions_at_site IS 'Decoded site defects such as defective signals or an obscured sign.';
COMMENT ON COLUMN dim_road.carriageway_hazards IS 'Decoded carriageway hazards such as an object in the road or a previous accident.';
COMMENT ON COLUMN dim_road.trunk_road_flag IS 'Whether the road is a trunk road.';

COMMENT ON TABLE  dim_weather IS 'Junk dimension combining police-observed STATS19 conditions with Open-Meteo measurements for the collision hour.';
COMMENT ON COLUMN dim_weather.weather_id IS 'Surrogate key.';
COMMENT ON COLUMN dim_weather.weather_conditions IS 'Decoded STATS19 weather as recorded by the attending officer.';
COMMENT ON COLUMN dim_weather.road_surface_conditions IS 'Decoded STATS19 road surface state (Dry, Wet or damp, Snow, Frost or ice, Flood).';
COMMENT ON COLUMN dim_weather.light_conditions IS 'Decoded STATS19 lighting at the time of the collision.';
COMMENT ON COLUMN dim_weather.temperature_c IS 'Open-Meteo 2m air temperature in Celsius for the collision hour and grid cell. NULL where that grid cell has not been backfilled yet.';
COMMENT ON COLUMN dim_weather.precipitation_mm IS 'Open-Meteo precipitation in millimetres for the collision hour.';
COMMENT ON COLUMN dim_weather.wind_speed_kmh IS 'Open-Meteo 10m wind speed in km/h for the collision hour.';
COMMENT ON COLUMN dim_weather.weather_code_meteo IS 'Open-Meteo WMO weather interpretation code.';
COMMENT ON COLUMN dim_weather.is_raining IS 'Derived: precipitation greater than zero.';
COMMENT ON COLUMN dim_weather.is_freezing IS 'Derived: temperature at or below zero Celsius.';

COMMENT ON TABLE  fact_accidents IS 'Fact table at collision grain: exactly one row per STATS19 collision that survived validation.';
COMMENT ON COLUMN fact_accidents.accident_key IS 'Surrogate key.';
COMMENT ON COLUMN fact_accidents.accident_index IS 'Business key, from the source collision_index. Unique, and the conflict target of the load''s upsert.';
COMMENT ON COLUMN fact_accidents.date_key IS 'Foreign key to dim_date.';
COMMENT ON COLUMN fact_accidents.location_id IS 'Foreign key to dim_location.';
COMMENT ON COLUMN fact_accidents.road_id IS 'Foreign key to dim_road.';
COMMENT ON COLUMN fact_accidents.weather_id IS 'Foreign key to dim_weather.';
COMMENT ON COLUMN fact_accidents.severity_id IS 'Foreign key to dim_severity. The classification target for the Part 2 model.';
COMMENT ON COLUMN fact_accidents.number_of_vehicles IS 'Vehicles involved in the collision.';
COMMENT ON COLUMN fact_accidents.number_of_casualties IS 'People injured in the collision.';
COMMENT ON COLUMN fact_accidents.police_force IS 'Reporting police force.';
COMMENT ON COLUMN fact_accidents.did_police_officer_attend IS 'Decoded: whether an officer attended the scene.';
COMMENT ON COLUMN fact_accidents.source_year IS 'Year of the source STATS19 export the row came from.';
COMMENT ON COLUMN fact_accidents.loaded_at IS 'When the row was written by load.py.';

COMMENT ON TABLE  fact_casualties IS 'Fact table at casualty grain: one row per injured person. A separate grain from fact_accidents, because one collision injures several people who each have their own severity.';
COMMENT ON COLUMN fact_casualties.casualty_key IS 'Surrogate key.';
COMMENT ON COLUMN fact_casualties.accident_index IS 'Foreign key to fact_accidents.accident_index.';
COMMENT ON COLUMN fact_casualties.vehicle_reference IS 'Which vehicle in the collision the casualty was associated with.';
COMMENT ON COLUMN fact_casualties.casualty_reference IS 'Casualty number within the collision.';
COMMENT ON COLUMN fact_casualties.casualty_class IS 'Decoded: Driver or rider, Passenger, or Pedestrian.';
COMMENT ON COLUMN fact_casualties.sex_of_casualty IS 'Decoded sex of the casualty.';
COMMENT ON COLUMN fact_casualties.age_of_casualty IS 'Age in years; -1 where STATS19 recorded it as missing.';
COMMENT ON COLUMN fact_casualties.age_band_of_casualty IS 'Decoded age band.';
COMMENT ON COLUMN fact_casualties.casualty_severity_id IS 'Foreign key to dim_severity: this person''s injury severity, which may differ from the collision''s overall severity.';
COMMENT ON COLUMN fact_casualties.pedestrian_location IS 'Decoded pedestrian location relative to the carriageway.';
COMMENT ON COLUMN fact_casualties.pedestrian_movement IS 'Decoded pedestrian movement at the time of impact.';
COMMENT ON COLUMN fact_casualties.car_passenger IS 'Decoded: front seat, rear seat, or not a car passenger.';
COMMENT ON COLUMN fact_casualties.bus_or_coach_passenger IS 'Decoded bus or coach passenger status.';
COMMENT ON COLUMN fact_casualties.casualty_type IS 'Decoded road-user type (Car occupant, Pedestrian, Cyclist, Motorcycle rider, ...).';
COMMENT ON COLUMN fact_casualties.casualty_imd_decile IS 'Index of Multiple Deprivation decile of the casualty''s home area.';

COMMENT ON TABLE  fact_vehicles IS 'Fact table at vehicle grain: one row per vehicle involved in a collision.';
COMMENT ON COLUMN fact_vehicles.vehicle_key IS 'Surrogate key.';
COMMENT ON COLUMN fact_vehicles.accident_index IS 'Foreign key to fact_accidents.accident_index.';
COMMENT ON COLUMN fact_vehicles.vehicle_reference IS 'Vehicle number within the collision.';
COMMENT ON COLUMN fact_vehicles.vehicle_type IS 'Decoded vehicle type (Car, Van, Bus or coach, Motorcycle by engine size, ...).';
COMMENT ON COLUMN fact_vehicles.towing_and_articulation IS 'Decoded towing or articulation configuration.';
COMMENT ON COLUMN fact_vehicles.vehicle_manoeuvre IS 'Decoded manoeuvre the vehicle was performing at impact.';
COMMENT ON COLUMN fact_vehicles.junction_location IS 'Decoded position of the vehicle relative to the junction.';
COMMENT ON COLUMN fact_vehicles.skidding_and_overturning IS 'Decoded skidding, jack-knifing or overturning.';
COMMENT ON COLUMN fact_vehicles.first_point_of_impact IS 'Decoded first point of impact on the vehicle.';
COMMENT ON COLUMN fact_vehicles.sex_of_driver IS 'Decoded sex of the driver.';
COMMENT ON COLUMN fact_vehicles.age_of_driver IS 'Driver age in years; -1 where STATS19 recorded it as missing.';
COMMENT ON COLUMN fact_vehicles.age_band_of_driver IS 'Decoded driver age band.';
COMMENT ON COLUMN fact_vehicles.engine_capacity_cc IS 'Engine capacity in cubic centimetres.';
COMMENT ON COLUMN fact_vehicles.propulsion_code IS 'Decoded propulsion type (Petrol, Heavy oil, Electric, Hybrid, ...).';
COMMENT ON COLUMN fact_vehicles.age_of_vehicle IS 'Vehicle age in years at the time of the collision.';
COMMENT ON COLUMN fact_vehicles.journey_purpose IS 'Decoded journey purpose of the driver.';

COMMENT ON TABLE  etl_audit_log IS 'One row per source and stage of every pipeline run. Mirrored to etl/data/audit/audit_log.jsonl so the trail survives a database rebuild.';
COMMENT ON COLUMN etl_audit_log.audit_id IS 'Surrogate key.';
COMMENT ON COLUMN etl_audit_log.run_id IS 'Identifier shared by every row belonging to one pipeline run.';
COMMENT ON COLUMN etl_audit_log.source IS 'What was processed: collisions, casualties, vehicles, weather, road_context or load.';
COMMENT ON COLUMN etl_audit_log.stage IS 'Pipeline stage: extract, transform or load.';
COMMENT ON COLUMN etl_audit_log.row_count IS 'Rows accepted by the stage.';
COMMENT ON COLUMN etl_audit_log.rejected_count IS 'Rows rejected by the stage; the per-reason breakdown is in message.';
COMMENT ON COLUMN etl_audit_log.status IS 'success, partial (for example a quota-limited enrichment) or failed.';
COMMENT ON COLUMN etl_audit_log.message IS 'Human-readable detail, including the per-reason rejection breakdown.';
COMMENT ON COLUMN etl_audit_log.started_at IS 'When the stage started.';
COMMENT ON COLUMN etl_audit_log.finished_at IS 'When the stage finished.';
