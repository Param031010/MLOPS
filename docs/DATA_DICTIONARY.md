# Data Dictionary

Every table and column of the traffic-accident warehouse, generated from
the live PostgreSQL schema by `docs/generate_data_dictionary.py`.
Descriptions are the `COMMENT ON` values defined in `sql/schema.sql`.

Generated 2026-10-11 17:06 UTC.

Validation rules applied before a row reaches these tables are listed in
[Validation rules](#validation-rules) at the end.

## Tables

### `dim_severity`

Severity lookup, seeded from STATS19 codes. -1 is STATS19's own "data missing or out of range" value, kept as a labelled category rather than dropped.

*4 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `severity_id` | smallint | no | PK | STATS19 severity code: 1=Fatal, 2=Serious, 3=Slight, -1=Data missing. |
| `severity_label` | text | no |  | Human-readable severity label. |

### `dim_date`

Date + hour grain. One row per distinct hour in which a collision occurred.

*42,170 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `date_key` | bigint | no | PK | Surrogate key, format YYYYMMDDHH (e.g. 2025063017 = 17:00 on 30 June 2025). |
| `full_date` | date | no |  | Calendar date of the collision. |
| `hour` | smallint | no |  | Hour of day, 0-23, from the STATS19 time field. |
| `day_of_week` | smallint | no |  | STATS19 day numbering: 1=Sunday through 7=Saturday. |
| `day_name` | text | no |  | Decoded day name. |
| `day_of_month` | smallint | no |  | Day of month, 1-31. |
| `month` | smallint | no |  | Month number, 1-12. |
| `month_name` | text | no |  | Month name. |
| `quarter` | smallint | no |  | Calendar quarter, 1-4. |
| `year` | smallint | no |  | Calendar year. |
| `is_weekend` | boolean | no |  | True for Saturday and Sunday (STATS19 codes 7 and 1). |

### `dim_location`

One row per distinct collision coordinate, enriched with TomTom street context where quota allowed.

*512,309 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `location_id` | bigint | no | PK | Surrogate key. |
| `latitude` | numeric(9,6) | no | UQ | WGS84 latitude. Validated against Great Britain's bounding box (49.8 to 60.9). |
| `longitude` | numeric(9,6) | no | UQ | WGS84 longitude. Validated against Great Britain's bounding box (-8.6 to 1.8). |
| `location_easting_osgr` | integer | yes |  | OS grid reference easting, as supplied by STATS19. |
| `location_northing_osgr` | integer | yes |  | OS grid reference northing, as supplied by STATS19. |
| `lsoa_of_accident_location` | text | yes |  | Lower Layer Super Output Area code of the collision location. |
| `local_authority_district` | text | yes |  | Local authority district name. |
| `local_authority_ons_district` | text | yes |  | ONS code for the local authority district. |
| `local_authority_highway` | text | yes |  | Highway authority responsible for the road. |
| `police_force` | text | yes |  | Police force area reporting the collision. |
| `urban_or_rural_area` | text | yes |  | Decoded STATS19 classification: Urban, Rural, Unallocated or Data missing. |
| `street_name` | text | yes |  | Street name from TomTom reverse geocoding. NULL where the free-tier quota has not yet reached this coordinate; coverage is reported by /accidents/overview rather than hidden. |
| `road_context_resolved_at` | timestamptz | yes |  | When the street name was last resolved, stamped by the load that first learned it. |

### `dim_road`

Junk dimension over the road-configuration attributes of a collision.

*70,100 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `road_id` | bigint | no | PK | Surrogate key. |
| `first_road_class` | text | yes | UQ | Decoded class of the primary road (Motorway, A(M), A, B, C, Unclassified). |
| `road_type` | text | yes | UQ | Decoded carriageway layout (Single carriageway, Dual carriageway, Roundabout, Slip road, ...). |
| `speed_limit` | smallint | yes | UQ | Posted speed limit in mph at the collision location. |
| `junction_detail` | text | yes | UQ | Decoded junction type, or "Not at junction or within 20 metres". |
| `junction_control` | text | yes | UQ | Decoded control at the junction (auto traffic signal, give way, stop sign, ...). |
| `second_road_class` | text | yes | UQ | Decoded class of the second road where the collision was at a junction. |
| `pedestrian_crossing` | text | yes | UQ | Decoded pedestrian crossing facility within 50 metres. |
| `special_conditions_at_site` | text | yes | UQ | Decoded site defects such as defective signals or an obscured sign. |
| `carriageway_hazards` | text | yes | UQ | Decoded carriageway hazards such as an object in the road or a previous accident. |
| `trunk_road_flag` | text | yes | UQ | Whether the road is a trunk road. |

### `dim_weather`

Junk dimension combining police-observed STATS19 conditions with Open-Meteo measurements for the collision hour.

*362,256 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `weather_id` | bigint | no | PK | Surrogate key. |
| `weather_conditions` | text | yes | UQ | Decoded STATS19 weather as recorded by the attending officer. |
| `road_surface_conditions` | text | yes | UQ | Decoded STATS19 road surface state (Dry, Wet or damp, Snow, Frost or ice, Flood). |
| `light_conditions` | text | yes | UQ | Decoded STATS19 lighting at the time of the collision. |
| `temperature_c` | numeric(5,2) | yes | UQ | Open-Meteo 2m air temperature in Celsius for the collision hour and grid cell. NULL where that grid cell has not been backfilled yet. |
| `precipitation_mm` | numeric(6,2) | yes | UQ | Open-Meteo precipitation in millimetres for the collision hour. |
| `wind_speed_kmh` | numeric(6,2) | yes | UQ | Open-Meteo 10m wind speed in km/h for the collision hour. |
| `weather_code_meteo` | smallint | yes | UQ | Open-Meteo WMO weather interpretation code. |
| `is_raining` | boolean | yes |  | Derived: precipitation greater than zero. |
| `is_freezing` | boolean | yes |  | Derived: temperature at or below zero Celsius. |

### `fact_accidents`

Fact table at collision grain: exactly one row per STATS19 collision that survived validation.

*513,748 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `accident_key` | bigint | no | PK | Surrogate key. |
| `accident_index` | text | no | UQ | Business key, from the source collision_index. Unique, and the conflict target of the load's upsert. |
| `date_key` | bigint | no | FK -> dim_date.date_key | Foreign key to dim_date. |
| `location_id` | bigint | no | FK -> dim_location.location_id | Foreign key to dim_location. |
| `road_id` | bigint | no | FK -> dim_road.road_id | Foreign key to dim_road. |
| `weather_id` | bigint | no | FK -> dim_weather.weather_id | Foreign key to dim_weather. |
| `severity_id` | smallint | no | FK -> dim_severity.severity_id | Foreign key to dim_severity. The classification target for the Part 2 model. |
| `number_of_vehicles` | smallint | yes |  | Vehicles involved in the collision. |
| `number_of_casualties` | smallint | yes |  | People injured in the collision. |
| `police_force` | text | yes |  | Reporting police force. |
| `did_police_officer_attend` | text | yes |  | Decoded: whether an officer attended the scene. |
| `source_year` | smallint | yes |  | Year of the source STATS19 export the row came from. |
| `loaded_at` | timestamptz | no |  | When the row was written by load.py. |

### `fact_casualties`

Fact table at casualty grain: one row per injured person. A separate grain from fact_accidents, because one collision injures several people who each have their own severity.

*652,763 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `casualty_key` | bigint | no | PK | Surrogate key. |
| `accident_index` | text | no | FK -> fact_accidents.accident_index | Foreign key to fact_accidents.accident_index. |
| `vehicle_reference` | smallint | yes |  | Which vehicle in the collision the casualty was associated with. |
| `casualty_reference` | smallint | yes |  | Casualty number within the collision. |
| `casualty_class` | text | yes |  | Decoded: Driver or rider, Passenger, or Pedestrian. |
| `sex_of_casualty` | text | yes |  | Decoded sex of the casualty. |
| `age_of_casualty` | smallint | yes |  | Age in years; -1 where STATS19 recorded it as missing. |
| `age_band_of_casualty` | text | yes |  | Decoded age band. |
| `casualty_severity_id` | smallint | yes | FK -> dim_severity.severity_id | Foreign key to dim_severity: this person's injury severity, which may differ from the collision's overall severity. |
| `pedestrian_location` | text | yes |  | Decoded pedestrian location relative to the carriageway. |
| `pedestrian_movement` | text | yes |  | Decoded pedestrian movement at the time of impact. |
| `car_passenger` | text | yes |  | Decoded: front seat, rear seat, or not a car passenger. |
| `bus_or_coach_passenger` | text | yes |  | Decoded bus or coach passenger status. |
| `casualty_type` | text | yes |  | Decoded road-user type (Car occupant, Pedestrian, Cyclist, Motorcycle rider, ...). |
| `casualty_imd_decile` | smallint | yes |  | Index of Multiple Deprivation decile of the casualty's home area. |

### `fact_vehicles`

Fact table at vehicle grain: one row per vehicle involved in a collision.

*937,164 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `vehicle_key` | bigint | no | PK | Surrogate key. |
| `accident_index` | text | no | FK -> fact_accidents.accident_index | Foreign key to fact_accidents.accident_index. |
| `vehicle_reference` | smallint | yes |  | Vehicle number within the collision. |
| `vehicle_type` | text | yes |  | Decoded vehicle type (Car, Van, Bus or coach, Motorcycle by engine size, ...). |
| `towing_and_articulation` | text | yes |  | Decoded towing or articulation configuration. |
| `vehicle_manoeuvre` | text | yes |  | Decoded manoeuvre the vehicle was performing at impact. |
| `junction_location` | text | yes |  | Decoded position of the vehicle relative to the junction. |
| `skidding_and_overturning` | text | yes |  | Decoded skidding, jack-knifing or overturning. |
| `first_point_of_impact` | text | yes |  | Decoded first point of impact on the vehicle. |
| `sex_of_driver` | text | yes |  | Decoded sex of the driver. |
| `age_of_driver` | smallint | yes |  | Driver age in years; -1 where STATS19 recorded it as missing. |
| `age_band_of_driver` | text | yes |  | Decoded driver age band. |
| `engine_capacity_cc` | integer | yes |  | Engine capacity in cubic centimetres. |
| `propulsion_code` | text | yes |  | Decoded propulsion type (Petrol, Heavy oil, Electric, Hybrid, ...). |
| `age_of_vehicle` | smallint | yes |  | Vehicle age in years at the time of the collision. |
| `journey_purpose` | text | yes |  | Decoded journey purpose of the driver. |

### `etl_audit_log`

One row per source and stage of every pipeline run. Mirrored to etl/data/audit/audit_log.jsonl so the trail survives a database rebuild.

*10 rows at generation time.*

| Column | Type | Null | Key | Description |
|---|---|---|---|---|
| `audit_id` | bigint | no | PK | Surrogate key. |
| `run_id` | text | no |  | Identifier shared by every row belonging to one pipeline run. |
| `source` | text | no |  | What was processed: collisions, casualties, vehicles, weather, road_context or load. |
| `stage` | text | no |  | Pipeline stage: extract, transform or load. |
| `row_count` | integer | yes |  | Rows accepted by the stage. |
| `rejected_count` | integer | yes |  | Rows rejected by the stage; the per-reason breakdown is in message. |
| `status` | text | no |  | success, partial (for example a quota-limited enrichment) or failed. |
| `message` | text | yes |  | Human-readable detail, including the per-reason rejection breakdown. |
| `started_at` | timestamptz | no |  | When the stage started. |
| `finished_at` | timestamptz | no |  | When the stage finished. |

## Analytical layer (materialized views)

Each view backs one dashboard page and is refreshed at the end of every `load.py` run.

| View | Columns | Rows |
|---|---|---|
| `mart_by_hour_weekday` | None | 2,514 |
| `mart_casualties_by_type` | None | 571 |
| `mart_casualty_demographics` | None | 494 |
| `mart_hotspots` | None | 513,748 |
| `mart_monthly_trend` | None | 180 |
| `mart_road_type_severity` | None | 89 |
| `mart_severity_distribution` | None | 3 |
| `mart_top_locations` | None | 512,309 |
| `mart_vehicles_by_type` | None | 320 |
| `mart_weather_impact` | None | 132 |

## Validation rules

Applied by `etl/transform.py`. Every rejection is counted by reason and
written to both `etl/data/audit/audit_log.jsonl` and `etl_audit_log`.

| Rule | Action on failure |
|---|---|
| `collision_index` must be unique | Duplicate rows dropped, first kept |
| `latitude` and `longitude` must be present | Row rejected |
| Coordinates must fall in Great Britain's bounding box (lat 49.8-60.9, lon -8.6-1.8) | Row rejected |
| `date` and `time` must parse (day-first date, `HH:MM` time) | Row rejected |
| Casualty and vehicle rows must reference a surviving collision | Row dropped before load, so the load cannot violate a foreign key |
| STATS19 numeric codes must decode to a known label | Unmapped codes become `Unknown code N` rather than raising, so a new code is visible instead of fatal |
| `-1` ("data missing or out of range") | Preserved as its own labelled category, never silently dropped |

Post-load, `etl/quality_check.py` gates the run: it fails if any table or
mart is empty, if a fact row cannot resolve its dimension, or if the most
recent `etl_audit_log` load row is not `success`.
