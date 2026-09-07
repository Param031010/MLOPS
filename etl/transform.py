"""Transform stage: clean, decode, and join the raw snapshots + enrichment
parquet files into staging parquet files ready for load.py.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audit import log_event  # noqa: E402
import stats19_lookups as lk  # noqa: E402

RAW_DIR = Path(__file__).parent / "data" / "raw"
STAGING_DIR = Path(__file__).parent / "data" / "staging"

GRID_STEP = 0.5
GB_LAT_RANGE = (49.8, 60.9)
GB_LON_RANGE = (-8.6, 1.8)


def latest_snapshot(name: str) -> Path:
    candidates = sorted(RAW_DIR.glob(f"{name}_*.csv"))
    if not candidates:
        raise SystemExit(
            f"No snapshot found for '{name}' in {RAW_DIR}. Run extract.py first."
        )
    return candidates[-1]


def snap_to_grid(value: float) -> float:
    return round(value / GRID_STEP) * GRID_STEP


def clean_collisions(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    started = datetime.now(timezone.utc)
    n0 = len(df)
    rejections: dict[str, int] = {}

    before = len(df)
    df = df.drop_duplicates(subset=["collision_index"], keep="first")
    rejections["duplicate_collision_index"] = before - len(df)

    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    rejections["missing_coordinates"] = before - len(df)

    before = len(df)
    df = df[
        df["latitude"].between(*GB_LAT_RANGE) & df["longitude"].between(*GB_LON_RANGE)
    ]
    rejections["coordinates_outside_gb_bbox"] = before - len(df)

    before = len(df)
    parsed_date = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    parsed_time = pd.to_datetime(df["time"], format="%H:%M", errors="coerce")
    df = df.assign(_parsed_date=parsed_date, _parsed_time=parsed_time)
    df = df.dropna(subset=["_parsed_date", "_parsed_time"])
    rejections["unparseable_date_or_time"] = before - len(df)

    total_rejected = sum(rejections.values())
    log_event(
        run_id, "collisions", "transform", status="success",
        row_count=len(df), rejected_count=total_rejected,
        message=f"kept {len(df)}/{n0}; rejections={rejections}",
        started_at=started,
    )
    print(f"[transform] collisions: kept {len(df)}/{n0} ({rejections})")
    return df


def decode_collisions(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["accident_index"] = df["collision_index"]

    full_date = df["_parsed_date"]
    hour = df["_parsed_time"].dt.hour
    d["full_date"] = full_date.dt.date
    d["hour"] = hour
    d["date_key"] = (
        full_date.dt.strftime("%Y%m%d").astype("int64") * 100 + hour.astype("int64")
    )
    d["day_of_week"] = df["day_of_week"]
    d["day_name"] = lk.decode_series(df["day_of_week"], lk.DAY_OF_WEEK)
    d["day_of_month"] = full_date.dt.day
    d["month"] = full_date.dt.month
    d["month_name"] = full_date.dt.strftime("%B")
    d["quarter"] = full_date.dt.quarter
    d["year"] = full_date.dt.year
    d["is_weekend"] = df["day_of_week"].isin([1, 7])

    d["latitude"] = df["latitude"]
    d["longitude"] = df["longitude"]
    d["location_easting_osgr"] = df.get("location_easting_osgr")
    d["location_northing_osgr"] = df.get("location_northing_osgr")
    d["lsoa_of_accident_location"] = df.get("lsoa_of_accident_location")
    d["local_authority_district"] = df.get("local_authority_district")
    d["local_authority_ons_district"] = df.get("local_authority_ons_district")
    d["local_authority_highway"] = df.get("local_authority_highway_current",
                                           df.get("local_authority_highway"))
    d["police_force"] = df.get("police_force")
    d["urban_or_rural_area"] = lk.decode_series(df["urban_or_rural_area"], lk.URBAN_OR_RURAL_AREA)

    d["first_road_class"] = lk.decode_series(df["first_road_class"], lk.ROAD_CLASS)
    d["road_type"] = lk.decode_series(df["road_type"], lk.ROAD_TYPE)
    d["speed_limit"] = df["speed_limit"]
    d["junction_detail"] = lk.decode_series(df["junction_detail"], lk.JUNCTION_DETAIL)
    d["junction_control"] = lk.decode_series(df["junction_control"], lk.JUNCTION_CONTROL)
    d["second_road_class"] = lk.decode_series(df["second_road_class"], lk.ROAD_CLASS)
    d["pedestrian_crossing"] = lk.decode_series(df["pedestrian_crossing"], lk.PEDESTRIAN_CROSSING)
    d["special_conditions_at_site"] = lk.decode_series(
        df["special_conditions_at_site"], lk.SPECIAL_CONDITIONS_AT_SITE
    )
    d["carriageway_hazards"] = lk.decode_series(df["carriageway_hazards"], lk.CARRIAGEWAY_HAZARDS)
    d["trunk_road_flag"] = lk.decode_series(df.get("trunk_road_flag", pd.Series(index=df.index)),
                                             lk.TRUNK_ROAD_FLAG)

    d["weather_conditions"] = lk.decode_series(df["weather_conditions"], lk.WEATHER_CONDITIONS)
    d["road_surface_conditions"] = lk.decode_series(df["road_surface_conditions"], lk.ROAD_SURFACE_CONDITIONS)
    d["light_conditions"] = lk.decode_series(df["light_conditions"], lk.LIGHT_CONDITIONS)

    d["severity_id"] = pd.to_numeric(df["collision_severity"], errors="coerce").fillna(-1).astype(int)
    d["number_of_vehicles"] = df["number_of_vehicles"]
    d["number_of_casualties"] = df["number_of_casualties"]
    d["did_police_officer_attend_scene_of_accident"] = lk.decode_series(
        df["did_police_officer_attend_scene_of_accident"], lk.DID_POLICE_OFFICER_ATTEND
    )

    d["grid_lat"] = df["latitude"].apply(snap_to_grid)
    d["grid_lon"] = df["longitude"].apply(snap_to_grid)

    return d


def join_weather(staged: pd.DataFrame, run_id: str) -> pd.DataFrame:
    started = datetime.now(timezone.utc)
    weather_path = RAW_DIR / "weather_grid.parquet"
    if not weather_path.exists():
        print("[transform] no weather_grid.parquet found - weather fields will be null")
        for col in ["temperature_c", "precipitation_mm", "wind_speed_kmh", "weather_code_meteo"]:
            staged[col] = None
        log_event(run_id, "weather", "transform", status="success", row_count=0,
                  message="no weather parquet available", started_at=started)
        return staged

    weather = pd.read_parquet(weather_path)
    weather = weather[["grid_lat", "grid_lon", "date", "hour",
                        "temperature_c", "precipitation_mm", "wind_speed_kmh", "weather_code_meteo"]]

    staged = staged.merge(
        weather,
        left_on=["grid_lat", "grid_lon", "full_date", "hour"],
        right_on=["grid_lat", "grid_lon", "date", "hour"],
        how="left",
    ).drop(columns=["date"])

    matched = staged["temperature_c"].notna().sum()
    log_event(run_id, "weather", "transform", status="success",
              row_count=matched, rejected_count=len(staged) - matched,
              message=f"matched weather for {matched}/{len(staged)} accidents",
              started_at=started)
    print(f"[transform] weather matched for {matched}/{len(staged)} accidents")
    return staged


def join_road_context(staged: pd.DataFrame, run_id: str) -> pd.DataFrame:
    started = datetime.now(timezone.utc)
    road_path = RAW_DIR / "road_context.parquet"
    if not road_path.exists():
        staged["street_name"] = None
        log_event(run_id, "road_context", "transform", status="success", row_count=0,
                  message="no road_context parquet available", started_at=started)
        return staged

    road = pd.read_parquet(road_path)[["latitude", "longitude", "street_name"]]
    staged = staged.merge(road, on=["latitude", "longitude"], how="left")
    matched = staged["street_name"].notna().sum()
    log_event(run_id, "road_context", "transform", status="success",
              row_count=matched, rejected_count=len(staged) - matched,
              message=f"matched road context for {matched}/{len(staged)} accidents "
                      f"({matched / len(staged):.1%} coverage)",
              started_at=started)
    print(f"[transform] road context matched for {matched}/{len(staged)} accidents "
          f"({matched / len(staged):.1%} coverage)")
    return staged


def finalize_weather_flags(staged: pd.DataFrame) -> pd.DataFrame:
    staged["is_raining"] = staged["precipitation_mm"].fillna(0) > 0
    staged["is_freezing"] = staged["temperature_c"] <= 0
    return staged


def transform_casualties(raw: pd.DataFrame, valid_accident_index: set, run_id: str) -> pd.DataFrame:
    started = datetime.now(timezone.utc)
    n0 = len(raw)
    df = raw[raw["collision_index"].isin(valid_accident_index)].copy()

    out = pd.DataFrame(index=df.index)
    out["accident_index"] = df["collision_index"]
    out["vehicle_reference"] = df.get("vehicle_reference")
    out["casualty_reference"] = df.get("casualty_reference")
    out["casualty_class"] = lk.decode_series(df["casualty_class"], lk.CASUALTY_CLASS)
    out["sex_of_casualty"] = lk.decode_series(df["sex_of_casualty"], lk.SEX)
    out["age_of_casualty"] = df.get("age_of_casualty")
    out["age_band_of_casualty"] = lk.decode_series(df["age_band_of_casualty"], lk.AGE_BAND)
    out["casualty_severity_id"] = pd.to_numeric(df["casualty_severity"], errors="coerce").fillna(-1).astype(int)
    out["pedestrian_location"] = lk.decode_series(df.get("pedestrian_location", pd.Series(index=df.index)), lk.PEDESTRIAN_LOCATION)
    out["pedestrian_movement"] = lk.decode_series(df.get("pedestrian_movement", pd.Series(index=df.index)), lk.PEDESTRIAN_MOVEMENT)
    out["car_passenger"] = lk.decode_series(df.get("car_passenger", pd.Series(index=df.index)), lk.CAR_PASSENGER)
    out["bus_or_coach_passenger"] = lk.decode_series(df.get("bus_or_coach_passenger", pd.Series(index=df.index)), lk.BUS_OR_COACH_PASSENGER)
    out["casualty_type"] = lk.decode_series(df["casualty_type"], lk.CASUALTY_TYPE)
    out["casualty_imd_decile"] = df.get("casualty_imd_decile")

    log_event(run_id, "casualties", "transform", status="success",
              row_count=len(out), rejected_count=n0 - len(df),
              message=f"kept {len(out)}/{n0} (dropped rows whose accident was cleaned out)",
              started_at=started)
    print(f"[transform] casualties: kept {len(out)}/{n0}")
    return out


def transform_vehicles(raw: pd.DataFrame, valid_accident_index: set, run_id: str) -> pd.DataFrame:
    started = datetime.now(timezone.utc)
    n0 = len(raw)
    df = raw[raw["collision_index"].isin(valid_accident_index)].copy()

    out = pd.DataFrame(index=df.index)
    out["accident_index"] = df["collision_index"]
    out["vehicle_reference"] = df.get("vehicle_reference")
    out["vehicle_type"] = lk.decode_series(df["vehicle_type"], lk.VEHICLE_TYPE)
    out["towing_and_articulation"] = lk.decode_series(df.get("towing_and_articulation", pd.Series(index=df.index)), lk.TOWING_AND_ARTICULATION)
    out["vehicle_manoeuvre"] = lk.decode_series(df.get("vehicle_manoeuvre", pd.Series(index=df.index)), lk.VEHICLE_MANOEUVRE)
    out["junction_location"] = lk.decode_series(df.get("junction_location", pd.Series(index=df.index)), lk.JUNCTION_LOCATION)
    out["skidding_and_overturning"] = lk.decode_series(df.get("skidding_and_overturning", pd.Series(index=df.index)), lk.SKIDDING_AND_OVERTURNING)
    out["first_point_of_impact"] = lk.decode_series(df.get("first_point_of_impact", pd.Series(index=df.index)), lk.FIRST_POINT_OF_IMPACT)
    out["sex_of_driver"] = lk.decode_series(df.get("sex_of_driver", pd.Series(index=df.index)), lk.SEX)
    out["age_of_driver"] = df.get("age_of_driver")
    out["age_band_of_driver"] = lk.decode_series(df.get("age_band_of_driver", pd.Series(index=df.index)), lk.AGE_BAND)
    out["engine_capacity_cc"] = df.get("engine_capacity_cc")
    out["propulsion_code"] = lk.decode_series(df.get("propulsion_code", pd.Series(index=df.index)), lk.PROPULSION_CODE)
    out["age_of_vehicle"] = df.get("age_of_vehicle")
    out["journey_purpose"] = lk.decode_series(df.get("journey_purpose_of_driver", pd.Series(index=df.index)), lk.JOURNEY_PURPOSE)

    log_event(run_id, "vehicles", "transform", status="success",
              row_count=len(out), rejected_count=n0 - len(df),
              message=f"kept {len(out)}/{n0} (dropped rows whose accident was cleaned out)",
              started_at=started)
    print(f"[transform] vehicles: kept {len(out)}/{n0}")
    return out


def main() -> None:
    run_id = uuid.uuid4().hex[:12]
    print(f"[transform] run_id={run_id}")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    collisions_raw = pd.read_csv(latest_snapshot("collisions"), low_memory=False)
    casualties_raw = pd.read_csv(latest_snapshot("casualties"), low_memory=False)
    vehicles_raw = pd.read_csv(latest_snapshot("vehicles"), low_memory=False)

    cleaned = clean_collisions(collisions_raw, run_id)
    staged_accidents = decode_collisions(cleaned)
    staged_accidents = join_weather(staged_accidents, run_id)
    staged_accidents = join_road_context(staged_accidents, run_id)
    staged_accidents = finalize_weather_flags(staged_accidents)
    staged_accidents = staged_accidents.drop(columns=["grid_lat", "grid_lon"])

    valid_index = set(staged_accidents["accident_index"])
    staged_casualties = transform_casualties(casualties_raw, valid_index, run_id)
    staged_vehicles = transform_vehicles(vehicles_raw, valid_index, run_id)

    staged_accidents.to_parquet(STAGING_DIR / "accidents_staged.parquet", index=False)
    staged_casualties.to_parquet(STAGING_DIR / "casualties_staged.parquet", index=False)
    staged_vehicles.to_parquet(STAGING_DIR / "vehicles_staged.parquet", index=False)

    print(f"[transform] wrote staging parquet files to {STAGING_DIR}")
    print(f"  accidents:  {len(staged_accidents)} rows")
    print(f"  casualties: {len(staged_casualties)} rows")
    print(f"  vehicles:   {len(staged_vehicles)} rows")


if __name__ == "__main__":
    main()
