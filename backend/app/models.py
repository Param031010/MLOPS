"""SQLAlchemy ORM models mirroring sql/schema.sql.

These are the typed, in-code description of the star schema. Nothing
imports them at runtime today: the routers read the `mart_*` views with
raw SQL (they return plain aggregates, not entities) and ml/train.py
pulls its training set with a single hand-written join, both of which are
a better fit than an ORM for read-only analytical queries.

They are kept because they are the authoritative Python-side mirror of
`sql/schema.sql` -- the reference for column names and types when writing
those queries, and the starting point if any write-path or per-row entity
access is ever added.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, Numeric, SmallInteger, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DimSeverity(Base):
    __tablename__ = "dim_severity"

    severity_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    severity_label: Mapped[str] = mapped_column(String)


class DimDate(Base):
    __tablename__ = "dim_date"

    date_key: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    full_date: Mapped[date] = mapped_column(Date)
    hour: Mapped[int] = mapped_column(SmallInteger)
    day_of_week: Mapped[int] = mapped_column(SmallInteger)
    day_name: Mapped[str] = mapped_column(String)
    day_of_month: Mapped[int] = mapped_column(SmallInteger)
    month: Mapped[int] = mapped_column(SmallInteger)
    month_name: Mapped[str] = mapped_column(String)
    quarter: Mapped[int] = mapped_column(SmallInteger)
    year: Mapped[int] = mapped_column(SmallInteger)
    is_weekend: Mapped[bool] = mapped_column(Boolean)


class DimLocation(Base):
    __tablename__ = "dim_location"

    location_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    latitude: Mapped[float] = mapped_column(Numeric(9, 6))
    longitude: Mapped[float] = mapped_column(Numeric(9, 6))
    location_easting_osgr: Mapped[int | None]
    location_northing_osgr: Mapped[int | None]
    lsoa_of_accident_location: Mapped[str | None]
    local_authority_district: Mapped[str | None]
    local_authority_ons_district: Mapped[str | None]
    local_authority_highway: Mapped[str | None]
    police_force: Mapped[str | None]
    urban_or_rural_area: Mapped[str | None]
    street_name: Mapped[str | None]
    road_context_resolved_at: Mapped[datetime | None]


class DimRoad(Base):
    __tablename__ = "dim_road"

    road_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    first_road_class: Mapped[str | None]
    road_type: Mapped[str | None]
    speed_limit: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    junction_detail: Mapped[str | None]
    junction_control: Mapped[str | None]
    second_road_class: Mapped[str | None]
    pedestrian_crossing: Mapped[str | None]
    special_conditions_at_site: Mapped[str | None]
    carriageway_hazards: Mapped[str | None]
    trunk_road_flag: Mapped[str | None]


class DimWeather(Base):
    __tablename__ = "dim_weather"

    weather_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    weather_conditions: Mapped[str | None]
    road_surface_conditions: Mapped[str | None]
    light_conditions: Mapped[str | None]
    temperature_c: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    precipitation_mm: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    weather_code_meteo: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    is_raining: Mapped[bool | None]
    is_freezing: Mapped[bool | None]


class FactAccident(Base):
    __tablename__ = "fact_accidents"

    accident_key: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    accident_index: Mapped[str] = mapped_column(String, unique=True)
    date_key: Mapped[int] = mapped_column(BigInteger, ForeignKey("dim_date.date_key"))
    location_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("dim_location.location_id"))
    road_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("dim_road.road_id"))
    weather_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("dim_weather.weather_id"))
    severity_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("dim_severity.severity_id"))
    number_of_vehicles: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    number_of_casualties: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    police_force: Mapped[str | None]
    did_police_officer_attend: Mapped[str | None]
    source_year: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    loaded_at: Mapped[datetime]


class FactCasualty(Base):
    __tablename__ = "fact_casualties"

    casualty_key: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    accident_index: Mapped[str] = mapped_column(String, ForeignKey("fact_accidents.accident_index"))
    vehicle_reference: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    casualty_reference: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    casualty_class: Mapped[str | None]
    sex_of_casualty: Mapped[str | None]
    age_of_casualty: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    age_band_of_casualty: Mapped[str | None]
    casualty_severity_id: Mapped[int | None] = mapped_column(
        SmallInteger, ForeignKey("dim_severity.severity_id"), nullable=True
    )
    pedestrian_location: Mapped[str | None]
    pedestrian_movement: Mapped[str | None]
    car_passenger: Mapped[str | None]
    bus_or_coach_passenger: Mapped[str | None]
    casualty_type: Mapped[str | None]
    casualty_imd_decile: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)


class FactVehicle(Base):
    __tablename__ = "fact_vehicles"

    vehicle_key: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    accident_index: Mapped[str] = mapped_column(String, ForeignKey("fact_accidents.accident_index"))
    vehicle_reference: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    vehicle_type: Mapped[str | None]
    towing_and_articulation: Mapped[str | None]
    vehicle_manoeuvre: Mapped[str | None]
    junction_location: Mapped[str | None]
    skidding_and_overturning: Mapped[str | None]
    first_point_of_impact: Mapped[str | None]
    sex_of_driver: Mapped[str | None]
    age_of_driver: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    age_band_of_driver: Mapped[str | None]
    engine_capacity_cc: Mapped[int | None]
    propulsion_code: Mapped[str | None]
    age_of_vehicle: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    journey_purpose: Mapped[str | None]


class EtlAuditLog(Base):
    __tablename__ = "etl_audit_log"

    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String)
    row_count: Mapped[int | None]
    rejected_count: Mapped[int | None]
    status: Mapped[str] = mapped_column(String)
    message: Mapped[str | None]
    started_at: Mapped[datetime]
    finished_at: Mapped[datetime]
