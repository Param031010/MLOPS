"""Pydantic response models, one per mart/endpoint shape."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel


class Overview(BaseModel):
    total_accidents: int
    total_casualties: int
    total_vehicles: int
    fatal_count: int
    serious_count: int
    slight_count: int
    years_covered: list[int]
    road_context_coverage_pct: float
    weather_coverage_pct: float


class SeverityDistributionRow(BaseModel):
    severity_id: int
    severity_label: str
    accident_count: int
    total_casualties: Optional[int] = None


class ByHourWeekdayRow(BaseModel):
    hour: int
    day_of_week: int
    day_name: str
    severity_id: int
    severity_label: str
    accident_count: int


class MonthlyTrendRow(BaseModel):
    year: int
    month: int
    month_name: str
    severity_id: int
    severity_label: str
    accident_count: int
    total_casualties: Optional[int] = None
    total_vehicles: Optional[int] = None


class WeatherImpactRow(BaseModel):
    weather_conditions: Optional[str] = None
    severity_id: int
    severity_label: str
    accident_count: int
    avg_temperature_c: Optional[float] = None
    avg_precipitation_mm: Optional[float] = None


class RoadTypeSeverityRow(BaseModel):
    road_type: Optional[str] = None
    severity_id: int
    severity_label: str
    accident_count: int


class HotspotRow(BaseModel):
    accident_index: str
    latitude: float
    longitude: float
    street_name: Optional[str] = None
    full_date: date
    year: int
    severity_id: int
    severity_label: str
    number_of_casualties: Optional[int] = None


class TopLocationRow(BaseModel):
    latitude: float
    longitude: float
    street_name: Optional[str] = None
    local_authority_district: Optional[str] = None
    accident_count: int
    total_casualties: Optional[int] = None
    fatal_count: int


class YearlyCasualtyVehicleRow(BaseModel):
    year: int
    total_accidents: int
    total_casualties: int
    total_vehicles: int


class CasualtyByTypeRow(BaseModel):
    casualty_type: Optional[str] = None
    casualty_class: Optional[str] = None
    severity_id: int
    severity_label: str
    casualty_count: int


class CasualtyDemographicsRow(BaseModel):
    age_band_of_casualty: Optional[str] = None
    sex_of_casualty: Optional[str] = None
    severity_id: int
    severity_label: str
    casualty_count: int


class VehicleByTypeRow(BaseModel):
    vehicle_type: Optional[str] = None
    severity_id: int
    severity_label: str
    vehicle_count: int


class EtlRunStatus(BaseModel):
    status: str  # idle | running | success | failed
    run_id: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None


class PredictRequest(BaseModel):
    hour: int
    day_of_week: int
    month: int
    road_type: str
    speed_limit: int
    junction_detail: str
    light_conditions: str
    weather_conditions: str
    road_surface_conditions: str
    urban_or_rural_area: str
    number_of_vehicles: int
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    is_weekend: Optional[bool] = None


class PredictResponse(BaseModel):
    predicted_severity_id: int
    predicted_severity_label: str
    probabilities: dict[str, float]
    model_version: str


class ModelInfo(BaseModel):
    model_name: str
    version: str
    stage: str
    algorithm: Optional[str] = None
    trained_at: Optional[str] = None
    metrics: dict[str, float] = {}


class ModelHealth(BaseModel):
    model_loaded: bool
    model_version: Optional[str] = None


class DriftReport(BaseModel):
    # kind distinguishes the three drift families the project monitors:
    # "feature" (model inputs), "location" (where accidents are happening) and
    # "class" (the severity mix itself). For class rows, baseline_value /
    # current_value are shares of that class rather than means.
    kind: str  # feature | location | class
    feature: str
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    psi: Optional[float] = None
    drifted: bool


class LatencyReport(BaseModel):
    sampled_requests: int
    predictions_served: int
    failures: int
    failure_rate: float
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    max_ms: Optional[float] = None


class PerformanceReport(BaseModel):
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    macro_f1: Optional[float] = None
    recall_macro: Optional[float] = None
    false_negative_rate_fatal_serious: Optional[float] = None
    sample_size: int
