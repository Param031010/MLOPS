from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncConnection

from ..db import fetch_all, fetch_one, get_connection
from ..schemas import (
    ByHourWeekdayRow,
    HotspotRow,
    MonthlyTrendRow,
    Overview,
    RoadTypeSeverityRow,
    SeverityDistributionRow,
    TopLocationRow,
    WeatherImpactRow,
    YearlyCasualtyVehicleRow,
)

router = APIRouter(prefix="/accidents", tags=["accidents"])


@router.get("/overview", response_model=Overview)
async def overview(conn: AsyncConnection = Depends(get_connection)):
    row = await fetch_one(conn, """
        WITH acc AS (
            SELECT COUNT(*) AS total_accidents,
                   SUM(CASE WHEN severity_id = 1 THEN 1 ELSE 0 END) AS fatal_count,
                   SUM(CASE WHEN severity_id = 2 THEN 1 ELSE 0 END) AS serious_count,
                   SUM(CASE WHEN severity_id = 3 THEN 1 ELSE 0 END) AS slight_count
            FROM fact_accidents
        ),
        cas AS (SELECT COUNT(*) AS total_casualties FROM fact_casualties),
        veh AS (SELECT COUNT(*) AS total_vehicles FROM fact_vehicles),
        yrs AS (SELECT COALESCE(array_agg(DISTINCT source_year ORDER BY source_year), '{}') AS years_covered
                FROM fact_accidents),
        road_cov AS (
            SELECT COALESCE(AVG(CASE WHEN l.street_name IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 0) AS pct
            FROM fact_accidents fa JOIN dim_location l ON l.location_id = fa.location_id
        ),
        weather_cov AS (
            SELECT COALESCE(AVG(CASE WHEN w.temperature_c IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 0) AS pct
            FROM fact_accidents fa JOIN dim_weather w ON w.weather_id = fa.weather_id
        )
        SELECT acc.total_accidents, cas.total_casualties, veh.total_vehicles,
               acc.fatal_count, acc.serious_count, acc.slight_count,
               yrs.years_covered, road_cov.pct AS road_context_coverage_pct,
               weather_cov.pct AS weather_coverage_pct
        FROM acc, cas, veh, yrs, road_cov, weather_cov
    """)
    if row is None or row["total_accidents"] == 0:
        return Overview(
            total_accidents=0, total_casualties=0, total_vehicles=0,
            fatal_count=0, serious_count=0, slight_count=0, years_covered=[],
            road_context_coverage_pct=0.0, weather_coverage_pct=0.0,
        )
    return row


@router.get("/severity-distribution", response_model=list[SeverityDistributionRow])
async def severity_distribution(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT severity_id, severity_label, accident_count, total_casualties
        FROM mart_severity_distribution ORDER BY severity_id
    """)


@router.get("/by-hour", response_model=list[ByHourWeekdayRow])
async def by_hour(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT hour, day_of_week, day_name, severity_id, severity_label, accident_count
        FROM mart_by_hour_weekday ORDER BY day_of_week, hour, severity_id
    """)


@router.get("/monthly-trend", response_model=list[MonthlyTrendRow])
async def monthly_trend(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT year, month, month_name, severity_id, severity_label,
               accident_count, total_casualties, total_vehicles
        FROM mart_monthly_trend ORDER BY year, month, severity_id
    """)


@router.get("/weather-impact", response_model=list[WeatherImpactRow])
async def weather_impact(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT weather_conditions, severity_id, severity_label, accident_count,
               avg_temperature_c, avg_precipitation_mm
        FROM mart_weather_impact ORDER BY accident_count DESC
    """)


@router.get("/road-type-severity", response_model=list[RoadTypeSeverityRow])
async def road_type_severity(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT road_type, severity_id, severity_label, accident_count
        FROM mart_road_type_severity ORDER BY accident_count DESC
    """)


@router.get("/hotspots", response_model=list[HotspotRow])
async def hotspots(
    severity: Optional[int] = Query(None, description="Filter by severity_id"),
    date_from: Optional[date] = Query(None),
    date_to: Optional[date] = Query(None),
    limit: int = Query(5000, le=20000),
    conn: AsyncConnection = Depends(get_connection),
):
    clauses = []
    params: dict = {"limit": limit}
    if severity is not None:
        clauses.append("severity_id = :severity")
        params["severity"] = severity
    if date_from is not None:
        clauses.append("full_date >= :date_from")
        params["date_from"] = date_from
    if date_to is not None:
        clauses.append("full_date <= :date_to")
        params["date_to"] = date_to
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return await fetch_all(conn, f"""
        SELECT accident_index, latitude, longitude, street_name, full_date, year,
               severity_id, severity_label, number_of_casualties
        FROM mart_hotspots
        {where}
        ORDER BY full_date DESC
        LIMIT :limit
    """, params)


@router.get("/top-locations", response_model=list[TopLocationRow])
async def top_locations(limit: int = Query(50, le=500), conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT latitude, longitude, street_name, local_authority_district,
               accident_count, total_casualties, fatal_count
        FROM mart_top_locations
        ORDER BY accident_count DESC
        LIMIT :limit
    """, {"limit": limit})


@router.get("/casualties", response_model=list[YearlyCasualtyVehicleRow])
async def casualties_per_year(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT d.year AS year,
               COUNT(DISTINCT fa.accident_index) AS total_accidents,
               COUNT(DISTINCT fc.casualty_key) AS total_casualties,
               COUNT(DISTINCT fv.vehicle_key) AS total_vehicles
        FROM fact_accidents fa
        JOIN dim_date d ON d.date_key = fa.date_key
        LEFT JOIN fact_casualties fc ON fc.accident_index = fa.accident_index
        LEFT JOIN fact_vehicles fv ON fv.accident_index = fa.accident_index
        GROUP BY d.year
        ORDER BY d.year
    """)
