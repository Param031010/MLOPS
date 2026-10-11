from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncConnection

from ..db import fetch_all, get_connection, mart_filters
from ..schemas import VehicleByTypeRow

router = APIRouter(prefix="/vehicles", tags=["vehicles"])


@router.get("/by-type", response_model=list[VehicleByTypeRow])
async def by_type(severity: Optional[list[int]] = Query(None), year_from: Optional[int] = Query(None),
                  year_to: Optional[int] = Query(None), conn: AsyncConnection = Depends(get_connection)):
    where, params = mart_filters(severity, year_from, year_to)
    return await fetch_all(conn, f"""
        SELECT vehicle_type, severity_id, severity_label, SUM(vehicle_count)::bigint AS vehicle_count
        FROM mart_vehicles_by_type {where}
        GROUP BY vehicle_type, severity_id, severity_label
        ORDER BY vehicle_count DESC
    """, params)
