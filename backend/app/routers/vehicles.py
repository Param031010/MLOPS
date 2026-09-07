from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncConnection

from ..db import fetch_all, get_connection
from ..schemas import VehicleByTypeRow

router = APIRouter(prefix="/vehicles", tags=["vehicles"])


@router.get("/by-type", response_model=list[VehicleByTypeRow])
async def by_type(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT vehicle_type, severity_id, severity_label, vehicle_count
        FROM mart_vehicles_by_type ORDER BY vehicle_count DESC
    """)
