from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncConnection

from ..db import fetch_all, get_connection
from ..schemas import CasualtyByTypeRow, CasualtyDemographicsRow

router = APIRouter(prefix="/casualties", tags=["casualties"])


@router.get("/by-type", response_model=list[CasualtyByTypeRow])
async def by_type(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT casualty_type, casualty_class, severity_id, severity_label, casualty_count
        FROM mart_casualties_by_type ORDER BY casualty_count DESC
    """)


@router.get("/demographics", response_model=list[CasualtyDemographicsRow])
async def demographics(conn: AsyncConnection = Depends(get_connection)):
    return await fetch_all(conn, """
        SELECT age_band_of_casualty, sex_of_casualty, severity_id, severity_label, casualty_count
        FROM mart_casualty_demographics ORDER BY age_band_of_casualty, sex_of_casualty
    """)
