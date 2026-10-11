from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncConnection

from ..db import fetch_all, get_connection, mart_filters
from ..schemas import CasualtyByTypeRow, CasualtyDemographicsRow

router = APIRouter(prefix="/casualties", tags=["casualties"])


@router.get("/by-type", response_model=list[CasualtyByTypeRow])
async def by_type(severity: Optional[list[int]] = Query(None), year_from: Optional[int] = Query(None),
                  year_to: Optional[int] = Query(None), conn: AsyncConnection = Depends(get_connection)):
    where, params = mart_filters(severity, year_from, year_to)
    return await fetch_all(conn, f"""
        SELECT casualty_type, casualty_class, severity_id, severity_label,
               SUM(casualty_count)::bigint AS casualty_count
        FROM mart_casualties_by_type {where}
        GROUP BY casualty_type, casualty_class, severity_id, severity_label
        ORDER BY casualty_count DESC
    """, params)


@router.get("/demographics", response_model=list[CasualtyDemographicsRow])
async def demographics(severity: Optional[list[int]] = Query(None), year_from: Optional[int] = Query(None),
                       year_to: Optional[int] = Query(None), conn: AsyncConnection = Depends(get_connection)):
    where, params = mart_filters(severity, year_from, year_to)
    return await fetch_all(conn, f"""
        SELECT age_band_of_casualty, sex_of_casualty, severity_id, severity_label,
               SUM(casualty_count)::bigint AS casualty_count
        FROM mart_casualty_demographics {where}
        GROUP BY age_band_of_casualty, sex_of_casualty, severity_id, severity_label
        ORDER BY age_band_of_casualty, sex_of_casualty
    """, params)
