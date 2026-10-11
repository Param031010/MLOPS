"""Lazy async SQLAlchemy engine. Must not fail at import time if
DATABASE_URL is unset -- a request-time dependency raises a friendly 503
instead, so `uvicorn app.main:app` still boots for local frontend work.
"""

from __future__ import annotations

import os

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

_engine: AsyncEngine | None = None


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def get_engine() -> AsyncEngine | None:
    global _engine
    if _engine is not None:
        return _engine
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    _engine = create_async_engine(_to_asyncpg_url(url), pool_pre_ping=True)
    return _engine


async def get_connection():
    engine = get_engine()
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not configured on the server.",
        )
    async with engine.connect() as conn:
        yield conn


async def fetch_all(conn: AsyncConnection, sql: str, params: dict | None = None) -> list[dict]:
    result = await conn.execute(text(sql), params or {})
    return [dict(row) for row in result.mappings().all()]


async def fetch_one(conn: AsyncConnection, sql: str, params: dict | None = None) -> dict | None:
    rows = await fetch_all(conn, sql, params)
    return rows[0] if rows else None


def mart_filters(severity: list[int] | None = None, year_from: int | None = None,
                 year_to: int | None = None, **extra: list[str] | None) -> tuple[str, dict]:
    """WHERE clause + params for the optional dashboard filters on a mart.

    Every value is bound as a parameter; `extra` maps a column name to a list
    of allowed values (column names come from code, never from the request).
    """
    clauses: list[str] = []
    params: dict = {}
    if severity:
        clauses.append("severity_id = ANY(:severity)")
        params["severity"] = list(severity)
    if year_from is not None:
        clauses.append("year >= :year_from")
        params["year_from"] = year_from
    if year_to is not None:
        clauses.append("year <= :year_to")
        params["year_to"] = year_to
    for col, values in extra.items():
        if values:
            clauses.append(f"{col} = ANY(:{col})")
            params[col] = list(values)
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params
