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
