"""Thin, cached HTTP client every page uses to reach the FastAPI backend.
No page renders bundled/placeholder data -- everything comes from here."""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:9001")


class ApiError(RuntimeError):
    pass


@st.cache_data(ttl=60, show_spinner=False)
def get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ApiError(f"GET {path} failed: {exc}") from exc
    return resp.json()


def post(path: str, json: dict | None = None, headers: dict | None = None):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json, headers=headers, timeout=30)
    except requests.RequestException as exc:
        raise ApiError(f"POST {path} failed: {exc}") from exc
    return resp


SEVERITY_IDS = {"Fatal": 1, "Serious": 2, "Slight": 3}


def sidebar_filters() -> dict:
    """Shared sidebar: year range + severity. Returns the query params every
    mart endpoint understands, so a change re-queries the API server-side
    instead of filtering a cached full table in the browser."""
    overview = get("/accidents/overview")
    years = overview.get("years_covered") or []
    st.sidebar.subheader("Filters")
    params: dict = {}
    if len(years) > 1:
        lo, hi = st.sidebar.slider("Years", min(years), max(years), (min(years), max(years)))
        params.update(year_from=lo, year_to=hi)
    chosen = st.sidebar.multiselect("Severity", list(SEVERITY_IDS), default=list(SEVERITY_IDS))
    if chosen and len(chosen) < len(SEVERITY_IDS):
        params["severity"] = [SEVERITY_IDS[c] for c in chosen]
    return params
