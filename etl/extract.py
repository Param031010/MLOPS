"""Extract stage: read local STATS19 CSVs, snapshot them immutably, and
enrich with Open-Meteo weather history and TomTom road context.

Usage:
    python extract.py [--skip-road] [--road-limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent))
from audit import log_event  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

RAW_DATASET_DIR = Path(__file__).parent.parent / "dataset" / "raw"
RAW_SNAPSHOT_DIR = Path(__file__).parent / "data" / "raw"

REQUIRED_COLUMNS = {
    "collisions.csv": {
        "collision_index", "collision_year", "longitude", "latitude",
        "collision_severity", "date", "time",
    },
    "casualties.csv": {
        "collision_index", "casualty_class", "casualty_severity",
    },
    "vehicles.csv": {
        "collision_index", "vehicle_type",
    },
}

GRID_STEP = 0.5
CHECKPOINT_EVERY = 25  # cells between weather_grid.parquet checkpoints
SNAPSHOT_RETENTION = 2  # how many timestamped raw snapshots to keep per source
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
TOMTOM_URL = "https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"


def snap_to_grid(value: float) -> float:
    return round(value / GRID_STEP) * GRID_STEP


def validate_and_read(filename: str) -> pd.DataFrame:
    path = RAW_DATASET_DIR / filename
    if not path.exists():
        raise SystemExit(
            f"Missing required input file: {path}\n"
            f"Place the STATS19 export there before running extract.py."
        )
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS[filename] - set(df.columns)
    if missing:
        raise SystemExit(
            f"{filename} is missing required column(s): {sorted(missing)}\n"
            f"Found columns: {sorted(df.columns)}"
        )
    return df


def snapshot(df: pd.DataFrame, name: str) -> Path:
    RAW_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RAW_SNAPSHOT_DIR / f"{name}_{stamp}.csv"
    df.to_csv(out_path, index=False)
    prune_snapshots(name)
    return out_path


def prune_snapshots(name: str, keep: int = SNAPSHOT_RETENTION) -> None:
    """Keep only the newest `keep` snapshots per source. Each set is ~250MB,
    so without this every run permanently grows the raw zone."""
    existing = sorted(RAW_SNAPSHOT_DIR.glob(f"{name}_*.csv"))
    for stale in existing[:-keep]:
        stale.unlink(missing_ok=True)
        print(f"  pruned old snapshot {stale.name}")


class RateLimited(RuntimeError):
    """Open-Meteo's free tier refused the request (quota exhausted)."""


def _retryable(exc: BaseException) -> bool:
    """Retry transient network/HTTP errors, but never a quota refusal --
    hammering a 429 just burns the window that's already closed."""
    return not isinstance(exc, RateLimited)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30),
       retry=retry_if_exception(_retryable), reraise=True)
def fetch_weather_year(grid_lat: float, grid_lon: float, year: int) -> pd.DataFrame:
    params = {
        "latitude": grid_lat,
        "longitude": grid_lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
        "timezone": "Europe/London",
    }
    key = os.getenv("OPEN_METEO_API_KEY")
    if key:
        params["apikey"] = key
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    if resp.status_code == 429:
        raise RateLimited(f"429 from Open-Meteo for ({grid_lat},{grid_lon},{year})")
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    out = pd.DataFrame({
        "datetime_local": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "precipitation_mm": hourly["precipitation"],
        "wind_speed_kmh": hourly["windspeed_10m"],
        "weather_code_meteo": hourly["weathercode"],
    })
    out["grid_lat"] = grid_lat
    out["grid_lon"] = grid_lon
    out["year"] = year
    out["date"] = out["datetime_local"].dt.date
    out["hour"] = out["datetime_local"].dt.hour
    return out


def extract_weather(collisions: pd.DataFrame, run_id: str) -> None:
    started = datetime.now(timezone.utc)
    coords = collisions.dropna(subset=["latitude", "longitude", "collision_year"]).copy()
    coords["grid_lat"] = coords["latitude"].apply(snap_to_grid)
    coords["grid_lon"] = coords["longitude"].apply(snap_to_grid)
    cells = coords[["grid_lat", "grid_lon", "collision_year"]].drop_duplicates()
    cells = cells.rename(columns={"collision_year": "year"})
    cells["year"] = cells["year"].astype(int)

    out_path = RAW_SNAPSHOT_DIR / "weather_grid.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else None
    done_keys = set()
    if existing is not None:
        done_keys = set(zip(existing["grid_lat"], existing["grid_lon"], existing["year"]))

    todo = [
        (row.grid_lat, row.grid_lon, row.year)
        for row in cells.itertuples()
        if (row.grid_lat, row.grid_lon, row.year) not in done_keys
    ]

    print(f"[extract] weather: {len(done_keys)} grid-year cells cached, "
          f"{len(todo)} to fetch from Open-Meteo")

    def checkpoint(frames: list[pd.DataFrame]) -> None:
        """Persist what's been fetched so far. Called periodically so an
        interrupted or rate-limited run keeps its progress -- the next run
        skips these cells instead of re-fetching them."""
        if not frames:
            return
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["grid_lat", "grid_lon", "date", "hour"])
        combined.to_parquet(out_path, index=False)

    frames = [existing] if existing is not None else []
    failures = 0
    rate_limited = False
    fetched = 0

    for i, (grid_lat, grid_lon, year) in enumerate(todo, start=1):
        try:
            frames.append(fetch_weather_year(grid_lat, grid_lon, year))
            fetched += 1
        except RateLimited:
            # The free tier's quota is weighted by locations x days x variables,
            # so a full-year hourly backfill exhausts it well before the request
            # count looks high. Stop cleanly, keep what we have, and let the
            # next run pick up exactly where this one left off.
            print(f"  ! Open-Meteo rate limit reached after {fetched} new cells this run.")
            print("    Progress is checkpointed -- re-run extract.py later to continue.")
            rate_limited = True
            break
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ! weather fetch failed for ({grid_lat},{grid_lon},{year}): {exc}")
            continue

        if i % CHECKPOINT_EVERY == 0:
            checkpoint(frames)
        if i % 10 == 0 or i == len(todo):
            print(f"  fetched {i}/{len(todo)} grid-year cells", flush=True)
        time.sleep(0.2)  # be polite to the free API

    checkpoint(frames)

    remaining = len(todo) - fetched - failures
    status = "success" if (failures == 0 and not rate_limited) else "partial"
    log_event(
        run_id, "weather", "extract",
        status=status,
        row_count=fetched,
        rejected_count=failures,
        message=(f"{len(todo)} grid-year cells to fetch, {fetched} fetched, "
                 f"{failures} failed, {remaining} remaining"
                 + (" (rate limited -- re-run to continue)" if rate_limited else "")),
        started_at=started,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10),
       reraise=True)
def reverse_geocode(lat: float, lon: float, api_key: str) -> str | None:
    resp = requests.get(
        TOMTOM_URL.format(lat=lat, lon=lon),
        params={"key": api_key},
        timeout=15,
    )
    if resp.status_code == 403:
        raise RuntimeError("TomTom quota exhausted or key rejected (403)")
    resp.raise_for_status()
    addresses = resp.json().get("addresses", [])
    if not addresses:
        return None
    address = addresses[0].get("address", {})
    return address.get("streetName") or address.get("municipality")


def extract_road_context(collisions: pd.DataFrame, run_id: str, limit: int) -> None:
    started = datetime.now(timezone.utc)
    out_path = RAW_SNAPSHOT_DIR / "road_context.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(
        columns=["latitude", "longitude", "street_name", "resolved_at"]
    )
    resolved = set(zip(existing["latitude"], existing["longitude"]))

    counts = (
        collisions.dropna(subset=["latitude", "longitude"])
        .groupby(["latitude", "longitude"]).size()
        .reset_index(name="accident_count")
        .sort_values("accident_count", ascending=False)
    )
    todo = [
        (row.latitude, row.longitude)
        for row in counts.itertuples()
        if (row.latitude, row.longitude) not in resolved
    ][:limit]

    api_key = os.getenv("TOMTOM_API_KEY")
    if not api_key:
        print("[extract] TOMTOM_API_KEY not set - skipping road context enrichment")
        log_event(run_id, "road_context", "extract", status="success",
                  row_count=0, message="skipped: no TOMTOM_API_KEY", started_at=started)
        return

    print(f"[extract] road context: {len(resolved)} coordinates cached, "
          f"resolving up to {len(todo)} more (busiest first)")

    new_rows = []
    failures = 0
    for i, (lat, lon) in enumerate(todo, start=1):
        try:
            street = reverse_geocode(lat, lon, api_key)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ! reverse geocode failed for ({lat},{lon}): {exc}")
            if "quota" in str(exc).lower() or "403" in str(exc):
                print("  stopping road-context enrichment: quota appears exhausted")
                break
            continue
        new_rows.append({
            "latitude": lat, "longitude": lon, "street_name": street,
            "resolved_at": datetime.now(timezone.utc),
        })
        if i % 25 == 0 or i == len(todo):
            print(f"  resolved {i}/{len(todo)} coordinates")
        time.sleep(0.1)

    if new_rows:
        combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_parquet(out_path, index=False)

    log_event(
        run_id, "road_context", "extract",
        status="success" if failures == 0 else "partial",
        row_count=len(new_rows), rejected_count=failures,
        message=f"resolved {len(new_rows)} of {len(todo)} attempted",
        started_at=started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract stage of the traffic accident ETL")
    parser.add_argument("--skip-road", action="store_true", help="Skip TomTom road-context enrichment")
    parser.add_argument("--road-limit", type=int, default=300,
                        help="Max new coordinates to reverse-geocode this run")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:12]
    print(f"[extract] run_id={run_id}")

    for filename in ["collisions.csv", "casualties.csv", "vehicles.csv"]:
        started = datetime.now(timezone.utc)
        df = validate_and_read(filename)
        name = filename.replace(".csv", "")
        snapshot(df, name)
        log_event(run_id, name, "extract", status="success", row_count=len(df),
                  message=f"read and snapshotted {filename}", started_at=started)
        print(f"[extract] {filename}: {len(df)} rows snapshotted")
        if name == "collisions":
            collisions = df

    extract_weather(collisions, run_id)

    if args.skip_road:
        log_event(run_id, "road_context", "extract", status="success",
                  row_count=0, message="skipped via --skip-road")
        print("[extract] road context skipped (--skip-road)")
    else:
        extract_road_context(collisions, run_id, args.road_limit)

    print("[extract] done.")


if __name__ == "__main__":
    main()
