#!/usr/bin/env python3
"""Fetch and cache raw NYISO demand, the operator's day-ahead forecast, and NYC temperature.

    export EIA_API_KEY=your_key
    python fetch_data.py                 # first run: Jan 1 2022 to now; later runs append new hours
    python fetch_data.py --refresh       # re-download everything

Writes three CSVs to forecasting/data/ so later steps never re-hit the APIs:
    nyiso_demand_raw.csv     EIA type D  (demand), hourly, UTC
    nyiso_forecast_raw.csv   EIA type DF (operator day-ahead demand forecast), hourly, UTC
    nyc_temp_raw.csv         Open-Meteo archive, hourly temperature in F, UTC
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
NYC_LAT, NYC_LON = 40.71, -74.01
PAGE = 5000  # EIA API v2 returns at most 5,000 rows per request


def fetch_eia(api_key: str, type_code: str, start: str, end: str, respondent: str = "NYIS") -> pd.DataFrame:
    """Hourly EIA region-data for one respondent and one type code, paging through all results.
    start/end look like 2022-01-01T00. EIA timestamps are UTC."""
    rows, offset = [], 0
    while True:
        params = [
            ("api_key", api_key), ("frequency", "hourly"), ("data[0]", "value"),
            ("facets[respondent][]", respondent), ("facets[type][]", type_code),
            ("start", start), ("end", end),
            ("sort[0][column]", "period"), ("sort[0][direction]", "asc"),
            ("offset", offset), ("length", PAGE),
        ]
        r = requests.get(EIA_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()["response"].get("data", [])
        rows.extend(data)
        if len(data) < PAGE:
            break
        offset += PAGE
        time.sleep(0.25)  # be polite to the API
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "value"])
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (df[["timestamp_utc", "value"]].drop_duplicates("timestamp_utc", keep="last")
            .sort_values("timestamp_utc").reset_index(drop=True))


def fetch_temperature(start: date, end: date) -> pd.DataFrame:
    """Hourly 2 m temperature (F) for New York City from the Open-Meteo archive, requested in UTC
    so it lines up with the EIA timestamps. Fetched one year at a time."""
    frames, cur = [], start
    while cur <= end:
        chunk_end = min(date(cur.year, 12, 31), end)
        r = requests.get(OPEN_METEO_URL, timeout=60, params={
            "latitude": NYC_LAT, "longitude": NYC_LON,
            "start_date": cur.isoformat(), "end_date": chunk_end.isoformat(),
            "hourly": "temperature_2m", "temperature_unit": "fahrenheit", "timezone": "UTC",
        })
        r.raise_for_status()
        h = r.json()["hourly"]
        frames.append(pd.DataFrame({
            "timestamp_utc": pd.to_datetime(h["time"], utc=True),
            "value": pd.to_numeric(pd.Series(h["temperature_2m"]), errors="coerce"),
        }))
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp_utc", keep="last")


def _merge_cache(path: Path, new: pd.DataFrame, keep_old: bool = True) -> pd.DataFrame:
    if keep_old and path.exists():
        old = pd.read_csv(path, parse_dates=["timestamp_utc"])
        new = pd.concat([old, new], ignore_index=True)
    new["timestamp_utc"] = pd.to_datetime(new["timestamp_utc"], utc=True)
    return new.drop_duplicates("timestamp_utc", keep="last").sort_values("timestamp_utc").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2022-01-01", help="first day to fetch (default 2022-01-01)")
    ap.add_argument("--refresh", action="store_true", help="ignore existing caches and re-download")
    ap.add_argument("--key", default=os.environ.get("EIA_API_KEY"), help="EIA API key (or set EIA_API_KEY)")
    args = ap.parse_args()
    if not args.key:
        print("Set EIA_API_KEY or pass --key.")
        return 1

    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    end_eia = now.strftime("%Y-%m-%dT%H")

    for type_code, fname in [("D", "nyiso_demand_raw.csv"), ("DF", "nyiso_forecast_raw.csv")]:
        path = DATA_DIR / fname
        start = f"{args.start}T00"
        if path.exists() and not args.refresh:
            last = pd.read_csv(path, parse_dates=["timestamp_utc"])["timestamp_utc"].max()
            start = (last - timedelta(days=7)).strftime("%Y-%m-%dT%H")  # overlap to pick up revisions
        print(f"EIA type {type_code}: {start} -> {end_eia}")
        df = fetch_eia(args.key, type_code, start, end_eia)
        out = _merge_cache(path, df, keep_old=not args.refresh)
        out.to_csv(path, index=False)
        print(f"  {len(df):,} rows fetched, {len(out):,} cached in {fname}")

    path = DATA_DIR / "nyc_temp_raw.csv"
    start_d = date.fromisoformat(args.start)
    if path.exists() and not args.refresh:
        last = pd.read_csv(path, parse_dates=["timestamp_utc"])["timestamp_utc"].max()
        start_d = (last - timedelta(days=14)).date()
    end_d = (now - timedelta(days=1)).date()
    print(f"Open-Meteo temperature: {start_d} -> {end_d}")
    out = _merge_cache(path, fetch_temperature(start_d, end_d), keep_old=not args.refresh)
    out.to_csv(path, index=False)
    print(f"  {len(out):,} rows cached in nyc_temp_raw.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
