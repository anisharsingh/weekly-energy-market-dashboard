#!/usr/bin/env python3
"""Refresh data/weekly.csv: add new weeks and fill blank cells.

Sources (all free):
  EIA API v2   crude/Cushing/gasoline stocks, refinery utilization, gas storage,
               Henry Hub spot, daily US lower-48 electricity demand
  Yahoo Finance (yfinance)  WTI (CL=F) and Brent (BZ=F) front-month Friday closes
  Open-Meteo   daily temperatures -> weekly HDD/CDD proxy (5 big US cities)

Usage:
  export EIA_API_KEY=...          # free key: https://www.eia.gov/opendata/
  python update_data.py           # fills blanks only
  python update_data.py --overwrite   # replace existing values too

Each source is wrapped in try/except, so one failing API never blocks the rest.
A "week" is the EIA week: Saturday through Friday, labelled by the Friday.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from contract_rolls import roll_dates_by_week

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "weekly.csv"
EIA_BASE = "https://api.eia.gov/v2"

# column -> (route, series id, scale to the unit used in the CSV)
EIA_WEEKLY = {
    "crude_stocks_mb": ("petroleum/stoc/wstk", "WCESTUS1", 1 / 1000),            # kbbl -> mb
    "cushing_mb": ("petroleum/stoc/wstk", "W_EPC0_SAX_YCUOK_MBBL", 1 / 1000),
    "gasoline_stocks_mb": ("petroleum/stoc/wstk", "WGTSTUS1", 1 / 1000),
    "refinery_util_pct": ("petroleum/pnp/wiup", "WPULEUS3", 1.0),
    "natgas_storage_bcf": ("natural-gas/stor/wkly", "NW2_EPG0_SWO_R48_BCF", 1.0),
}

CITIES = {  # rough national proxy; NOAA CPC publishes the population-weighted version
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Houston": (29.76, -95.37),
    "Atlanta": (33.75, -84.39),
    "Los Angeles": (34.05, -118.24),
}


def eia(route: str, key: str, start: str, end: str, frequency: str, **facets) -> pd.Series:
    params = [
        ("api_key", key), ("frequency", frequency), ("data[0]", "value"),
        ("start", start), ("end", end),
        ("sort[0][column]", "period"), ("sort[0][direction]", "asc"), ("length", "5000"),
    ]
    for name, val in facets.items():
        params.append((f"facets[{name}][]", val))
    r = requests.get(f"{EIA_BASE}/{route}/data/", params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("period")["value"].sort_index()


def fill(df: pd.DataFrame, col: str, values: pd.Series, overwrite: bool) -> int:
    n = 0
    for wk, v in values.items():
        if pd.isna(v):
            continue
        mask = df["week_ending"] == wk
        if mask.any() and (overwrite or df.loc[mask, col].isna().all()):
            df.loc[mask, col] = round(float(v), 3)
            n += 1
    return n


def last_friday(today: date) -> date:
    return today - timedelta(days=(today.weekday() - 4) % 7)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true", help="replace existing values, not just blanks")
    ap.add_argument("--key", default=os.environ.get("EIA_API_KEY"), help="EIA API key (or set EIA_API_KEY)")
    args = ap.parse_args()

    df = pd.read_csv(CSV, parse_dates=["week_ending"])
    df = df.sort_values("week_ending").reset_index(drop=True)

    # 1) add any missing Fridays up to the most recent one
    all_weeks = pd.date_range(df["week_ending"].min(), pd.Timestamp(last_friday(date.today())), freq="W-FRI")
    new = [w for w in all_weeks if w not in set(df["week_ending"])]
    if new:
        df = pd.concat([df, pd.DataFrame({"week_ending": new})], ignore_index=True).sort_values("week_ending")
        df = df.reset_index(drop=True)
        print(f"Added {len(new)} new week(s): {', '.join(w.strftime('%Y-%m-%d') for w in new)}")
    weeks = pd.DatetimeIndex(df["week_ending"])
    start = (weeks.min() - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (weeks.max() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 2) EIA weekly series
    if args.key:
        for col, (route, series, scale) in EIA_WEEKLY.items():
            try:
                s = eia(route, args.key, start, end, "weekly", series=series) * scale
                print(f"{col}: filled {fill(df, col, s.reindex(weeks), args.overwrite)} cell(s)")
            except Exception as e:  # noqa: BLE001
                print(f"{col}: skipped ({e})")

        # Henry Hub spot (daily): value on the Friday, or the last prior trading day
        try:
            s = eia("natural-gas/pri/fut", args.key, start, end, "daily", series="RNGWHHD")
            s = s.reindex(pd.date_range(s.index.min(), s.index.max())).ffill(limit=3)
            print(f"henry_hub_spot: filled {fill(df, 'henry_hub_spot', s.reindex(weeks), args.overwrite)} cell(s)")
        except Exception as e:  # noqa: BLE001
            print(f"henry_hub_spot: skipped ({e})")

        # Lower-48 electricity demand: average GWh/day over the Sat-Fri week
        try:
            s = eia("electricity/rto/daily-region-data", args.key, start, end, "daily",
                    respondent="US48", type="D", timezone="Eastern")
            vals = {}
            for wk in weeks:
                chunk = s[wk - timedelta(days=6): wk]
                if len(chunk) == 7:
                    vals[wk] = chunk.mean() / 1000  # MWh -> GWh
            print(f"elec_demand_gwh_day: filled {fill(df, 'elec_demand_gwh_day', pd.Series(vals), args.overwrite)} cell(s)")
        except Exception as e:  # noqa: BLE001
            print(f"elec_demand_gwh_day: skipped ({e})")
    else:
        print("No EIA key: skipping inventories, utilization, gas storage, Henry Hub, electricity.")

    # 3) Futures settles from Yahoo Finance
    try:
        import yfinance as yf

        px = yf.download(["CL=F", "BZ=F"], start=start, end=end, progress=False, auto_adjust=False)["Close"].ffill()
        px = px.reindex(weeks, method="ffill")
        print(f"wti_settle: filled {fill(df, 'wti_settle', px['CL=F'], args.overwrite)} cell(s)")
        print(f"brent_settle: filled {fill(df, 'brent_settle', px['BZ=F'], args.overwrite)} cell(s)")
    except Exception as e:  # noqa: BLE001
        print(f"prices: skipped ({e})")

    # 4) Weather proxy from Open-Meteo
    try:
        frames = []
        for name, (lat, lon) in CITIES.items():
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=30, params={
                "latitude": lat, "longitude": lon, "start_date": start, "end_date": min(end, date.today().isoformat()),
                "daily": "temperature_2m_mean", "temperature_unit": "fahrenheit", "timezone": "America/New_York"})
            r.raise_for_status()
            d = r.json()["daily"]
            frames.append(pd.Series(d["temperature_2m_mean"], index=pd.to_datetime(d["time"]), name=name))
        t = pd.concat(frames, axis=1).mean(axis=1)
        hdd, cdd = {}, {}
        for wk in weeks:
            chunk = t[wk - timedelta(days=6): wk].dropna()
            if len(chunk) == 7:
                hdd[wk] = (65 - chunk).clip(lower=0).sum()
                cdd[wk] = (chunk - 65).clip(lower=0).sum()
        print(f"hdd: filled {fill(df, 'hdd', pd.Series(hdd), args.overwrite)} cell(s)")
        print(f"cdd: filled {fill(df, 'cdd', pd.Series(cdd), args.overwrite)} cell(s)")
    except Exception as e:  # noqa: BLE001
        print(f"weather: skipped ({e})")

    # 5) Futures contract roll weeks: computed from the exchange expiry rules (no network needed)
    rolls = roll_dates_by_week(weeks)
    for col in ("wti_roll_date", "brent_roll_date"):
        values = [rolls[col].get(w, "") for w in df["week_ending"]]
        if col in df.columns:
            df[col] = values
        else:
            pos = df.columns.get_loc("data_notes") if "data_notes" in df.columns else len(df.columns)
            df.insert(pos, col, values)
    print(f"contract rolls: {sum(1 for c in ('wti_roll_date', 'brent_roll_date') for v in df[c] if v)} roll flag(s) set")

    df["week_ending"] = df["week_ending"].dt.strftime("%Y-%m-%d")
    df.to_csv(CSV, index=False)
    print(f"Saved {CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
