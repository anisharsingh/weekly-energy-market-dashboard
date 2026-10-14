"""Clean the raw caches into one hourly table, check the timezone, and write a quality report.

    python clean.py        # reads data/*_raw.csv, writes data/hourly.csv, quality_report.json,
                           # charts/daily_shape.png

Everything is kept on a regular hourly UTC index (no duplicate or missing timestamps), so lags
counted in rows equal lags counted in elapsed hours. Calendar features use America/New_York.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CHART_DIR = HERE / "charts"
TZ = "America/New_York"


def read_raw(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.drop_duplicates("timestamp_utc", keep="last").set_index("timestamp_utc")["value"].sort_index()


def clean_series(raw: pd.Series, full_index: pd.DatetimeIndex, max_gap: int = 3, spike_tol: float = 0.25,
                 positive_only: bool = True) -> tuple[pd.Series, dict]:
    """Put a series on the full hourly index and repair obvious data problems.

    1. Missing hours stay NaN (counted).
    2. Non-positive values are impossible for load and are set to NaN.
    3. A value is a spike if it differs from the MEDIAN of the four surrounding hours (t-2, t-1, t+1, t+2)
       by more than spike_tol (25%); hourly load almost never moves that much, so it is treated as an
       error. The median is used so a single bad hour does not make its good neighbours look like spikes.
    4. Gaps of at most max_gap hours are filled by linear interpolation in time; longer gaps stay NaN
       and the affected rows are dropped later rather than invented.
    """
    s = raw.reindex(full_index)
    report = {"expected_hours": int(len(full_index)), "missing_hours": int(s.isna().sum())}
    nonpos = (s <= 0) if positive_only else pd.Series(False, index=s.index)
    report["nonpositive_values"] = int(nonpos.sum())
    s = s.mask(nonpos)
    neighbours = pd.concat([s.shift(2), s.shift(1), s.shift(-1), s.shift(-2)], axis=1).median(axis=1)
    spikes = ((s - neighbours).abs() / neighbours) > spike_tol  # NaN comparisons evaluate to False
    report["spikes"] = int(spikes.sum())
    s = s.mask(spikes)
    before = int(s.isna().sum())
    if max_gap > 0:
        s = s.interpolate(method="time", limit=max_gap, limit_area="inside")
    report["interpolated_hours"] = before - int(s.isna().sum())
    report["remaining_missing_hours"] = int(s.isna().sum())
    return s, report


def check_daily_shape(demand: pd.Series, chart_path: Path | None = None) -> dict:
    """Timezone sanity check: average load by LOCAL hour should dip overnight and peak in the
    afternoon/evening. If timestamps were wrongly treated as local, the shape is shifted by 4-5 hours."""
    local_hour = demand.index.tz_convert(TZ).hour
    by_hour = demand.groupby(local_hour).mean()
    utc_by_hour = demand.groupby(demand.index.hour).mean()
    low, peak = int(by_hour.idxmin()), int(by_hour.idxmax())
    ok = (2 <= low <= 6) and (14 <= peak <= 20)
    result = {"local_min_hour": low, "local_peak_hour": peak, "shape_ok": bool(ok),
              "utc_min_hour": int(utc_by_hour.idxmin()), "utc_peak_hour": int(utc_by_hour.idxmax())}
    if chart_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(by_hour.index, by_hour.values, marker="o", label=f"Local ({TZ})")
        ax.plot(utc_by_hour.index, utc_by_hour.values, marker="x", ls="--", alpha=0.6, label="UTC (as delivered)")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Average NYISO demand, MW")
        ax.set_title("Daily load shape: local vs UTC clock")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        chart_path.parent.mkdir(exist_ok=True)
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
    return result


def build_hourly(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    dem_raw = read_raw(data_dir / "nyiso_demand_raw.csv")
    full = pd.date_range(dem_raw.index.min(), dem_raw.index.max(), freq="h", tz="UTC")
    demand, rep_d = clean_series(dem_raw, full)

    fc_path = data_dir / "nyiso_forecast_raw.csv"
    if fc_path.exists():
        # the operator forecast is a benchmark: repair nothing beyond impossible values
        fc, rep_f = clean_series(read_raw(fc_path), full, max_gap=0, spike_tol=10.0)
    else:
        fc, rep_f = pd.Series(np.nan, index=full), {"note": "no forecast file"}

    temp, rep_t = clean_series(read_raw(data_dir / "nyc_temp_raw.csv"), full, max_gap=3,
                               spike_tol=10.0, positive_only=False)
    df = pd.DataFrame({"demand": demand, "op_forecast": fc, "temp_f": temp})
    df.index.name = "timestamp_utc"
    report = {"demand": rep_d, "operator_forecast": rep_f, "temperature": rep_t,
              "first_hour_utc": str(full[0]), "last_hour_utc": str(full[-1]),
              "operator_forecast_coverage": float(df["op_forecast"].notna().mean())}
    return df, report


def main() -> None:
    df, report = build_hourly()
    report["timezone_check"] = check_daily_shape(df["demand"].dropna(), CHART_DIR / "daily_shape.png")
    df.to_csv(DATA_DIR / "hourly.csv")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["timezone_check"]["shape_ok"]:
        print("\nWARNING: daily load shape looks wrong. Check the timestamp handling before modeling.")
    if report["operator_forecast_coverage"] < 0.8:
        print("\nNOTE: the operator forecast (type DF) is sparse for NYIS; it will be skipped as a benchmark.")


if __name__ == "__main__":
    main()
