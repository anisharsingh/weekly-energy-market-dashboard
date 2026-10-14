"""Feature engineering for the day-ahead load forecast.

FORECAST DEFINITION
    Made at 10:00 local time on day D-1 for all 24 hours of day D.
    At that moment the latest observed demand is the hour ending 09:00 on D-1, so the farthest
    target hour (23:00 on D) is 38 hours ahead and the nearest (00:00 on D) is 15 hours ahead.
    Rule that guarantees no leakage: every demand-based feature is lagged by AT LEAST 48 hours.
    Same-hour-yesterday (24 h) is deliberately NOT used.

    Temperature enters as the ACTUAL temperature at the target hour. That assumes a perfect weather
    forecast, so the results are an upper bound on what a real day-ahead system would achieve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TZ = "America/New_York"
MIN_LAG_HOURS = 48
CAT_FEATURES = ["hour", "dow", "month"]
NUM_FEATURES = ["temp_f", "temp_sq", "hdd", "cdd", "is_weekend", "is_holiday",
                "lag_48", "lag_168", "roll_mean_48_24"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: regular hourly UTC index with columns demand, temp_f (and optionally op_forecast).
    Returns the frame with calendar, weather and lagged-demand features added."""
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted")
    steps = df.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(hours=1):
        raise ValueError("index must be a regular hourly grid so row lags equal hour lags")

    out = df.copy()
    local = out.index.tz_convert(TZ)
    out["hour"] = local.hour
    out["dow"] = local.dayofweek
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    local_date = pd.DatetimeIndex(local.tz_localize(None).normalize())
    holidays = USFederalHolidayCalendar().holidays(start=local_date.min(), end=local_date.max())
    out["is_holiday"] = local_date.isin(holidays).astype(int)

    out["temp_sq"] = out["temp_f"] ** 2
    out["hdd"] = (65.0 - out["temp_f"]).clip(lower=0)  # heating degrees, 65 F base
    out["cdd"] = (out["temp_f"] - 65.0).clip(lower=0)  # cooling degrees, 65 F base

    out["lag_48"] = out["demand"].shift(48)
    out["lag_168"] = out["demand"].shift(168)  # same hour last week
    out["roll_mean_48_24"] = out["demand"].shift(MIN_LAG_HOURS).rolling(24, min_periods=24).mean()
    return out


def modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a target, all features, and (for baselines) a same-hour-last-week value."""
    feats = build_features(df)
    return feats.dropna(subset=["demand"] + ALL_FEATURES)
