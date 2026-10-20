"""Time-based validation: expanding-window monthly folds plus a final hold-out, and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/New_York"


def local_month(index: pd.DatetimeIndex) -> pd.PeriodIndex:
    return index.tz_convert(TZ).tz_localize(None).to_period("M")


def make_folds(index: pd.DatetimeIndex, holdout_months: int = 3, min_train_months: int = 12):
    """Return (folds, holdout). Each fold is (name, train_mask, test_mask) over `index`.

    Fold k trains on every row before month m and tests on month m (one month of test data), for each
    month m from the first month after `min_train_months` of history up to the month before the
    hold-out. The hold-out trains on everything before its first month and tests on the last
    `holdout_months` months. Nothing is ever shuffled.
    """
    ym = local_month(index)
    months = ym.unique().sort_values()
    if len(months) < min_train_months + holdout_months + 1:
        raise ValueError("not enough history for the requested folds")
    hold_first = months[-holdout_months]
    folds = []
    for m in months[min_train_months:-holdout_months]:
        folds.append((str(m), np.asarray(ym < m), np.asarray(ym == m)))
    holdout = ("holdout", np.asarray(ym < hold_first), np.asarray(ym >= hold_first))
    return folds, holdout


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p) / np.abs(y)) * 100)


def metric_row(y, p, top_mask=None) -> dict:
    row = {"MAE": mae(y, p), "MAPE_%": mape(y, p), "RMSE": rmse(y, p)}
    if top_mask is not None and np.asarray(top_mask).any():
        row["MAE_top5%_hours"] = mae(np.asarray(y)[top_mask], np.asarray(p)[top_mask])
    return row


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """1 - MAE_model / MAE_baseline. Positive means better than the baseline; 0 means equal."""
    return 1.0 - model_mae / baseline_mae
