"""Tests for the load-forecast pipeline. Run from the repo root:  pytest forecasting/tests
No network or real data needed: they use a small synthetic hourly series."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clean import clean_series, check_daily_shape  # noqa: E402
from features import ALL_FEATURES, build_features, modeling_frame  # noqa: E402
from validation import make_folds, mae, mape, rmse, skill_score  # noqa: E402


@pytest.fixture(scope="module")
def hourly():
    idx = pd.date_range("2022-01-01", periods=24 * 800, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    local_hr = idx.tz_convert("America/New_York").hour.to_numpy()
    demand = 17000 + 2500 * np.exp(-((local_hr - 17) / 5) ** 2) + rng.normal(0, 200, len(idx))
    temp = 55 + 20 * np.sin(np.arange(len(idx)) / (24 * 365) * 2 * np.pi) + rng.normal(0, 3, len(idx))
    return pd.DataFrame({"demand": demand, "temp_f": temp}, index=idx)


def test_features_do_not_use_recent_demand(hourly):
    """Changing demand in the 47 hours before t must not change any feature at t (lags >= 48 h)."""
    t = 2000
    base = build_features(hourly)
    perturbed = hourly.copy()
    perturbed.iloc[t - 47: t + 1, perturbed.columns.get_loc("demand")] += 99999.0
    alt = build_features(perturbed)
    cols = ["lag_48", "lag_168", "roll_mean_48_24"]
    pd.testing.assert_series_equal(base.iloc[t][cols], alt.iloc[t][cols])


def test_lags_are_in_elapsed_hours(hourly):
    f = build_features(hourly)
    assert f["lag_168"].iloc[500] == hourly["demand"].iloc[500 - 168]
    assert f["lag_48"].iloc[500] == hourly["demand"].iloc[500 - 48]


def test_irregular_index_is_rejected(hourly):
    with pytest.raises(ValueError):
        build_features(hourly.drop(hourly.index[100]))


def test_local_hour_handles_daylight_saving():
    idx = pd.DatetimeIndex(["2024-01-01 05:00", "2024-07-01 04:00"], tz="UTC")
    df = pd.DataFrame({"demand": [1.0, 1.0], "temp_f": [50.0, 80.0]}, index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))
    local = idx.tz_convert("America/New_York")
    assert list(local.hour) == [0, 0]  # midnight local in both winter (UTC-5) and summer (UTC-4)


def test_folds_never_train_on_the_future(hourly):
    folds, holdout = make_folds(hourly.index, holdout_months=3, min_train_months=12)
    for _, train, test in folds + [holdout]:
        assert train.any() and test.any()
        assert hourly.index[train].max() < hourly.index[test].min()
        assert not (train & test).any()


def test_holdout_is_the_last_three_months(hourly):
    _, (_, train, test) = make_folds(hourly.index, holdout_months=3, min_train_months=12)
    assert hourly.index[test].max() == hourly.index.max()
    assert hourly.index[train].max() < hourly.index[test].min()


def test_metric_formulas():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert mae(y, p) == 15.0
    assert rmse(y, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(y, p) == pytest.approx((10 / 100 + 20 / 200) / 2 * 100)
    assert skill_score(8.0, 10.0) == pytest.approx(0.2)


def test_cleaning_repairs_short_gaps_and_flags_a_spike():
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    raw = pd.Series(1000.0 + 50 * np.sin(np.arange(200) / 5), index=idx)
    raw.iloc[50:52] = np.nan   # 2-hour gap: should be interpolated
    raw.iloc[100] = 5000.0     # spike: should be removed, then interpolated
    raw.iloc[150] = 0.0        # impossible value
    clean, report = clean_series(raw, idx)
    assert report["spikes"] == 1 and report["nonpositive_values"] == 1
    assert clean.isna().sum() == 0
    assert abs(clean.iloc[100] - 1000.0) < 200


def test_daily_shape_check_detects_wrong_timezone():
    idx = pd.date_range("2023-01-01", periods=24 * 60, freq="h", tz="UTC")
    local_hr = idx.tz_convert("America/New_York").hour.to_numpy()
    good = pd.Series(15000 + 3000 * np.exp(-((local_hr - 17) / 4) ** 2) - 2500 * np.exp(-((local_hr - 4) / 3) ** 2), index=idx)
    assert check_daily_shape(good)["shape_ok"]
    shifted = pd.Series(good.to_numpy(), index=idx + pd.Timedelta(hours=5))  # as if mislabeled
    assert not check_daily_shape(shifted)["shape_ok"]


def test_modeling_frame_has_all_features(hourly):
    f = modeling_frame(hourly)
    assert set(ALL_FEATURES).issubset(f.columns)
    assert not f[ALL_FEATURES].isna().any().any()
