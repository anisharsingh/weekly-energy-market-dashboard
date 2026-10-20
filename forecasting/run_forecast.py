#!/usr/bin/env python3
"""Day-ahead NYISO load forecast: baselines, two models, backtest, error analysis, intervals.

    python fetch_data.py     # once (needs EIA_API_KEY)
    python clean.py          # builds data/hourly.csv and the quality report
    python run_forecast.py   # everything below; writes results/ and charts/
    python run_forecast.py --quick   # fewer boosting iterations, for a fast smoke test

Models are re-trained inside every fold on past data only. Random seeds are fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import ALL_FEATURES, CAT_FEATURES, NUM_FEATURES, TZ, modeling_frame
from validation import make_folds, metric_row, skill_score

HERE = Path(__file__).resolve().parent
DATA_DIR, RESULTS, CHARTS = HERE / "data", HERE / "results", HERE / "charts"
SEED = 0
HEAT_WAVE_F = 90.0   # daily high at or above this counts as a heat-wave day
COLD_SNAP_F = 20.0   # daily low at or below this counts as a cold-snap day
MODELS = ["baseline_168h", "operator_DF", "ridge", "hgb"]


# ---------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------
def make_ridge() -> object:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),   # hour, day of week, month dummies
        ("num", StandardScaler(), NUM_FEATURES),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def make_hgb(quantile: float | None = None, quick: bool = False) -> HistGradientBoostingRegressor:
    kw = dict(
        max_iter=60 if quick else 300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
        early_stopping=False, random_state=SEED,
        categorical_features=[c in CAT_FEATURES for c in ALL_FEATURES],
    )
    if quantile is None:
        return HistGradientBoostingRegressor(loss="squared_error", **kw)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)


# ---------------------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------------------
def backtest(frame: pd.DataFrame, folds, holdout, quick: bool):
    """Fit every model on the training rows of each fold and predict that fold's test rows."""
    parts, hold_models = [], {}
    for name, tr, te in folds + [holdout]:
        Xtr, ytr = frame.loc[tr, ALL_FEATURES], frame.loc[tr, "demand"]
        Xte = frame.loc[te, ALL_FEATURES]
        ridge = make_ridge().fit(Xtr, ytr)
        hgb = make_hgb(quick=quick).fit(Xtr, ytr)
        part = frame.loc[te].copy()
        part["baseline_168h"] = part["lag_168"]
        part["operator_DF"] = part["op_forecast"]
        part["ridge"] = ridge.predict(Xte)
        part["hgb"] = hgb.predict(Xte)
        part["fold"] = name
        part["is_holdout"] = name == "holdout"
        parts.append(part)
        if name == "holdout":
            hold_models = {"hgb": hgb, "train_mask": tr, "test_mask": te}
        print(f"  fold {name}: train {tr.sum():,} rows, test {te.sum():,} rows")
    return pd.concat(parts), hold_models


def evaluate(oof: pd.DataFrame, models: list[str]):
    """Per-fold metrics, fold averages, hold-out metrics and skill scores."""
    top_threshold = oof["demand"].quantile(0.95)  # 'top 5% load hours' of everything evaluated
    rows = []
    for fold, g in oof.groupby("fold", sort=False):
        top = (g["demand"] >= top_threshold).to_numpy()
        for m in models:
            rows.append({"fold": fold, "model": m, "n": len(g), **metric_row(g["demand"], g[m], top)})
    fold_metrics = pd.DataFrame(rows)

    back = fold_metrics[fold_metrics["fold"] != "holdout"].groupby("model").mean(numeric_only=True)
    hold = fold_metrics[fold_metrics["fold"] == "holdout"].set_index("model")
    table = pd.DataFrame(index=models)
    for col in ["MAE", "MAPE_%", "RMSE", "MAE_top5%_hours"]:
        table[f"backtest_{col}"] = back[col]
        table[f"holdout_{col}"] = hold[col]
    for base, label in [("baseline_168h", "168h"), ("operator_DF", "operator")]:
        if base in models:
            table[f"skill_vs_{label}_backtest"] = [skill_score(table.loc[m, "backtest_MAE"], table.loc[base, "backtest_MAE"]) for m in models]
            table[f"skill_vs_{label}_holdout"] = [skill_score(table.loc[m, "holdout_MAE"], table.loc[base, "holdout_MAE"]) for m in models]
    return fold_metrics, table


# ---------------------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------------------
def group_mae(oof: pd.DataFrame, key: pd.Series, models: list[str], label: str) -> pd.DataFrame:
    out = pd.DataFrame({m: (oof["demand"] - oof[m]).abs().groupby(key).mean() for m in models})
    out.insert(0, "n_hours", oof.groupby(key).size())
    out.index = [f"{label}: {i}" for i in out.index]
    return out


def error_analysis(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    daytype = pd.Series(np.where(oof["is_holiday"] == 1, "holiday", np.where(oof["is_weekend"] == 1, "weekend", "weekday")),
                        index=oof.index)
    heat = pd.Series(np.where(oof["daily_tmax"] >= HEAT_WAVE_F, "heat-wave day", "other"), index=oof.index)
    cold = pd.Series(np.where(oof["daily_tmin"] <= COLD_SNAP_F, "cold-snap day", "other"), index=oof.index)
    return pd.concat([
        group_mae(oof, oof["hour"], models, "local hour"),
        group_mae(oof, daytype, models, "day type"),
        group_mae(oof, heat, models, f"daily high >= {HEAT_WAVE_F:.0f}F"),
        group_mae(oof, cold, models, f"daily low <= {COLD_SNAP_F:.0f}F"),
    ])


def importance(hold: dict, frame: pd.DataFrame) -> pd.DataFrame:
    te = hold["test_mask"]
    X, y = frame.loc[te, ALL_FEATURES], frame.loc[te, "demand"]
    res = permutation_importance(hold["hgb"], X, y, n_repeats=5, random_state=SEED,
                                 scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": ALL_FEATURES, "mae_increase_MW": res.importances_mean,
                          "std": res.importances_std}).sort_values("mae_increase_MW", ascending=False))


def intervals(frame: pd.DataFrame, hold: dict, quick: bool, q_lo: float = 0.10, q_hi: float = 0.90):
    """Quantile gradient-boosting models; check that the 80% interval covers about 80% of actuals."""
    tr, te = hold["train_mask"], hold["test_mask"]
    Xtr, ytr, Xte = frame.loc[tr, ALL_FEATURES], frame.loc[tr, "demand"], frame.loc[te, ALL_FEATURES]
    lo = make_hgb(q_lo, quick).fit(Xtr, ytr).predict(Xte)
    hi = make_hgb(q_hi, quick).fit(Xtr, ytr).predict(Xte)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)  # guard against quantile crossing
    return pd.DataFrame({"q10": lo, "q90": hi}, index=frame.index[te])


# ---------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------
def plot_error_by_hour(oof: pd.DataFrame, models: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in models:
        ax.plot(range(24), (oof["demand"] - oof[m]).abs().groupby(oof["hour"]).mean().reindex(range(24)), marker="o", label=m)
    ax.set_xlabel("Local hour of day")
    ax.set_ylabel("MAE, MW")
    ax.set_title("Forecast error by hour of day (all backtest folds + hold-out)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heat_wave_week(oof: pd.DataFrame, models: list[str], path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hold = oof[oof["is_holdout"]]
    pool = hold if hold["daily_tmax"].max() >= HEAT_WAVE_F else oof  # prefer the hold-out so the band can be shown
    local_date = pool.index.tz_convert(TZ).tz_localize(None).normalize()
    hot_day = pool["daily_tmax"].groupby(local_date).max().idxmax()
    start = hot_day - pd.Timedelta(days=3)
    end = hot_day + pd.Timedelta(days=4)
    all_dates = oof.index.tz_convert(TZ).tz_localize(None).normalize()
    w = oof[(all_dates >= start) & (all_dates < end)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(w.index, w["demand"], color="black", lw=2, label="Actual")
    for m, style in [("hgb", "-"), ("baseline_168h", ":"), ("operator_DF", "--")]:
        if m in models:
            ax.plot(w.index, w[m], style, lw=1.4, label=m)
    if "q10" in w and w["q10"].notna().any():
        ax.fill_between(w.index, w["q10"], w["q90"], alpha=0.2, label="hgb 80% interval")
    ax.set_ylabel("NYISO demand, MW")
    ax.set_title(f"Actual vs forecast around the hottest day ({hot_day.date()}, high {pool['daily_tmax'].max():.0f}F)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(hot_day.date())


# ---------------------------------------------------------------------------------------
def main(quick: bool = False, data_dir: Path = DATA_DIR) -> dict:
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    CHARTS.mkdir(exist_ok=True)

    raw = pd.read_csv(data_dir / "hourly.csv", index_col="timestamp_utc", parse_dates=True)
    raw.index = pd.to_datetime(raw.index, utc=True)
    frame = modeling_frame(raw)
    local_date = frame.index.tz_convert(TZ).tz_localize(None).normalize()
    frame["daily_tmax"] = frame.groupby(local_date)["temp_f"].transform("max")
    frame["daily_tmin"] = frame.groupby(local_date)["temp_f"].transform("min")
    print(f"Modeling rows: {len(frame):,}  ({frame.index.min()} to {frame.index.max()})")

    folds, holdout = make_folds(frame.index, holdout_months=3, min_train_months=12)
    print(f"{len(folds)} monthly folds + 1 hold-out of the last 3 months")
    oof, hold = backtest(frame, folds, holdout, quick)

    # Benchmark against the operator's own forecast only if it is well populated
    df_cov = float(oof["op_forecast"].notna().mean())
    models = list(MODELS)
    if df_cov >= 0.8:
        oof = oof[oof["op_forecast"].notna()]
        print(f"Operator forecast available for {df_cov:.0%} of rows; comparing all models on those rows.")
    else:
        models.remove("operator_DF")
        print(f"Operator forecast available for only {df_cov:.0%} of rows; skipping that benchmark.")

    fold_metrics, table = evaluate(oof, models)
    print("\nResults (MW unless noted):")
    print(table.round(3).T.to_string())

    errors = error_analysis(oof, models)
    print("\nError analysis (MAE, MW):")
    print(errors.round(1).to_string())

    imp = importance(hold, frame)
    print("\nPermutation importance (increase in hold-out MAE, MW):")
    print(imp.round(1).to_string(index=False))

    band = intervals(frame, hold, quick)
    oof = oof.join(band, how="left")
    h = oof[oof["is_holdout"]]
    inside = (h["demand"] >= h["q10"]) & (h["demand"] <= h["q90"])
    heat_h = h["daily_tmax"] >= HEAT_WAVE_F
    cov = {"nominal": 0.80, "holdout_coverage": float(inside.mean()), "mean_width_MW": float((h["q90"] - h["q10"]).mean()),
           "heat_wave_hours_in_holdout": int(heat_h.sum()),
           "heat_wave_coverage": float(inside[heat_h].mean()) if heat_h.any() else None}
    print("\nPrediction interval check:", json.dumps(cov, indent=2))

    plot_error_by_hour(oof, models, CHARTS / "error_by_hour.png")
    hot_day = plot_heat_wave_week(oof, models, CHARTS / "heatwave_week.png")

    table.to_csv(RESULTS / "metrics.csv")
    fold_metrics.to_csv(RESULTS / "fold_metrics.csv", index=False)
    errors.to_csv(RESULTS / "error_analysis.csv")
    imp.to_csv(RESULTS / "permutation_importance.csv", index=False)
    oof.to_csv(RESULTS / "oof_predictions.csv")   # large; ignored by git
    (RESULTS / "interval_coverage.json").write_text(json.dumps(cov, indent=2))
    print(f"\nSaved results to {RESULTS} and charts to {CHARTS} (heat-wave chart centered on {hot_day}).")
    return {"table": table, "errors": errors, "importance": imp, "coverage": cov}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer boosting iterations (smoke test)")
    main(quick=ap.parse_args().quick)
