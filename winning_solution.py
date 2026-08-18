#!/usr/bin/env python3
"""Leakage-safe solution for the Zindi global TWS forecasting challenge.

The important detail in this competition is that a masked test row is not a normal
missing-value row.  It belongs to a forecast rollout that starts at the most recent
mostly-observed TWS map.  This script learns direct 1..7 month forecasts from that
legal anchor and blends them with an exogenous (TWS-free) climate model.

Commands
--------
python winning_solution.py validate --data-dir .
python winning_solution.py train --data-dir . --artifact-dir artifacts
python winning_solution.py predict --data-dir . --artifact-dir artifacts \
    --output submission.csv
python winning_solution.py all --data-dir . --artifact-dir artifacts \
    --output submission.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


SEED = 20260817
CLIMATE = [
    "SPEI_01_t",
    "SPEI_03_t",
    "SPEI_06_t",
    "SPEI_12_t",
    "SOIL_MOISTURE_t",
]
DIRECT_FEATURES = [
    "anchor_tws",
    "anchor_missing",
    *[f"current_{c}" for c in CLIMATE],
    *[f"anchor_{c}" for c in CLIMATE],
    *[f"change_{c}" for c in CLIMATE],
    "month_sin",
    "month_cos",
    "target_month_sin",
    "target_month_cos",
    "horizon",
    "lat_scaled",
    "lon_scaled",
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos",
    "lon2_sin",
    "lon2_cos",
    "loc_mean",
    "loc_std",
    "loc_climatology",
    "loc_trend",
    "trend_minus_climatology",
    "anchor_minus_climatology",
    "anchor_minus_trend",
    "soil_minus_spei01",
    "spei01_minus_spei03",
    "spei03_minus_spei06",
    "spei06_minus_spei12",
]
EXOG_FEATURES = [
    *[f"current_{c}" for c in CLIMATE],
    "month_sin",
    "month_cos",
    "target_month_sin",
    "target_month_cos",
    "lat_scaled",
    "lon_scaled",
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos",
    "lon2_sin",
    "lon2_cos",
    "loc_mean",
    "loc_std",
    "loc_climatology",
    "loc_trend",
    "trend_minus_climatology",
    "soil_minus_spei01",
    "spei01_minus_spei03",
    "spei03_minus_spei06",
    "spei06_minus_spei12",
]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def month_number(values: pd.Series) -> np.ndarray:
    return (values.dt.year.to_numpy(np.int32) * 12 + values.dt.month.to_numpy(np.int32) - 1)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable integer space/time keys without relying on row order."""
    out = df.copy()
    out["month_idx"] = month_number(out["time"])
    lat_idx = np.rint(out["lat"].to_numpy() + 55.5).astype(np.int32)
    lon_idx = np.rint(out["lon"].to_numpy() + 179.5).astype(np.int32)
    out["loc_key"] = lat_idx * 360 + lon_idx
    return out


def locate_file(data_dir: Path, stem: str) -> Path:
    exact = data_dir / f"{stem}.csv"
    if exact.exists():
        return exact
    matches = sorted(data_dir.glob(f"{stem}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {stem}*.csv in {data_dir}, found {[p.name for p in matches]}"
        )
    return matches[0]


def load_data(data_dir: Path, need_test: bool = True) -> tuple[pd.DataFrame, pd.DataFrame | None, Path | None]:
    train_path = locate_file(data_dir, "Train")
    log(f"Reading {train_path.name}")
    train = pd.read_csv(train_path, parse_dates=["time"])
    train = train.rename(columns={"sample_id": "ID", "target": "Target"})
    required_train = {"ID", "time", "lat", "lon", "TWS_t", "Target", *CLIMATE}
    missing = required_train - set(train.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    train = add_keys(train)
    for c in ["lat", "lon", "TWS_t", "Target", "month_sin", "month_cos", *CLIMATE]:
        train[c] = train[c].astype(np.float32)

    if not need_test:
        return train, None, None

    test_path = locate_file(data_dir, "Test")
    sample_path = locate_file(data_dir, "SampleSubmission")
    log(f"Reading {test_path.name}")
    test = add_keys(pd.read_csv(test_path, parse_dates=["time"]))
    required_test = {"ID", "time", "lat", "lon", "TWS_t", "TWS_t_masked", *CLIMATE}
    missing = required_test - set(test.columns)
    if missing:
        raise ValueError(f"Test data is missing columns: {sorted(missing)}")
    for c in ["lat", "lon", "TWS_t", "month_sin", "month_cos", *CLIMATE]:
        test[c] = test[c].astype(np.float32)
    return train, test, sample_path


@dataclass
class HistoryStats:
    max_loc_key: int
    overall_mean: float
    overall_std: float
    reference_month: float
    loc_mean: np.ndarray
    loc_std: np.ndarray
    loc_mean_month: np.ndarray
    loc_slope: np.ndarray
    loc_month_mean: np.ndarray
    loc_month_count: np.ndarray
    global_month_mean: np.ndarray
    target_min: float
    target_max: float


def fit_history_stats(history: pd.DataFrame) -> HistoryStats:
    """Target-free historical summaries derived only from observed TWS_t."""
    n_loc = int(history["loc_key"].max()) + 1
    overall_mean = float(history["TWS_t"].mean())
    overall_std = float(history["TWS_t"].std())

    loc_mean = np.full(n_loc, overall_mean, np.float32)
    loc_std = np.full(n_loc, overall_std, np.float32)
    loc_mean_month = np.full(n_loc, float(history["month_idx"].mean()), np.float32)
    loc_slope = np.zeros(n_loc, np.float32)

    g = history.groupby("loc_key", sort=False).agg(
        y_mean=("TWS_t", "mean"),
        y_std=("TWS_t", "std"),
        x_mean=("month_idx", "mean"),
        n=("TWS_t", "size"),
    )
    keys = g.index.to_numpy(np.int32)
    loc_mean[keys] = g["y_mean"].to_numpy(np.float32)
    loc_std[keys] = g["y_std"].fillna(overall_std).to_numpy(np.float32)
    loc_mean_month[keys] = g["x_mean"].to_numpy(np.float32)

    # Per-location linear trend.  Clipping prevents rare short-history cells from
    # producing implausible multi-year extrapolations.
    temp = history[["loc_key", "month_idx", "TWS_t"]].copy()
    temp["dx"] = temp["month_idx"].to_numpy(np.float32) - loc_mean_month[temp["loc_key"]]
    temp["cross"] = temp["dx"] * (temp["TWS_t"].to_numpy(np.float32) - loc_mean[temp["loc_key"]])
    temp["square"] = temp["dx"] * temp["dx"]
    sums = temp.groupby("loc_key", sort=False)[["cross", "square"]].sum()
    denom = sums["square"].to_numpy(np.float64)
    slope = np.divide(
        sums["cross"].to_numpy(np.float64),
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-6,
    )
    loc_slope[sums.index.to_numpy(np.int32)] = np.clip(slope, -0.035, 0.035).astype(np.float32)
    del temp

    hist_month = history["time"].dt.month.to_numpy(np.int8) - 1
    loc_month_index = history["loc_key"].to_numpy(np.int32) * 12 + hist_month
    n_lm = n_loc * 12
    counts = np.bincount(loc_month_index, minlength=n_lm).astype(np.int16)
    sums_lm = np.bincount(
        loc_month_index,
        weights=history["TWS_t"].to_numpy(np.float64),
        minlength=n_lm,
    )
    raw_lm = np.divide(
        sums_lm,
        counts,
        out=np.zeros_like(sums_lm),
        where=counts > 0,
    ).astype(np.float32)

    global_month = (
        history.assign(_month=hist_month)
        .groupby("_month")["TWS_t"]
        .mean()
        .reindex(range(12), fill_value=overall_mean)
        .to_numpy(np.float32)
    )

    return HistoryStats(
        max_loc_key=n_loc - 1,
        overall_mean=overall_mean,
        overall_std=overall_std,
        reference_month=float(history["month_idx"].mean()),
        loc_mean=loc_mean,
        loc_std=loc_std,
        loc_mean_month=loc_mean_month,
        loc_slope=loc_slope,
        loc_month_mean=raw_lm,
        loc_month_count=counts,
        global_month_mean=global_month,
        target_min=float(history["Target"].min()),
        target_max=float(history["Target"].max()),
    )


def historical_features(current: pd.DataFrame, stats: HistoryStats) -> dict[str, np.ndarray]:
    keys = current["loc_key"].to_numpy(np.int32)
    valid_key = keys <= stats.max_loc_key
    safe_keys = np.where(valid_key, keys, 0)
    target_idx = current["month_idx"].to_numpy(np.int32) + 1
    target_month = target_idx % 12

    loc_mean = np.where(valid_key, stats.loc_mean[safe_keys], stats.overall_mean).astype(np.float32)
    loc_std = np.where(valid_key, stats.loc_std[safe_keys], stats.overall_std).astype(np.float32)
    slope = np.where(valid_key, stats.loc_slope[safe_keys], 0.0).astype(np.float32)
    mean_month = np.where(valid_key, stats.loc_mean_month[safe_keys], stats.reference_month)
    trend = (loc_mean + slope * (target_idx - mean_month)).astype(np.float32)

    lm_idx = safe_keys * 12 + target_month
    lm_count = np.where(valid_key, stats.loc_month_count[lm_idx], 0).astype(np.float32)
    lm_raw = np.where(valid_key & (lm_count > 0), stats.loc_month_mean[lm_idx], loc_mean)
    # Empirical Bayes shrinkage is important because there are only ~10 examples
    # for a location/calendar-month combination.
    alpha = np.float32(3.0)
    climatology = ((lm_count * lm_raw + alpha * loc_mean) / (lm_count + alpha)).astype(np.float32)
    no_history = ~valid_key
    if no_history.any():
        climatology[no_history] = stats.global_month_mean[target_month[no_history]]

    angle = 2.0 * np.pi * target_month.astype(np.float32) / 12.0
    return {
        "loc_mean": loc_mean,
        "loc_std": loc_std,
        "loc_climatology": climatology,
        "loc_trend": trend,
        "target_month_sin": np.sin(angle).astype(np.float32),
        "target_month_cos": np.cos(angle).astype(np.float32),
    }


def build_feature_frame(
    current: pd.DataFrame,
    stats: HistoryStats,
    anchor: pd.DataFrame | None = None,
    horizon: int | np.ndarray = 1,
) -> pd.DataFrame:
    n = len(current)
    hist = historical_features(current, stats)
    lat_rad = np.deg2rad(current["lat"].to_numpy(np.float32))
    lon_rad = np.deg2rad(current["lon"].to_numpy(np.float32))
    result: dict[str, np.ndarray] = {}

    for c in CLIMATE:
        result[f"current_{c}"] = current[c].to_numpy(np.float32)

    result.update(
        {
            "month_sin": current["month_sin"].to_numpy(np.float32),
            "month_cos": current["month_cos"].to_numpy(np.float32),
            "target_month_sin": hist["target_month_sin"],
            "target_month_cos": hist["target_month_cos"],
            "lat_scaled": current["lat"].to_numpy(np.float32) / 90.0,
            "lon_scaled": current["lon"].to_numpy(np.float32) / 180.0,
            "lat_sin": np.sin(lat_rad).astype(np.float32),
            "lat_cos": np.cos(lat_rad).astype(np.float32),
            "lon_sin": np.sin(lon_rad).astype(np.float32),
            "lon_cos": np.cos(lon_rad).astype(np.float32),
            "lon2_sin": np.sin(2.0 * lon_rad).astype(np.float32),
            "lon2_cos": np.cos(2.0 * lon_rad).astype(np.float32),
            "loc_mean": hist["loc_mean"],
            "loc_std": hist["loc_std"],
            "loc_climatology": hist["loc_climatology"],
            "loc_trend": hist["loc_trend"],
            "trend_minus_climatology": hist["loc_trend"] - hist["loc_climatology"],
            "soil_minus_spei01": current["SOIL_MOISTURE_t"].to_numpy(np.float32)
            - current["SPEI_01_t"].to_numpy(np.float32),
            "spei01_minus_spei03": current["SPEI_01_t"].to_numpy(np.float32)
            - current["SPEI_03_t"].to_numpy(np.float32),
            "spei03_minus_spei06": current["SPEI_03_t"].to_numpy(np.float32)
            - current["SPEI_06_t"].to_numpy(np.float32),
            "spei06_minus_spei12": current["SPEI_06_t"].to_numpy(np.float32)
            - current["SPEI_12_t"].to_numpy(np.float32),
        }
    )

    if anchor is not None:
        anchor_tws_raw = anchor["TWS_t"].to_numpy(np.float32)
        missing_anchor = ~np.isfinite(anchor_tws_raw)
        anchor_tws = np.where(missing_anchor, hist["loc_trend"], anchor_tws_raw).astype(np.float32)
        result["anchor_tws"] = anchor_tws
        result["anchor_missing"] = missing_anchor.astype(np.float32)
        for c in CLIMATE:
            anchor_raw = anchor[c].to_numpy(np.float32)
            missing = ~np.isfinite(anchor_raw)
            anchor_value = np.where(missing, result[f"current_{c}"], anchor_raw).astype(np.float32)
            result[f"anchor_{c}"] = anchor_value
            result[f"change_{c}"] = result[f"current_{c}"] - anchor_value
        result["horizon"] = np.broadcast_to(np.asarray(horizon, dtype=np.float32), n).copy()
        result["anchor_minus_climatology"] = anchor_tws - hist["loc_climatology"]
        result["anchor_minus_trend"] = anchor_tws - hist["loc_trend"]

    return pd.DataFrame(result, index=current.index)


def make_pair_frame(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Match current climate at t with the legal TWS anchor at t-h+1."""
    current_cols = [
        "ID", "time", "month_idx", "loc_key", "lat", "lon", "month_sin", "month_cos",
        "TWS_t", "Target", *CLIMATE,
    ]
    current = data[current_cols]
    if horizon == 1:
        out = current.copy()
        out["anchor_TWS_t"] = out["TWS_t"]
        for c in CLIMATE:
            out[f"anchor_{c}"] = out[c]
        return out

    anchor = data[["month_idx", "loc_key", "TWS_t", *CLIMATE]].copy()
    anchor["month_idx"] = anchor["month_idx"] + horizon - 1
    anchor = anchor.rename(
        columns={"TWS_t": "anchor_TWS_t", **{c: f"anchor_{c}" for c in CLIMATE}}
    )
    return current.merge(anchor, on=["month_idx", "loc_key"], how="inner", validate="one_to_one")


def pair_to_xy(pair: pd.DataFrame, stats: HistoryStats, horizon: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    anchor = pd.DataFrame(index=pair.index)
    anchor["TWS_t"] = pair["anchor_TWS_t"].to_numpy(np.float32)
    for c in CLIMATE:
        anchor[c] = pair[f"anchor_{c}"].to_numpy(np.float32)
    x = build_feature_frame(pair, stats, anchor=anchor, horizon=horizon)[DIRECT_FEATURES]
    target = pair["Target"].to_numpy(np.float32)
    residual = target - x["anchor_tws"].to_numpy(np.float32)
    return x, residual, target


def deterministic_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed, ignore_index=True)


@dataclass
class ModelConfig:
    max_horizon: int = 7
    rows_per_horizon: int = 320_000
    exog_rows: int = 1_500_000
    direct_estimators: int = 750
    exog_estimators: int = 700
    learning_rate: float = 0.035
    num_leaves: int = 64
    min_child_samples: int = 250
    n_jobs: int = -1


def lgb_params(config: ModelConfig, n_estimators: int) -> dict:
    return dict(
        objective="regression",
        metric="rmse",
        n_estimators=n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=-1,
        min_child_samples=config.min_child_samples,
        max_bin=255,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.86,
        reg_alpha=0.08,
        reg_lambda=2.5,
        random_state=SEED,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )


def build_direct_training(
    train: pd.DataFrame,
    stats: HistoryStats,
    config: ModelConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    xs: list[pd.DataFrame] = []
    ys: list[np.ndarray] = []
    for h in range(1, config.max_horizon + 1):
        log(f"Building horizon {h} direct examples")
        pair = make_pair_frame(train, h)
        pair = deterministic_sample(pair, config.rows_per_horizon, SEED + h)
        x, y_residual, _ = pair_to_xy(pair, stats, h)
        xs.append(x)
        ys.append(y_residual)
        del pair
        gc.collect()
    x_all = pd.concat(xs, ignore_index=True)
    y_all = np.concatenate(ys)
    return x_all, y_all


def build_exog_training(
    train: pd.DataFrame,
    stats: HistoryStats,
    config: ModelConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    sample = deterministic_sample(train, config.exog_rows, SEED + 99)
    x = build_feature_frame(sample, stats)[EXOG_FEATURES]
    climatology = x["loc_climatology"].to_numpy(np.float32)
    y = sample["Target"].to_numpy(np.float32) - climatology
    return x, y


def fit_models(
    train: pd.DataFrame,
    stats: HistoryStats,
    config: ModelConfig,
    direct_eval: tuple[pd.DataFrame, np.ndarray] | None = None,
    exog_eval: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
    x_direct, y_direct = build_direct_training(train, stats, config)
    log(f"Training direct model on {len(x_direct):,} examples")
    direct = lgb.LGBMRegressor(**lgb_params(config, config.direct_estimators))
    direct_fit: dict = {}
    if direct_eval is not None:
        direct_fit = {
            "eval_set": [direct_eval],
            "callbacks": [lgb.early_stopping(80, verbose=True), lgb.log_evaluation(100)],
        }
    direct.fit(x_direct, y_direct, **direct_fit)
    del x_direct, y_direct
    gc.collect()

    x_exog, y_exog = build_exog_training(train, stats, config)
    log(f"Training exogenous model on {len(x_exog):,} examples")
    exog = lgb.LGBMRegressor(**lgb_params(config, config.exog_estimators))
    exog_fit: dict = {}
    if exog_eval is not None:
        exog_fit = {
            "eval_set": [exog_eval],
            "callbacks": [lgb.early_stopping(80, verbose=True), lgb.log_evaluation(100)],
        }
    exog.fit(x_exog, y_exog, **exog_fit)
    del x_exog, y_exog
    gc.collect()
    return direct, exog


def validation_pairs(
    full_train: pd.DataFrame,
    stats: HistoryStats,
    anchors: Sequence[tuple[str, int]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    direct_xs: list[pd.DataFrame] = []
    exog_xs: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    horizons: list[np.ndarray] = []

    by_month = {int(k): v for k, v in full_train.groupby("month_idx", sort=False)}
    for anchor_date, max_h in anchors:
        anchor_idx = pd.Timestamp(anchor_date).year * 12 + pd.Timestamp(anchor_date).month - 1
        anchor_rows = by_month.get(anchor_idx)
        if anchor_rows is None:
            continue
        anchor_lookup = anchor_rows[["loc_key", "TWS_t", *CLIMATE]].rename(
            columns={"TWS_t": "anchor_TWS_t", **{c: f"anchor_{c}" for c in CLIMATE}}
        )
        for h in range(1, max_h + 1):
            current = by_month.get(anchor_idx + h - 1)
            if current is None:
                break
            pair = current.merge(anchor_lookup, on="loc_key", how="inner", validate="one_to_one")
            x_d, y_r, y = pair_to_xy(pair, stats, h)
            x_e = build_feature_frame(pair, stats)[EXOG_FEATURES]
            direct_xs.append(x_d)
            exog_xs.append(x_e)
            residuals.append(y_r)
            targets.append(y)
            horizons.append(np.full(len(pair), h, np.int8))

    if not targets:
        raise ValueError("No validation blocks could be constructed")
    return (
        pd.concat(direct_xs, ignore_index=True),
        np.concatenate(residuals),
        np.concatenate(targets),
        np.concatenate(horizons),
        pd.concat(exog_xs, ignore_index=True),
    )


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y, pred)))


def optimal_blend(y: np.ndarray, direct: np.ndarray, exog: np.ndarray) -> float:
    delta = direct - exog
    denom = float(np.dot(delta, delta))
    if denom <= 1e-12:
        return 0.5
    weight = -float(np.dot(delta, exog - y)) / denom
    return float(np.clip(weight, 0.0, 1.0))


def run_validation(data_dir: Path, artifact_dir: Path, config: ModelConfig) -> dict:
    train, _, _ = load_data(data_dir, need_test=False)
    # These blocks reproduce the leaderboard's mixture: repeated observed anchors,
    # several 2-3 month rollouts, and one long 7-month rollout across later years.
    anchors = [
        ("2012-07-01", 3),
        ("2012-12-01", 3),
        ("2013-05-01", 3),
        ("2013-11-01", 3),
        ("2014-04-01", 3),
        ("2014-09-01", 7),
    ]
    cutoff = pd.Timestamp(anchors[0][0])
    fit = train.loc[train["time"] < cutoff].copy()
    log(f"Chronological fit: {fit.time.min().date()} to {fit.time.max().date()} ({len(fit):,} rows)")
    stats = fit_history_stats(fit)
    val_d, val_residual, y_val, val_h, val_e = validation_pairs(train, stats, anchors)
    exog_val_residual = y_val - val_e["loc_climatology"].to_numpy(np.float32)
    direct, exog = fit_models(
        fit,
        stats,
        config,
        direct_eval=(val_d, val_residual),
        exog_eval=(val_e, exog_val_residual),
    )

    direct_pred = val_d["anchor_tws"].to_numpy(np.float32) + direct.predict(val_d)
    exog_pred = val_e["loc_climatology"].to_numpy(np.float32) + exog.predict(val_e)
    climatology = val_e["loc_climatology"].to_numpy(np.float32)
    trend = val_e["loc_trend"].to_numpy(np.float32)
    persistence = val_d["anchor_tws"].to_numpy(np.float32)

    # Estimate horizon-specific blend weights and shrink them toward a smooth,
    # conservative prior to avoid fitting one particular validation map.
    weights: dict[str, float] = {}
    rows = []
    for h in range(1, config.max_horizon + 1):
        mask = val_h == h
        if not mask.any():
            continue
        raw = optimal_blend(y_val[mask], direct_pred[mask], exog_pred[mask])
        prior = max(0.25, 0.92 - 0.09 * (h - 1))
        w = 0.7 * raw + 0.3 * prior
        weights[str(h)] = float(w)
        blend = w * direct_pred[mask] + (1.0 - w) * exog_pred[mask]
        rows.append(
            {
                "horizon": h,
                "n": int(mask.sum()),
                "persistence": rmse(y_val[mask], persistence[mask]),
                "climatology": rmse(y_val[mask], climatology[mask]),
                "trend": rmse(y_val[mask], trend[mask]),
                "direct": rmse(y_val[mask], direct_pred[mask]),
                "exogenous": rmse(y_val[mask], exog_pred[mask]),
                "blend_weight_direct": w,
                "blend": rmse(y_val[mask], blend),
            }
        )
    metrics = pd.DataFrame(rows)
    all_weights = np.array([weights[str(int(h))] for h in val_h], np.float32)
    blend_all = all_weights * direct_pred + (1.0 - all_weights) * exog_pred
    summary = {
        "cutoff": str(cutoff.date()),
        "anchors": anchors,
        "rmse": {
            "persistence": rmse(y_val, persistence),
            "climatology": rmse(y_val, climatology),
            "trend": rmse(y_val, trend),
            "direct": rmse(y_val, direct_pred),
            "exogenous": rmse(y_val, exog_pred),
            "blend": rmse(y_val, blend_all),
        },
        "blend_weights": weights,
        "best_iterations": {
            "direct": int(direct.best_iteration_ or config.direct_estimators),
            "exogenous": int(exog.best_iteration_ or config.exog_estimators),
        },
        "by_horizon": rows,
        "config": asdict(config),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "validation.json").write_text(json.dumps(summary, indent=2))
    metrics.to_csv(artifact_dir / "validation_by_horizon.csv", index=False)
    log("Validation by horizon:\n" + metrics.to_string(index=False))
    log("Overall validation:\n" + json.dumps(summary["rmse"], indent=2))
    return summary


def default_blend_weights(max_horizon: int) -> dict[str, float]:
    return {str(h): float(max(0.25, 0.92 - 0.09 * (h - 1))) for h in range(1, max_horizon + 1)}


def train_final(data_dir: Path, artifact_dir: Path, config: ModelConfig) -> None:
    train, _, _ = load_data(data_dir, need_test=False)
    validation_path = artifact_dir / "validation.json"
    weights = default_blend_weights(config.max_horizon)
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        weights.update(validation.get("blend_weights", {}))
        best = validation.get("best_iterations", {})
        # Reuse early-stopped complexity, with a small allowance because final
        # training has ~25% more history.
        config.direct_estimators = max(250, int(best.get("direct", config.direct_estimators) * 1.08))
        config.exog_estimators = max(250, int(best.get("exogenous", config.exog_estimators) * 1.08))
    stats = fit_history_stats(train)
    direct, exog = fit_models(train, stats, config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": 1,
        "seed": SEED,
        "config": asdict(config),
        "blend_weights": weights,
        "stats": stats,
        "direct_model": direct,
        "exog_model": exog,
        "direct_features": DIRECT_FEATURES,
        "exog_features": EXOG_FEATURES,
    }
    output = artifact_dir / "tws_models.joblib"
    joblib.dump(bundle, output, compress=3)
    log(f"Saved model bundle to {output}")


def infer_global_anchors(test: pd.DataFrame) -> dict[int, int]:
    """Map each test month to the latest mostly-observed TWS anchor month."""
    availability = test.groupby("month_idx")["TWS_t"].agg(["count", "size"])
    anchor_months = sorted(availability.index[(availability["count"] / availability["size"]) > 0.5])
    if not anchor_months:
        raise ValueError("No mostly-observed TWS anchor month found in test")
    result: dict[int, int] = {}
    for month in sorted(test["month_idx"].unique()):
        eligible = [a for a in anchor_months if a <= month]
        if not eligible:
            raise ValueError(f"Test month {month} occurs before the first observed anchor")
        result[int(month)] = int(eligible[-1])
    return result


def build_test_features(
    test: pd.DataFrame,
    stats: HistoryStats,
    max_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    anchors = infer_global_anchors(test)
    by_month = {int(k): v for k, v in test.groupby("month_idx", sort=False)}
    direct_frames: list[pd.DataFrame] = []
    exog_frames: list[pd.DataFrame] = []
    horizons_all: list[np.ndarray] = []

    for month in sorted(by_month):
        current = by_month[month].copy()
        global_anchor_month = anchors[month]
        anchor_rows = by_month[global_anchor_month][["loc_key", "TWS_t", *CLIMATE]].rename(
            columns={"TWS_t": "anchor_TWS_t", **{c: f"anchor_{c}" for c in CLIMATE}}
        )
        pair = current.merge(anchor_rows, on="loc_key", how="left", validate="one_to_one")
        global_h = int(np.clip(month + 1 - global_anchor_month, 1, max_horizon))
        horizons = np.full(len(pair), global_h, np.int8)

        # A small number of rows in otherwise masked maps retain a legal TWS_t.
        # They should use that stronger current-month anchor as horizon 1.
        current_available = np.isfinite(pair["TWS_t"].to_numpy(np.float32))
        pair.loc[current_available, "anchor_TWS_t"] = pair.loc[current_available, "TWS_t"]
        for c in CLIMATE:
            pair.loc[current_available, f"anchor_{c}"] = pair.loc[current_available, c]
        horizons[current_available] = 1

        anchor_df = pd.DataFrame(index=pair.index)
        anchor_df["TWS_t"] = pair["anchor_TWS_t"].to_numpy(np.float32)
        for c in CLIMATE:
            anchor_df[c] = pair[f"anchor_{c}"].to_numpy(np.float32)
        x_direct = build_feature_frame(pair, stats, anchor=anchor_df, horizon=horizons)[DIRECT_FEATURES]
        x_exog = build_feature_frame(pair, stats)[EXOG_FEATURES]
        x_direct["_row_id"] = pair["_row_id"].to_numpy(np.int64)
        x_exog["_row_id"] = pair["_row_id"].to_numpy(np.int64)
        direct_frames.append(x_direct)
        exog_frames.append(x_exog)
        horizons_all.append(horizons)

    d = pd.concat(direct_frames, ignore_index=True).sort_values("_row_id")
    e = pd.concat(exog_frames, ignore_index=True).sort_values("_row_id")
    # Horizons must follow the same row-id order, so recover them from the direct
    # feature rather than concatenation order after sorting.
    horizons = d["horizon"].to_numpy(np.int8)
    return d.drop(columns="_row_id"), e.drop(columns="_row_id"), horizons


def predict_submission(data_dir: Path, artifact_dir: Path, output: Path) -> None:
    bundle = joblib.load(artifact_dir / "tws_models.joblib")
    _, test, sample_path = load_data(data_dir, need_test=True)
    assert test is not None and sample_path is not None
    if test["ID"].duplicated().any():
        raise ValueError("Duplicate test IDs")
    test = test.copy()
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    config = ModelConfig(**bundle["config"])
    x_direct, x_exog, horizons = build_test_features(test, bundle["stats"], config.max_horizon)
    direct = x_direct["anchor_tws"].to_numpy(np.float32) + bundle["direct_model"].predict(x_direct)
    exog = x_exog["loc_climatology"].to_numpy(np.float32) + bundle["exog_model"].predict(x_exog)
    weights_map = bundle["blend_weights"]
    weights = np.array(
        [weights_map.get(str(int(h)), default_blend_weights(config.max_horizon)[str(int(h))]) for h in horizons],
        dtype=np.float32,
    )
    pred = weights * direct + (1.0 - weights) * exog
    stats: HistoryStats = bundle["stats"]
    pred = np.clip(pred, stats.target_min - 0.25, stats.target_max + 0.25).astype(np.float32)

    raw = pd.DataFrame({"ID": test["ID"], "Target": pred})
    sample = pd.read_csv(sample_path)
    if set(sample.columns) != {"ID", "Target"}:
        raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")
    submission = sample[["ID"]].merge(raw, on="ID", how="left", validate="one_to_one")
    if submission["Target"].isna().any() or len(submission) != len(sample):
        raise ValueError("Submission does not contain one finite prediction per sample ID")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    log(f"Saved {len(submission):,} predictions to {output}")
    log("Prediction summary:\n" + submission["Target"].describe().to_string())
    log("Rows by effective horizon: " + str(pd.Series(horizons).value_counts().sort_index().to_dict()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "train", "predict", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--rows-per-horizon", type=int, default=320_000)
    parser.add_argument("--exog-rows", type=int, default=1_500_000)
    parser.add_argument("--direct-estimators", type=int, default=750)
    parser.add_argument("--exog-estimators", type=int, default=700)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()
    config = ModelConfig(
        rows_per_horizon=args.rows_per_horizon,
        exog_rows=args.exog_rows,
        direct_estimators=args.direct_estimators,
        exog_estimators=args.exog_estimators,
        n_jobs=args.n_jobs,
    )
    if args.command in {"validate", "all"}:
        run_validation(args.data_dir, args.artifact_dir, config)
    if args.command in {"train", "all"}:
        train_final(args.data_dir, args.artifact_dir, config)
    if args.command in {"predict", "all"}:
        predict_submission(args.data_dir, args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
