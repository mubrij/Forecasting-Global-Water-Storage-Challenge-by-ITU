#!/usr/bin/env python3
"""Walk-forward A/B test of anomaly and hydrologic-memory features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from history_experiment import LAGS, add_history
from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    CLIMATE, SEED, ModelConfig, fit_history_stats, lgb_params, load_data,
    pair_to_xy, rmse,
)


def calendar_climatology(stats, keys: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Empirical-Bayes per-cell climatology for an arbitrary calendar month."""
    safe = np.clip(keys, 0, stats.max_loc_key)
    index = safe * 12 + month
    count = stats.loc_month_count[index].astype(np.float32)
    raw = stats.loc_month_mean[index]
    mean = stats.loc_mean[safe]
    return ((count * raw + 3.0 * mean) / (count + 3.0)).astype(np.float32)


def add_advanced_features(x: pd.DataFrame, pair: pd.DataFrame, stats, horizon: int) -> pd.DataFrame:
    out = x.copy()
    keys = pair.loc_key.to_numpy(np.int32)
    anchor_month = pair.month_idx.to_numpy(np.int32) - horizon + 1
    anchor_clim = calendar_climatology(stats, keys, anchor_month % 12)
    scale = np.maximum(stats.loc_std[np.clip(keys, 0, stats.max_loc_key)], 0.15)
    anchor = out.anchor_tws.to_numpy(np.float32)
    anchor_anomaly = anchor - anchor_clim

    out["anchor_calendar_climatology"] = anchor_clim
    out["anchor_seasonal_anomaly"] = anchor_anomaly
    out["anchor_seasonal_z"] = anchor_anomaly / scale
    out["climatology_change"] = out.loc_climatology.to_numpy(np.float32) - anchor_clim
    out["seasonal_persistence"] = out.loc_climatology.to_numpy(np.float32) + anchor_anomaly
    out["seasonal_persistence_minus_anchor"] = out.seasonal_persistence - anchor

    anomaly_columns = {}
    for lag in LAGS:
        name = f"history_tws_lag{lag}"
        if name not in out:
            continue
        lag_clim = calendar_climatology(stats, keys, (anchor_month - lag) % 12)
        anomaly = out[name].to_numpy(np.float32) - lag_clim
        anomaly_columns[lag] = anomaly
        out[f"history_anomaly_lag{lag}"] = anomaly
        out[f"anomaly_change_lag{lag}"] = anchor_anomaly - anomaly
        out[f"anomaly_velocity_lag{lag}"] = (anchor_anomaly - anomaly) / lag

    if 1 in anomaly_columns and 3 in anomaly_columns:
        out["anomaly_acceleration_1_3"] = (
            anchor_anomaly - anomaly_columns[1]
            - (anomaly_columns[1] - anomaly_columns[3]) / 2.0
        )
    if 6 in anomaly_columns and 12 in anomaly_columns:
        out["anomaly_acceleration_6_12"] = (
            (anchor_anomaly - anomaly_columns[6]) / 6.0
            - (anomaly_columns[6] - anomaly_columns[12]) / 6.0
        )
    if 12 in anomaly_columns and 24 in anomaly_columns:
        out["interannual_anomaly_trend"] = (anomaly_columns[12] - anomaly_columns[24]) / 12.0

    h = out.horizon.to_numpy(np.float32)
    spei1 = out.current_SPEI_01_t.to_numpy(np.float32)
    spei3 = out.current_SPEI_03_t.to_numpy(np.float32)
    spei6 = out.current_SPEI_06_t.to_numpy(np.float32)
    spei12 = out.current_SPEI_12_t.to_numpy(np.float32)
    out["spei_memory_weighted"] = 0.40 * spei1 + 0.30 * spei3 + 0.20 * spei6 + 0.10 * spei12
    out["spei_short_long_gradient"] = 0.5 * (spei1 + spei3) - 0.5 * (spei6 + spei12)
    out["spei_dispersion"] = np.std(np.column_stack([spei1, spei3, spei6, spei12]), axis=1)
    out["soil_change_per_month"] = out.change_SOIL_MOISTURE_t.to_numpy(np.float32) / h
    out["spei01_change_per_month"] = out.change_SPEI_01_t.to_numpy(np.float32) / h
    out["anomaly_horizon"] = anchor_anomaly * h
    out["anomaly_sqrt_horizon"] = anchor_anomaly * np.sqrt(h)
    out["dryness_horizon"] = out.spei_memory_weighted.to_numpy(np.float32) * h
    out["soil_horizon"] = out.current_SOIL_MOISTURE_t.to_numpy(np.float32) * h
    out["absolute_latitude"] = np.abs(pair.lat.to_numpy(np.float32))
    out["tropical"] = (out.absolute_latitude < 23.5).astype(np.float32)
    out["high_latitude"] = (out.absolute_latitude > 55.0).astype(np.float32)
    return out


def feature_frame(pair, stats, history, horizon: int, advanced: bool,
                  anomaly_target: bool = True):
    x, anchor_residual, truth = pair_to_xy(pair, stats, horizon)
    x = add_spatial_x(x, pair)
    if not advanced:
        return x, anchor_residual, truth
    x = add_history(x, pair, history, horizon)
    x = add_advanced_features(x, pair, stats, horizon)
    # Explicit ablation: engineered predictors can retain the original
    # anchor-residual objective or use a seasonally stationary objective.
    target = (
        truth - x.loc_climatology.to_numpy(np.float32)
        if anomaly_target else anchor_residual
    )
    return x, target, truth


def build_training(fit, stats, advanced: bool, rows_per_horizon: int,
                   anomaly_target: bool = True):
    xs, ys = [], []
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        if len(pair) > rows_per_horizon:
            pair = pair.sample(rows_per_horizon, random_state=SEED + horizon).reset_index(drop=True)
        x, target, _ = feature_frame(
            pair, stats, fit, horizon, advanced, anomaly_target
        )
        xs.append(x); ys.append(target)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def build_validation(frame, fit, stats, anchor: int, max_horizon: int,
                     advanced: bool, anomaly_target: bool = True):
    anchor_rows = frame[frame.month_idx == anchor]
    history = pd.concat([fit, anchor_rows], ignore_index=True).drop_duplicates(
        ["month_idx", "loc_key"], keep="last"
    )
    xs, targets, truths, horizons = [], [], [], []
    for horizon in range(1, max_horizon + 1):
        pair = make_pair(frame, horizon)
        pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
        if pair.empty:
            break
        x, target, truth = feature_frame(
            pair, stats, history, horizon, advanced, anomaly_target
        )
        xs.append(x); targets.append(target); truths.append(truth)
        horizons.append(np.full(len(pair), horizon, np.int8))
    return pd.concat(xs, ignore_index=True), np.concatenate(targets), np.concatenate(truths), np.concatenate(horizons)


def fit_predict(tx, ty, vx, iterations: int, n_jobs: int):
    config = ModelConfig(
        direct_estimators=iterations, learning_rate=0.03, num_leaves=72,
        min_child_samples=400, n_jobs=n_jobs,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, iterations))
    model.fit(tx, ty, callbacks=[lgb.log_evaluation(0)])
    return model.predict(vx), model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-horizon", type=int, default=50000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument(
        "--advanced-target", choices=["seasonal", "anchor"], default="seasonal",
        help="Residual objective used by the advanced model.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_advanced_features"))
    args = parser.parse_args()
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    anomaly_target = args.advanced_target == "seasonal"
    records, arrays = [], {name: [] for name in ("truth", "horizon", "block", "baseline", "advanced")}

    for block, (date, max_horizon) in enumerate(VAL_BLOCKS):
        anchor = midx(date)
        fit = frame[frame.month_idx < anchor].copy()
        stats = fit_history_stats(fit)
        block_predictions = {}
        block_truth = block_horizon = None
        for advanced in (False, True):
            name = "advanced" if advanced else "baseline"
            tx, ty = build_training(
                fit, stats, advanced, args.rows_per_horizon,
                anomaly_target if advanced else False,
            )
            vx, _, truth, horizon = build_validation(
                frame, fit, stats, anchor, max_horizon, advanced,
                anomaly_target if advanced else False,
            )
            raw, _ = fit_predict(tx, ty, vx, args.iterations, args.n_jobs)
            prediction = (
                vx.loc_climatology.to_numpy(np.float32) + raw
                if advanced and anomaly_target
                else vx.anchor_tws.to_numpy(np.float32) + raw
            )
            block_predictions[name] = prediction
            block_truth, block_horizon = truth, horizon
            print(f"block={block} {date} {name} rmse={rmse(truth, prediction):.9f}", flush=True)
        assert block_truth is not None and block_horizon is not None
        records.append({
            "block": block, "date": date,
            "baseline": rmse(block_truth, block_predictions["baseline"]),
            "advanced": rmse(block_truth, block_predictions["advanced"]),
        })
        arrays["truth"].append(block_truth); arrays["horizon"].append(block_horizon)
        arrays["block"].append(np.full(len(block_truth), block, np.int8))
        for name in ("baseline", "advanced"):
            arrays[name].append(block_predictions[name])

    final = {name: np.concatenate(values) for name, values in arrays.items()}
    summary = {
        "overall": {name: rmse(final["truth"], final[name]) for name in ("baseline", "advanced")},
        "blocks": records,
        "horizon": {
            str(h): {name: rmse(final["truth"][final["horizon"] == h], final[name][final["horizon"] == h])
                     for name in ("baseline", "advanced")}
            for h in range(1, 8) if np.any(final["horizon"] == h)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "walkforward_predictions.npz", **final)
    (args.output_dir / "validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
