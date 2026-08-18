#!/usr/bin/env python3
"""Test CFSv2 continuous land-water-state updates between GRACE anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cfsv2_solution import CFSFeatures, select_validation_blend
from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED, ModelConfig, fit_history_stats, lgb_params, load_data, pair_to_xy, rmse,
)


def validation_rows(frame, stats, cfs):
    xs, residuals, truths, horizons, groups, keys = [], [], [], [], [], []
    group = 0
    for date, max_h in VAL_BLOCKS:
        anchor = midx(date)
        for horizon in range(1, max_h + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
            if pair.empty:
                break
            x, residual, truth = pair_to_xy(pair, stats, horizon)
            x = cfs.add_hydrologic_path(add_spatial_x(x, pair), pair, horizon)
            xs.append(x); residuals.append(residual); truths.append(truth)
            horizons.append(np.full(len(pair), horizon, np.int8))
            groups.append(np.full(len(pair), group, np.int16))
            keys.append(pair.loc_key.to_numpy(np.int32)); group += 1
    return (pd.concat(xs, ignore_index=True), np.concatenate(residuals),
            np.concatenate(truths), np.concatenate(horizons),
            np.concatenate(groups), np.concatenate(keys))


def training_rows(frame, stats, cfs, rows_per_horizon, use_path=True):
    xs, ys = [], []
    available = set(map(int, cfs.months))
    for horizon in range(1, 8):
        pair = make_pair(frame, horizon)
        anchor_month = pair.month_idx.to_numpy(np.int32) - horizon + 1
        legal = pair.month_idx.isin(available).to_numpy() & np.isin(anchor_month, list(available))
        pair = pair.loc[legal].reset_index(drop=True)
        if len(pair) > rows_per_horizon:
            pair = pair.sample(rows_per_horizon, random_state=SEED + horizon).reset_index(drop=True)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        spatial = add_spatial_x(x, pair)
        x = cfs.add_hydrologic_path(spatial, pair, horizon) if use_path else cfs.add(spatial, pair)
        xs.append(x); ys.append(residual)
        print(f"h={horizon}: {len(pair):,}", flush=True)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("external_cfsv2"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_cfsv2_path"))
    parser.add_argument("--rows-per-horizon", type=int, default=120_000)
    args = parser.parse_args()

    frame, _, _ = load_data(args.data_dir, need_test=False)
    frame = augment(frame)
    cutoff = midx(VAL_BLOCKS[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)
    cfs = CFSFeatures(args.cache_dir / "features_cfsv2.npz")
    vx, vr, truth, horizon, group, key = validation_rows(frame, stats, cfs)
    tx, ty = training_rows(fit, stats, cfs, args.rows_per_horizon)
    config = ModelConfig(direct_estimators=900, learning_rate=.025, num_leaves=72,
                         min_child_samples=350, n_jobs=20)
    model = lgb.LGBMRegressor(**lgb_params(config, config.direct_estimators))
    model.fit(tx, ty, eval_set=[(vx, vr)],
              callbacks=[lgb.early_stopping(120), lgb.log_evaluation(50)])
    prediction = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
    print("CFS path validation", rmse(truth, prediction), flush=True)
    print({h: rmse(truth[horizon == h], prediction[horizon == h]) for h in range(1, 8)}, flush=True)
    detail = select_validation_blend(truth, prediction, horizon, group, key)
    detail["best_iteration"] = int(model.best_iteration_)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact_dir / "validation_predictions.npz",
                       truth=truth, prediction=prediction, horizon=horizon,
                       group=group, key=key)
    (args.artifact_dir / "validation.json").write_text(json.dumps(detail, indent=2))
    print(json.dumps(detail, indent=2), flush=True)


if __name__ == "__main__":
    main()
