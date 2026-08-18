#!/usr/bin/env python3
"""Walk-forward validation for CFSv2 continuous hydrologic path features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from cfsv2_path_experiment import training_rows
from cfsv2_solution import CFSFeatures
from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import ModelConfig, fit_history_stats, lgb_params, load_data, pair_to_xy, rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-control", action="store_true")
    args = parser.parse_args()
    use_path = not args.baseline_control
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    cfs = CFSFeatures(Path("external_cfsv2/features_cfsv2.npz"))
    predictions, truths, horizons, groups, keys = [], [], [], [], []
    group_offset = 0
    block_scores = []

    for block, (date, max_horizon) in enumerate(VAL_BLOCKS):
        anchor_month = midx(date)
        fit = frame[frame.month_idx < anchor_month].copy()
        stats = fit_history_stats(fit)
        vxs, vrs, vys, vhs, vgs, vkeys = [], [], [], [], [], []
        for horizon in range(1, max_horizon + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor_month + horizon - 1].reset_index(drop=True)
            if pair.empty:
                break
            x, residual, truth = pair_to_xy(pair, stats, horizon)
            spatial = add_spatial_x(x, pair)
            x = cfs.add_hydrologic_path(spatial, pair, horizon) if use_path else cfs.add(spatial, pair)
            vxs.append(x); vrs.append(residual); vys.append(truth)
            vhs.append(np.full(len(pair), horizon, np.int8))
            vgs.append(np.full(len(pair), group_offset, np.int16))
            vkeys.append(pair.loc_key.to_numpy(np.int32))
            group_offset += 1
        vx, vr, vy = pd.concat(vxs, ignore_index=True), np.concatenate(vrs), np.concatenate(vys)
        vh, vg, vk = np.concatenate(vhs), np.concatenate(vgs), np.concatenate(vkeys)
        tx, ty = training_rows(fit, stats, cfs, 100_000, use_path=use_path)
        config = ModelConfig(direct_estimators=700, learning_rate=.025, num_leaves=72,
                             min_child_samples=350, n_jobs=20)
        model = lgb.LGBMRegressor(**lgb_params(config, config.direct_estimators))
        model.fit(tx, ty, eval_set=[(vx, vr)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
        pred = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
        score = rmse(vy, pred)
        block_scores.append(score)
        print(f"block={block} anchor={date} iter={model.best_iteration_} rmse={score:.9f}", flush=True)
        predictions.append(pred); truths.append(vy); horizons.append(vh); groups.append(vg); keys.append(vk)

    pred, truth = np.concatenate(predictions), np.concatenate(truths)
    horizon, group, key = np.concatenate(horizons), np.concatenate(groups), np.concatenate(keys)
    old = np.load("artifacts_cfsv2_walkforward/validation_predictions.npz")
    if not np.array_equal(group, old["group"]) or not np.array_equal(key, old["key"]):
        raise ValueError("walk-forward artifact alignment differs from prior CFS component")
    old_pred = old["prediction"]
    print("path overall", rmse(truth, pred), flush=True)
    print("old overall", rmse(truth, old_pred), flush=True)
    print("path horizon", {h: rmse(truth[horizon == h], pred[horizon == h]) for h in range(1, 8)}, flush=True)
    for weight in (0, .1, .2, .3, .5, .75, 1):
        mixed = old_pred + weight * (pred - old_pred)
        print(f"replace_weight={weight:.2f} rmse={rmse(truth, mixed):.9f}", flush=True)
    output = Path("artifacts_cfsv2_path_walkforward" if use_path else "artifacts_cfsv2_matched_walkforward")
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "validation_predictions.npz", truth=truth,
                        prediction=pred, horizon=horizon, group=group, key=key)
    (output / "validation.json").write_text(json.dumps({
        "overall": rmse(truth, pred), "old_overall": rmse(truth, old_pred),
        "blocks": block_scores,
    }, indent=2))


if __name__ == "__main__":
    main()
