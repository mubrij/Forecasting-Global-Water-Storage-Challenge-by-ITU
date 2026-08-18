#!/usr/bin/env python3
"""Test robust objectives for the recency-weighted regime model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from map_unet_solution import VAL_BLOCKS, midx
from recency_regime_experiment import LATE_BLOCKS, training_set, validation_set
from spatial_experiment import augment
from winning_solution import ModelConfig, fit_history_stats, lgb_params, load_data, rmse


def centered(values, groups):
    out = values.copy()
    for group in np.unique(groups):
        take = groups == group
        out[take] -= out[take].mean()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=["late", "original"], required=True)
    parser.add_argument("--rows-per-horizon", type=int, default=160000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    blocks = LATE_BLOCKS if args.schedule == "late" else VAL_BLOCKS
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    cutoff = midx(blocks[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)
    tx, ty, months = training_set(fit, stats, args.rows_per_horizon)
    vx, truth, horizon, group = validation_set(frame, stats, blocks)
    weight = np.power(0.5, months.max() - months)
    weight /= weight.mean()
    config = ModelConfig(direct_estimators=args.iterations, learning_rate=.03,
                         num_leaves=72, min_child_samples=400, n_jobs=args.n_jobs)
    predictions, summary = {}, {}
    for name, objective in (("huber", "huber"), ("l1", "regression_l1")):
        params = lgb_params(config, args.iterations)
        params.update(objective=objective, metric="l2")
        if name == "huber":
            params["alpha"] = .9
        model = lgb.LGBMRegressor(**params)
        model.fit(tx, ty, sample_weight=weight, callbacks=[lgb.log_evaluation(0)])
        pred = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
        predictions[name] = pred
        summary[name] = {
            "rmse": rmse(truth, pred),
            "centered_rmse": rmse(centered(truth, group), centered(pred, group)),
            "horizon": {str(h): rmse(truth[horizon == h], pred[horizon == h])
                        for h in range(1, 8) if np.any(horizon == h)},
        }
        print(name, json.dumps(summary[name]), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", truth=truth,
                        horizon=horizon, group=group, **predictions)
    (args.output_dir / "validation.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
