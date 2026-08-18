#!/usr/bin/env python3
"""Late-era backtest of recency-weighted spatial residual models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED, ModelConfig, fit_history_stats, lgb_params, load_data, pair_to_xy, rmse,
)


LATE_BLOCKS = [
    ("2014-09-01", 7),
    ("2014-12-01", 7),
    ("2015-01-01", 7),
    ("2015-02-01", 7),
    ("2015-03-01", 6),
]


def training_set(fit, stats, rows_per_horizon):
    xs, ys, months = [], [], []
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        if len(pair) > rows_per_horizon:
            pair = pair.sample(rows_per_horizon, random_state=SEED + horizon).reset_index(drop=True)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        xs.append(add_spatial_x(x, pair)); ys.append(residual)
        months.append(pair.month_idx.to_numpy(np.int32))
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(months)


def validation_set(frame, stats, blocks):
    xs, ys, horizons, groups = [], [], [], []
    group = 0
    for date, max_horizon in blocks:
        anchor = midx(date)
        for horizon in range(1, max_horizon + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
            if pair.empty:
                break
            x, _, truth = pair_to_xy(pair, stats, horizon)
            xs.append(add_spatial_x(x, pair)); ys.append(truth)
            horizons.append(np.full(len(pair), horizon, np.int8))
            groups.append(np.full(len(pair), group, np.int16)); group += 1
    return (pd.concat(xs, ignore_index=True), np.concatenate(ys),
            np.concatenate(horizons), np.concatenate(groups))


def centered(values, groups):
    result = values.copy()
    for group in np.unique(groups):
        take = groups == group
        result[take] -= result[take].mean()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-horizon", type=int, default=80000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--schedule", choices=["late", "original"], default="late")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_recency_regime"))
    args = parser.parse_args()
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    blocks = LATE_BLOCKS if args.schedule == "late" else VAL_BLOCKS
    cutoff = midx(blocks[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)
    tx, ty, train_month = training_set(fit, stats, args.rows_per_horizon)
    vx, truth, horizon, group = validation_set(frame, stats, blocks)
    config = ModelConfig(direct_estimators=args.iterations, learning_rate=.03,
                         num_leaves=72, min_child_samples=400, n_jobs=args.n_jobs)
    predictions = {}
    summary = {}
    max_month = int(train_month.max())
    for half_life in (None, 1):
        name = "uniform" if half_life is None else f"half_life_{half_life}"
        weight = None
        if half_life is not None:
            weight = np.power(0.5, (max_month - train_month) / half_life)
            weight /= weight.mean()
        model = lgb.LGBMRegressor(**lgb_params(config, args.iterations))
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

    base = predictions["uniform"]
    blend = {}
    for name, pred in predictions.items():
        if name == "uniform":
            continue
        direction = centered(pred - base, group)
        rows = []
        for weight in np.linspace(-1.0, 1.0, 41):
            candidate = base + weight * direction
            rows.append((float(weight), rmse(truth, candidate)))
        blend[name] = {"best_weight": min(rows, key=lambda row: row[1])[0],
                       "best_rmse": min(row[1] for row in rows), "grid": rows}
        print("blend", name, blend[name]["best_weight"], blend[name]["best_rmse"], flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", truth=truth,
                        horizon=horizon, group=group, **predictions)
    (args.output_dir / "validation.json").write_text(json.dumps(
        {"models": summary, "blends": blend, "blocks": blocks}, indent=2
    ))


if __name__ == "__main__":
    main()
