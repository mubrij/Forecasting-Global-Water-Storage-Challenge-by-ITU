#!/usr/bin/env python3
"""Chronological A/B test of label-reconstructed historical TWS features."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from history_experiment import LAGS, add_history
from map_unet_solution import VAL_BLOCKS, midx
from recovered_history import append_observed_anchor, reconstruct_tws_history
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED,
    ModelConfig,
    fit_history_stats,
    lgb_params,
    load_data,
    pair_to_xy,
    rmse,
)


def validation_set(frame, fit_history, stats):
    xs, residuals, targets, horizons, blocks = [], [], [], [], []
    for block, (date, max_h) in enumerate(VAL_BLOCKS):
        anchor_month = midx(date)
        anchor_rows = frame[frame.month_idx == anchor_month]
        legal_history = append_observed_anchor(fit_history, anchor_rows)
        allowed = set(map(int, legal_history.month_idx.unique()))
        for horizon in range(1, max_h + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor_month + horizon - 1].reset_index(drop=True)
            if pair.empty:
                break
            x, residual, target = pair_to_xy(pair, stats, horizon)
            x = add_history(add_spatial_x(x, pair), pair, legal_history, horizon, allowed)
            xs.append(x)
            residuals.append(residual)
            targets.append(target)
            horizons.append(np.full(len(pair), horizon, np.int8))
            blocks.append(np.full(len(pair), block, np.int8))
    return (
        pd.concat(xs, ignore_index=True),
        np.concatenate(residuals),
        np.concatenate(targets),
        np.concatenate(horizons),
        np.concatenate(blocks),
    )


def training_set(fit, fit_history, stats, rows_per_horizon):
    xs, residuals = [], []
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        if len(pair) > rows_per_horizon:
            pair = pair.sample(rows_per_horizon, random_state=SEED + horizon)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        x = add_history(add_spatial_x(x, pair), pair, fit_history, horizon)
        # Match deployment sparsity: recent lags may be absent between anchors.
        rng = np.random.RandomState(SEED + 100 + horizon)
        for lag in LAGS:
            if lag < 6:
                drop = rng.random(len(x)) < 0.35
                for name in (
                    f"history_tws_lag{lag}",
                    f"anchor_change_lag{lag}",
                    f"anchor_velocity_lag{lag}",
                ):
                    x.loc[drop, name] = np.nan
                x.loc[drop, f"history_missing_lag{lag}"] = 1.0
        xs.append(x)
        residuals.append(residual)
        print(f"training h={horizon}: {len(x):,}", flush=True)
        del pair, x
        gc.collect()
    return pd.concat(xs, ignore_index=True), np.concatenate(residuals)


def run_variant(name, frame, fit, history, rows_per_horizon, estimators):
    stats = fit_history_stats(history)
    vx, vr, target, vh, vb = validation_set(frame, history, stats)
    tx, tr = training_set(fit, history, stats, rows_per_horizon)
    config = ModelConfig(
        direct_estimators=estimators,
        learning_rate=0.03,
        num_leaves=80,
        min_child_samples=300,
        n_jobs=20,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, estimators))
    model.fit(
        tx,
        tr,
        eval_set=[(vx, vr)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )
    prediction = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
    result = {
        "name": name,
        "score": rmse(target, prediction),
        "iteration": int(model.best_iteration_),
        "horizon": {h: rmse(target[vh == h], prediction[vh == h]) for h in range(1, 8)},
        "block": {b: rmse(target[vb == b], prediction[vb == b]) for b in range(len(VAL_BLOCKS))},
    }
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-horizon", type=int, default=60_000)
    parser.add_argument("--estimators", type=int, default=800)
    parser.add_argument("--variant", choices=["baseline", "recovered", "both"], default="both")
    args = parser.parse_args()

    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    cutoff = midx(VAL_BLOCKS[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    baseline_history = fit[["time", "month_idx", "loc_key", "TWS_t", "Target"]].copy()
    recovered_history = reconstruct_tws_history(frame, before_month=cutoff)
    print(
        "history maps",
        baseline_history.month_idx.nunique(),
        "->",
        recovered_history.month_idx.nunique(),
        "rows",
        len(baseline_history),
        "->",
        len(recovered_history),
        flush=True,
    )

    if args.variant in ("baseline", "both"):
        run_variant("baseline", frame, fit, baseline_history, args.rows_per_horizon, args.estimators)
    if args.variant in ("recovered", "both"):
        run_variant("recovered", frame, fit, recovered_history, args.rows_per_horizon, args.estimators)


if __name__ == "__main__":
    main()
