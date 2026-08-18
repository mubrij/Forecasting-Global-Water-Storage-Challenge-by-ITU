#!/usr/bin/env python3
"""Honest chronological benchmark for multi-lag TWS history features."""
from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED, ModelConfig, fit_history_stats, lgb_params, load_data, pair_to_xy, rmse,
)

LAGS = [1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36]


def add_history(x, pair, frame, horizon, allowed_months=None):
    out = x.copy()
    anchor = out.anchor_tws.to_numpy(np.float32)
    current_month = int(pair.month_idx.iloc[0]) if pair.month_idx.nunique() == 1 else None
    for lag in LAGS:
        source_month = None if current_month is None else current_month - horizon + 1 - lag
        if allowed_months is not None and source_month not in allowed_months:
            values = np.full(len(pair), np.nan, np.float32)
        else:
            history = frame[["month_idx", "loc_key", "TWS_t"]].copy()
            history["month_idx"] += horizon - 1 + lag
            name = f"history_tws_lag{lag}"
            history = history.rename(columns={"TWS_t": name})
            joined = pair[["month_idx", "loc_key"]].merge(
                history, on=["month_idx", "loc_key"], how="left", validate="one_to_one"
            )
            values = joined[name].to_numpy(np.float32)
        out[f"history_tws_lag{lag}"] = values
        out[f"anchor_change_lag{lag}"] = anchor - values
        out[f"anchor_velocity_lag{lag}"] = (anchor - values) / lag
        out[f"history_missing_lag{lag}"] = (~np.isfinite(values)).astype(np.float32)
    # Recent seasonal acceleration is often more stable than an absolute trend.
    if "history_tws_lag12" in out and "history_tws_lag24" in out:
        out["seasonal_year_change"] = out.history_tws_lag12 - out.history_tws_lag24
        out["anchor_vs_last_year"] = out.anchor_tws - out.history_tws_lag12
    return out


def main():
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    cutoff = midx(VAL_BLOCKS[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)

    # Only pre-cutoff maps and the declared validation anchors are observable.
    allowed = set(map(int, fit.month_idx.unique()))
    val_x, val_r, truth, horizons = [], [], [], []
    for date, max_h in VAL_BLOCKS:
        anchor = midx(date)
        allowed.add(anchor)
        for horizon in range(1, max_h + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
            if pair.empty:
                break
            x, residual, target = pair_to_xy(pair, stats, horizon)
            x = add_history(add_spatial_x(x, pair), pair, frame, horizon, allowed)
            val_x.append(x); val_r.append(residual); truth.append(target)
            horizons.append(np.full(len(pair), horizon, np.int8))
    vx = pd.concat(val_x, ignore_index=True)
    vr, y, vh = np.concatenate(val_r), np.concatenate(truth), np.concatenate(horizons)

    xs, ys = [], []
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        if len(pair) > 100_000:
            take = np.random.RandomState(SEED + horizon).choice(len(pair), 100_000, False)
            pair, x, residual = pair.iloc[take], x.iloc[take], residual[take]
        x = add_history(add_spatial_x(x, pair), pair, fit, horizon)
        # History dropout teaches the tree the sparse-anchor patterns in test.
        rng = np.random.RandomState(SEED + 100 + horizon)
        for lag in LAGS:
            if lag < 6:
                drop = rng.random(len(x)) < 0.35
                for name in (f"history_tws_lag{lag}", f"anchor_change_lag{lag}",
                             f"anchor_velocity_lag{lag}"):
                    x.loc[drop, name] = np.nan
                x.loc[drop, f"history_missing_lag{lag}"] = 1.0
        xs.append(x); ys.append(residual)
        print("h", horizon, len(x), flush=True)
        del pair, x
        gc.collect()
    tx, ty = pd.concat(xs, ignore_index=True), np.concatenate(ys)
    config = ModelConfig(direct_estimators=800, learning_rate=.03, num_leaves=80,
                         min_child_samples=300, n_jobs=20)
    model = lgb.LGBMRegressor(**lgb_params(config, 800))
    model.fit(tx, ty, eval_set=[(vx, vr)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
    pred = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
    print("history overall", rmse(y, pred))
    print({h: rmse(y[vh == h], pred[vh == h]) for h in range(1, 8)})


if __name__ == "__main__":
    main()
