#!/usr/bin/env python3
"""Align chronological tree/ridge and residual-U-Net maps; test legal blends."""
from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ensemble_solution import BLEND_WEIGHTS
from map_unet_solution import MapStore, VAL_BLOCKS, midx, validation_examples
from ridge_experiment import fit_local, ridge_matrix
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED, ModelConfig, fit_history_stats, lgb_params, load_data, pair_to_xy, rmse,
)


def optimal_weight(y, a, b):
    direction = a - b
    return float(np.clip(np.dot(y - b, direction) / np.dot(direction, direction), 0, 1))


def smooth(values, keys, groups, sigma):
    result = np.empty_like(values)
    for group in np.unique(groups):
        take = np.flatnonzero(groups == group)
        kk = keys[take]
        ii, jj = kk // 360, kk % 360
        grid = np.zeros((140, 360), np.float32)
        mask = np.zeros((140, 360), np.float32)
        grid[ii, jj], mask[ii, jj] = values[take], 1
        num = gaussian_filter(grid, sigma=sigma, mode=("nearest", "wrap"))
        den = gaussian_filter(mask, sigma=sigma, mode=("nearest", "wrap"))
        result[take] = np.divide(num, den, out=grid, where=den > 1e-6)[ii, jj]
    return result


def main():
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    cutoff = midx(VAL_BLOCKS[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)

    # Validation rows in the same group order as map_unet_solution.
    val_xs, val_y, val_h, val_g, val_k = [], [], [], [], []
    group = 0
    for date, max_h in VAL_BLOCKS:
        anchor_idx = midx(date)
        for horizon in range(1, max_h + 1):
            current_idx = anchor_idx + horizon - 1
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == current_idx].reset_index(drop=True)
            if pair.empty:
                break
            x, _, y = pair_to_xy(pair, stats, horizon)
            val_xs.append(add_spatial_x(x, pair))
            val_y.append(y)
            val_h.append(np.full(len(pair), horizon, np.int8))
            val_g.append(np.full(len(pair), group, np.int16))
            val_k.append(pair.loc_key.to_numpy(np.int32))
            group += 1
    vx = pd.concat(val_xs, ignore_index=True)
    truth = np.concatenate(val_y)
    horizons, groups, keys = map(np.concatenate, (val_h, val_g, val_k))

    xs, ys = [], []
    ridge = np.empty(len(truth), np.float32)
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        x, residual, y = pair_to_xy(pair, stats, horizon)
        betas, _ = fit_local(pair, x, y.astype(np.float64), stats.max_loc_key, 100.0)
        mask = horizons == horizon
        ridge[mask] = np.einsum(
            "ij,ij->i", ridge_matrix(vx.loc[mask]), betas[keys[mask]]
        )
        if len(pair) > 60_000:
            take = np.random.RandomState(SEED + horizon).choice(len(pair), 60_000, False)
            pair, x, residual = pair.iloc[take], x.iloc[take], residual[take]
        xs.append(add_spatial_x(x, pair)); ys.append(residual)
        del pair, x, y, betas
        gc.collect()
    tx, ty = pd.concat(xs, ignore_index=True), np.concatenate(ys)
    config = ModelConfig(direct_estimators=500, n_jobs=20)
    tree_model = lgb.LGBMRegressor(**lgb_params(config, 500))
    tree_model.fit(
        tx, ty,
        eval_set=[(vx, truth - vx.anchor_tws.to_numpy(np.float32))],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)],
    )
    tree = vx.anchor_tws.to_numpy(np.float32) + tree_model.predict(vx)
    weights = np.array([BLEND_WEIGHTS[int(h)] for h in horizons], np.float32)
    tabular = weights * tree + (1 - weights) * ridge
    print("tree", rmse(truth, tree), "ridge", rmse(truth, ridge), "tabular", rmse(truth, tabular))

    # Rebuild U-Net location keys, then align on (map group, grid key).
    raw = np.load("artifacts_map_unet/map_unet_val_predictions.npz")
    plain, _, _ = load_data(Path("."), need_test=False)
    store = MapStore(plain, cutoff)
    uk, ug = [], []
    for ex in validation_examples(store):
        _, mask = store.field("Target", ex.current, 0.0)
        kk = np.flatnonzero(mask.ravel()).astype(np.int32)
        uk.append(kk); ug.append(np.full(len(kk), ex.group, np.int16))
    uframe = pd.DataFrame({
        "group": np.concatenate(ug), "key": np.concatenate(uk),
        "unet": raw["prediction"], "uy": raw["truth"],
    })
    aligned = pd.DataFrame({
        "group": groups, "key": keys, "truth": truth, "horizon": horizons,
        "tree": tree, "ridge": ridge, "tabular": tabular,
    }).merge(uframe, on=["group", "key"], validate="one_to_one")
    y = aligned.truth.to_numpy(); unet = aligned.unet.to_numpy(); tab = aligned.tabular.to_numpy()
    print("aligned n", len(aligned), "target check", np.max(np.abs(y - aligned.uy)))
    print("unet", rmse(y, unet), "tabular", rmse(y, tab))
    w = optimal_weight(y, unet, tab)
    print("optimal unet weight", w, "blend", rmse(y, w * unet + (1 - w) * tab))
    for sigma in (0.5, 0.75, 1.0, 1.5, 2.0):
        sm = smooth(tab, aligned.key.to_numpy(), aligned.group.to_numpy(), sigma)
        sw = optimal_weight(y, sm, tab)
        print("smooth", sigma, "raw-smoothed weight", sw, "score", rmse(y, sw * sm + (1-sw) * tab))
    print("by horizon")
    for horizon in range(1, 8):
        m = aligned.horizon.to_numpy() == horizon
        wh = optimal_weight(y[m], unet[m], tab[m])
        print(horizon, "n", m.sum(), "tab", rmse(y[m], tab[m]), "unet", rmse(y[m], unet[m]), "w", wh, "blend", rmse(y[m], wh*unet[m]+(1-wh)*tab[m]))
    aligned.to_pickle("artifacts_map_unet/aligned_validation.pkl")


if __name__ == "__main__":
    main()
