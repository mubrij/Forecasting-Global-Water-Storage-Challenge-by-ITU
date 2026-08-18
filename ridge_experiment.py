#!/usr/bin/env python3
"""Fast experiment for hierarchical per-grid-cell direct forecasts."""

from pathlib import Path
import math
import numpy as np
import pandas as pd

from winning_solution import CLIMATE, load_data, validation_pairs, fit_history_stats, rmse


RIDGE_COLS = ["intercept", "anchor_tws", *[f"current_{c}" for c in CLIMATE],
              "target_month_sin", "target_month_cos"]


def ridge_matrix(x: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(x), np.float32),
        x["anchor_tws"].to_numpy(np.float32),
        *[x[f"current_{c}"].to_numpy(np.float32) for c in CLIMATE],
        x["target_month_sin"].to_numpy(np.float32),
        x["target_month_cos"].to_numpy(np.float32),
    ]).astype(np.float64)


def global_beta(x: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0.01
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def fit_local(pair: pd.DataFrame, x_features: pd.DataFrame, y: np.ndarray,
              max_key: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    x = ridge_matrix(x_features)
    prior = global_beta(x, y)
    betas = np.broadcast_to(prior, (max_key + 1, len(prior))).copy()
    counts = np.zeros(max_key + 1, np.int16)
    keys = pair["loc_key"].to_numpy(np.int32)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    ends = np.r_[starts[1:], len(order)]
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = lam * 0.05
    for start, end in zip(starts, ends):
        idx = order[start:end]
        key = sorted_keys[start]
        xx = x[idx]
        yy = y[idx]
        betas[key] = np.linalg.solve(xx.T @ xx + penalty, xx.T @ yy + penalty @ prior)
        counts[key] = len(idx)
    return betas, counts


def main() -> None:
    train, _, _ = load_data(Path('.'), need_test=False)
    cutoff = pd.Timestamp("2012-07-01")
    fit = train[train.time < cutoff].copy()
    stats = fit_history_stats(fit)
    anchors = [("2012-07-01", 3), ("2012-12-01", 3), ("2013-05-01", 3),
               ("2013-11-01", 3), ("2014-04-01", 3), ("2014-09-01", 7)]
    val_x, _, y_val, val_h, _ = validation_pairs(train, stats, anchors)
    # Recover validation keys by reconstructing in the same anchor/horizon order.
    val_keys = []
    by_month = {int(k): v for k, v in train.groupby('month_idx', sort=False)}
    for date, max_h in anchors:
        ai = pd.Timestamp(date).year * 12 + pd.Timestamp(date).month - 1
        akeys = by_month[ai][['loc_key']]
        for h in range(1, max_h + 1):
            cur = by_month.get(ai + h - 1)
            if cur is None:
                break
            val_keys.append(cur[['loc_key']].merge(akeys, on='loc_key', how='inner')['loc_key'].to_numpy(np.int32))
    val_keys = np.concatenate(val_keys)

    for lam in [3.0, 10.0, 30.0, 100.0]:
        pred = np.empty(len(y_val), np.float64)
        for h in range(1, 8):
            from winning_solution import make_pair_frame, pair_to_xy
            pair = make_pair_frame(fit, h)
            x_train, _, y_train = pair_to_xy(pair, stats, h)
            betas, _ = fit_local(pair, x_train, y_train.astype(np.float64), stats.max_loc_key, lam)
            mask = val_h == h
            xv = ridge_matrix(val_x.loc[mask])
            pred[mask] = np.einsum('ij,ij->i', xv, betas[val_keys[mask]])
        print('lambda', lam, 'overall', rmse(y_val, pred))
        for h in range(1, 8):
            m = val_h == h
            print(h, rmse(y_val[m], pred[m]))


if __name__ == '__main__':
    main()
