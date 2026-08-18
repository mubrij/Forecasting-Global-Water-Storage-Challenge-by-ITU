#!/usr/bin/env python3
"""Chronological benchmark for legal within-rollout climate sequence features.

For a horizon-h forecast the test data exposes climate maps for every month from
the observed TWS anchor through the current predictor month.  Earlier models only
used the two endpoints.  This experiment summarizes the complete legal path and
checks the gain on test-shaped held-out anchor blocks.
"""
from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    CLIMATE,
    SEED,
    ModelConfig,
    fit_history_stats,
    lgb_params,
    load_data,
    pair_to_xy,
    rmse,
)


def add_sequence_features(
    x: pd.DataFrame, pair: pd.DataFrame, frame: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Add summaries for anchor..current climate, returning complete-path mask."""
    out = x.copy()
    n = len(pair)
    # Endpoints are already carried by pair.  Join exact intermediate calendar
    # months; examples crossing a missing source month are excluded from fitting.
    stacks: dict[str, list[np.ndarray]] = {
        c: [pair[f"anchor_{c}"].to_numpy(np.float32)] for c in CLIMATE
    }
    complete = np.ones(n, dtype=bool)
    if horizon > 1:
        for lag in range(horizon - 2, 0, -1):
            middle = frame[["month_idx", "loc_key", *CLIMATE]].copy()
            middle["month_idx"] += lag
            rename = {c: f"_seq_{lag}_{c}" for c in CLIMATE}
            middle = middle.rename(columns=rename)
            pair = pair.merge(
                middle, on=["month_idx", "loc_key"], how="left", validate="one_to_one"
            )
            for c in CLIMATE:
                values = pair[rename[c]].to_numpy(np.float32)
                complete &= np.isfinite(values)
                stacks[c].append(values)
        for c in CLIMATE:
            stacks[c].append(pair[c].to_numpy(np.float32))

    for c, parts in stacks.items():
        values = np.column_stack(parts).astype(np.float32)
        missing = ~np.isfinite(values)
        row_mean = np.nanmean(values, axis=1).astype(np.float32)
        values = np.where(missing, row_mean[:, None], values).astype(np.float32)
        out[f"path_missing_{c}"] = missing.mean(axis=1).astype(np.float32)
        out[f"path_mean_{c}"] = values.mean(axis=1).astype(np.float32)
        out[f"path_std_{c}"] = np.nanstd(values, axis=1).astype(np.float32)
        out[f"path_min_{c}"] = np.nanmin(values, axis=1).astype(np.float32)
        out[f"path_max_{c}"] = np.nanmax(values, axis=1).astype(np.float32)
        if values.shape[1] > 1:
            weights = np.arange(values.shape[1], dtype=np.float32)
            weights -= weights.mean()
            denom = float(np.dot(weights, weights))
            out[f"path_slope_{c}"] = (values @ weights / denom).astype(np.float32)
        else:
            out[f"path_slope_{c}"] = np.zeros(n, np.float32)
    return out, complete


def build_validation(frame: pd.DataFrame, stats, anchors):
    xs, ys, truths, horizons, maps = [], [], [], [], []
    by_month = {int(k): v for k, v in frame.groupby("month_idx", sort=False)}
    map_id = 0
    for date, max_h in anchors:
        anchor_idx = pd.Timestamp(date).year * 12 + pd.Timestamp(date).month - 1
        for horizon in range(1, max_h + 1):
            current_idx = anchor_idx + horizon - 1
            if current_idx not in by_month:
                break
            pair = make_pair(frame, horizon)
            pair = pair[pair["month_idx"] == current_idx].reset_index(drop=True)
            x, residual, target = pair_to_xy(pair, stats, horizon)
            x = add_spatial_x(x, pair)
            x, complete = add_sequence_features(x, pair, frame, horizon)
            xs.append(x)
            ys.append(residual)
            truths.append(target)
            horizons.append(np.full(len(pair), horizon, np.int8))
            maps.append(np.full(len(pair), map_id, np.int16))
            map_id += 1
    return (
        pd.concat(xs, ignore_index=True), np.concatenate(ys),
        np.concatenate(truths), np.concatenate(horizons), np.concatenate(maps),
    )


def main() -> None:
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    anchors = [
        ("2012-07-01", 3), ("2012-12-01", 3), ("2013-05-01", 3),
        ("2013-11-01", 3), ("2014-04-01", 3), ("2014-09-01", 7),
    ]
    fit = frame[frame["time"] < pd.Timestamp(anchors[0][0])].copy()
    stats = fit_history_stats(fit)
    val_x, val_residual, truth, val_h, _ = build_validation(frame, stats, anchors)

    xs, ys = [], []
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        x = add_spatial_x(x, pair)
        x, complete = add_sequence_features(x, pair, fit, horizon)
        # Retain cells absent from an intermediate map after path imputation.
        if len(x) > 80_000:
            take = np.random.RandomState(SEED + horizon).choice(len(x), 80_000, False)
            x, residual = x.iloc[take], residual[take]
        xs.append(x)
        ys.append(residual)
        print(f"h={horizon}: {len(x):,} complete examples", flush=True)
        del pair
        gc.collect()

    train_x = pd.concat(xs, ignore_index=True)
    train_y = np.concatenate(ys)
    config = ModelConfig(
        direct_estimators=700, learning_rate=0.035, num_leaves=80,
        min_child_samples=300, n_jobs=20,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, config.direct_estimators))
    model.fit(
        train_x, train_y, eval_set=[(val_x, val_residual)],
        callbacks=[lgb.early_stopping(90), lgb.log_evaluation(50)],
    )
    prediction = val_x["anchor_tws"].to_numpy(np.float32) + model.predict(val_x)
    print("sequence overall", rmse(truth, prediction), flush=True)
    print(
        "by horizon",
        {h: rmse(truth[val_h == h], prediction[val_h == h]) for h in range(1, 8)},
        flush=True,
    )


if __name__ == "__main__":
    main()
