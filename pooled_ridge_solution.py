#!/usr/bin/env python3
"""Pooled global ridge forecast with spatial smoothing.

The original ensemble fits a separate ridge coefficient vector at every grid
cell. Chronological validation shows that those local coefficients are too
variable for the short 2002--2015 record. This component uses one coefficient
vector per forecast horizon, then restores spatial coherence with a legal
Gaussian map smoother.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_solution import ANCHOR_COLUMNS, observed_anchor_months
from map_blend_experiment import smooth
from ridge_experiment import global_beta, ridge_matrix
from spatial_experiment import augment, make_pair
from winning_solution import (
    CLIMATE, build_feature_frame, fit_history_stats, load_data, pair_to_xy,
)


def fit_models(frame: pd.DataFrame, stats) -> dict[int, np.ndarray]:
    models = {}
    for horizon in range(1, 8):
        pair = make_pair(frame, horizon)
        x, _, target = pair_to_xy(pair, stats, horizon)
        models[horizon] = global_beta(
            ridge_matrix(x), target.astype(np.float64), lam=10.0
        )
        print(f"h={horizon}: {len(pair):,} pooled observations", flush=True)
    return models


def test_frame(test: pd.DataFrame, stats, models: dict[int, np.ndarray]) -> pd.DataFrame:
    test = augment(test).copy()
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    anchors = observed_anchor_months(test)
    by_month = {int(key): value for key, value in test.groupby("month_idx", sort=False)}
    outputs = []
    for map_group, month in enumerate(sorted(by_month)):
        current = by_month[month].copy()
        anchor_month = max(value for value in anchors if value <= month)
        anchor = by_month[anchor_month][["loc_key", *ANCHOR_COLUMNS]].rename(
            columns={column: f"anchor_{column}" for column in ANCHOR_COLUMNS}
        )
        pair = current.merge(anchor, on="loc_key", how="left", validate="one_to_one")
        horizon = np.full(len(pair), min(7, month + 1 - anchor_month), np.int8)
        observed = np.isfinite(pair["TWS_t"].to_numpy(np.float32))
        for column in ANCHOR_COLUMNS:
            pair.loc[observed, f"anchor_{column}"] = pair.loc[observed, column]
        horizon[observed] = 1

        anchor_frame = pd.DataFrame({"TWS_t": pair["anchor_TWS_t"].to_numpy()})
        for column in CLIMATE:
            anchor_frame[column] = pair[f"anchor_{column}"].to_numpy()
        x = build_feature_frame(pair, stats, anchor=anchor_frame, horizon=horizon)
        prediction = np.empty(len(pair), np.float64)
        for value in range(1, 8):
            take = horizon == value
            if take.any():
                prediction[take] = ridge_matrix(x.loc[take]) @ models[value]
        outputs.append(pd.DataFrame({
            "_row_id": pair["_row_id"].to_numpy(), "ID": pair["ID"].to_numpy(),
            "Target": prediction, "key": pair["loc_key"].to_numpy(np.int32),
            "group": np.full(len(pair), map_group, np.int16),
        }))
    return pd.concat(outputs, ignore_index=True).sort_values("_row_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("Submission_GlobalRidge_Smoothed.csv"))
    parser.add_argument("--sigma", type=float, default=1.5)
    args = parser.parse_args()

    train, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    train = augment(train)
    stats = fit_history_stats(train)
    models = fit_models(train, stats)
    raw = test_frame(test, stats, models)
    raw["Target"] = smooth(
        raw["Target"].to_numpy(), raw["key"].to_numpy(),
        raw["group"].to_numpy(), args.sigma,
    )
    raw["Target"] = np.clip(raw["Target"], stats.target_min - .25, stats.target_max + .25)
    sample = pd.read_csv(sample_path)[["ID"]]
    submission = sample.merge(raw[["ID", "Target"]], on="ID", validate="one_to_one")
    if len(submission) != len(sample) or not np.isfinite(submission.Target).all():
        raise ValueError("invalid pooled-ridge submission")
    submission.to_csv(args.output, index=False)
    print(f"saved {args.output}: {len(submission):,} rows; sigma={args.sigma}")


if __name__ == "__main__":
    main()
