#!/usr/bin/env python3
"""Calibrate submission map means with a legal low-dimensional forecast head."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_solution import observed_anchor_months
from map_blend_experiment import smooth
from map_unet_solution import VAL_BLOCKS, midx
from winning_solution import CLIMATE, load_data, rmse


SUMMARY_COLUMNS = ["TWS_t", *CLIMATE]
LAGS = [1, 2, 3, 6, 12]


def moments(values: pd.Series) -> np.ndarray:
    raw = values.dropna().to_numpy(np.float64)
    return np.asarray([
        raw.mean(), raw.std(), *np.quantile(raw, [.1, .5, .9])
    ], np.float64)


def build_summaries(frame: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    climate, tws = {}, {}
    for month, rows in frame.groupby("month_idx", sort=False):
        month = int(month)
        climate[month] = np.concatenate([moments(rows[column]) for column in CLIMATE])
        if rows["TWS_t"].notna().mean() > .5:
            tws[month] = moments(rows["TWS_t"])
    return climate, tws


def map_feature(
    anchor: int, current: int, horizon: int,
    climate: dict[int, np.ndarray], tws: dict[int, np.ndarray],
    time_mean: float, time_std: float,
) -> np.ndarray:
    anchor_tws = tws[anchor]
    path = [climate[month] for month in range(anchor, current + 1) if month in climate]
    values = [*anchor_tws, *climate[current], *climate[anchor], *np.mean(path, axis=0)]
    for lag in LAGS:
        values.extend(tws.get(anchor - lag, anchor_tws))
    target_month = (current + 1) % 12
    values.extend([
        np.sin(2 * np.pi * target_month / 12),
        np.cos(2 * np.pi * target_month / 12),
        (current - time_mean) / time_std,
        float(horizon), float(horizon * horizon),
    ])
    return np.asarray(values, np.float64)


def training_maps(
    train: pd.DataFrame, climate: dict[int, np.ndarray], tws: dict[int, np.ndarray],
    before: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    times = np.sort(train.loc[
        train.month_idx < before if before is not None else np.ones(len(train), bool),
        "month_idx",
    ].unique()).astype(np.int32)
    time_mean, time_std = float(times.mean()), float(times.std())
    target_mean = train.groupby("month_idx").Target.mean().to_dict()
    x, y = [], []
    for horizon in range(1, 8):
        for current in times:
            anchor = int(current) - horizon + 1
            if anchor in tws:
                x.append(map_feature(
                    anchor, int(current), horizon, climate, tws, time_mean, time_std
                ))
                y.append(target_mean[int(current)])
    return np.stack(x), np.asarray(y), time_mean, time_std


def fit_head(x: np.ndarray, y: np.ndarray):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x, y)


def validate(train: pd.DataFrame, strength: float) -> None:
    climate, tws = build_summaries(train)
    cutoff = midx(VAL_BLOCKS[0][0])
    x, y, time_mean, time_std = training_maps(train, climate, tws, cutoff)
    model = fit_head(x, y)
    aligned = pd.read_pickle("artifacts_map_unet/aligned_validation.pkl")
    truth = aligned.truth.to_numpy(np.float64)
    groups = aligned.group.to_numpy(np.int16)
    keys = aligned.key.to_numpy(np.int32)
    base = .55 * smooth(aligned.tabular.to_numpy(), keys, groups, 2.0)
    base += .45 * smooth(aligned.unet.to_numpy(), keys, groups, 1.0)

    predictions = []
    for date, max_horizon in VAL_BLOCKS:
        anchor = midx(date)
        for horizon in range(1, max_horizon + 1):
            predictions.append(model.predict(map_feature(
                anchor, anchor + horizon - 1, horizon, climate, tws,
                time_mean, time_std,
            )[None])[0])
    predictions = np.asarray(predictions)
    base_means = np.asarray([base[groups == group].mean() for group in range(len(predictions))])
    correction = (predictions - base_means)[groups]
    output = base + strength * correction
    print(f"base={rmse(truth, base):.9f} calibrated={rmse(truth, output):.9f}")
    print("predicted map means:", np.round(predictions, 4).tolist())


def predict(train: pd.DataFrame, test: pd.DataFrame, sample_path: Path,
            base_path: Path, output_path: Path, strength: float) -> None:
    climate_train, tws_train = build_summaries(train)
    climate_test, tws_test = build_summaries(test)
    climate = {**climate_train, **climate_test}
    tws = {**tws_train, **tws_test}
    x, y, time_mean, time_std = training_maps(train, climate_train, tws_train, None)
    model = fit_head(x, y)
    anchors = observed_anchor_months(test)
    month_to_anchor = {
        month: max(anchor for anchor in anchors if anchor <= month)
        for month in sorted(climate_test)
    }
    predicted_mean = {}
    for month, anchor in month_to_anchor.items():
        horizon = min(7, month + 1 - anchor)
        predicted_mean[month] = float(model.predict(map_feature(
            anchor, month, horizon, climate, tws, time_mean, time_std
        )[None])[0])

    sample = pd.read_csv(sample_path)[["ID"]]
    base = sample.merge(pd.read_csv(base_path), on="ID", validate="one_to_one")
    lookup = test[["ID", "month_idx"]]
    frame = base.merge(lookup, on="ID", validate="one_to_one")
    base_map_mean = frame.groupby("month_idx").Target.mean().to_dict()
    correction = frame.month_idx.map(
        {month: predicted_mean[month] - base_map_mean[month] for month in predicted_mean}
    ).to_numpy(np.float64)
    submission = frame[["ID"]].copy()
    submission["Target"] = frame.Target.to_numpy(np.float64) + strength * correction
    submission = sample.merge(submission, on="ID", validate="one_to_one")
    if len(submission) != len(sample) or not np.isfinite(submission.Target).all():
        raise ValueError("invalid map-calibrated submission")
    submission.to_csv(output_path, index=False)
    print("predicted test map means:", {
        f"{month // 12:04d}-{month % 12 + 1:02d}": round(value, 5)
        for month, value in predicted_mean.items()
    })
    print(f"saved {output_path}: {len(submission):,} rows; strength={strength}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "predict"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--base", type=Path, default=Path("Submission_V13_WalkForwardTriBlend.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V14_MapMeanCalibrated.csv"))
    parser.add_argument("--strength", type=float, default=.5)
    args = parser.parse_args()
    train, test, sample_path = load_data(args.data_dir, need_test=args.command == "predict")
    if args.command == "validate":
        validate(train, args.strength)
    else:
        assert test is not None and sample_path is not None
        predict(train, test, sample_path, args.base, args.output, args.strength)


if __name__ == "__main__":
    main()
