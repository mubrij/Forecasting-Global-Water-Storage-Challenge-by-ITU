#!/usr/bin/env python3
"""Train/predict the leakage-safe previous-anchor momentum TWS model."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ensemble_solution import ANCHOR_COLUMNS, observed_anchor_months
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    CLIMATE, SEED, ModelConfig, build_feature_frame, fit_history_stats,
    lgb_params, load_data, pair_to_xy, seed_everything,
)

MOMENTUM_FEATURES = [
    "previous_anchor_tws", "previous_anchor_gap", "anchor_velocity",
    "anchor_change", "momentum_horizon",
]


def add_previous(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["loc_key", "month_idx"]).copy()
    grouped = out.groupby("loc_key", sort=False)
    out["prev_tws"] = grouped["TWS_t"].shift()
    out["prev_month"] = grouped["month_idx"].shift()
    out["prev_gap"] = out["month_idx"] - out["prev_month"]
    out["velocity"] = (out["TWS_t"] - out["prev_tws"]) / out["prev_gap"]
    return out.sort_index()


def momentum_pair(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pair = make_pair(frame, horizon)
    previous = frame[
        ["month_idx", "loc_key", "prev_tws", "prev_gap", "velocity"]
    ].copy()
    previous["month_idx"] += horizon - 1
    return pair.merge(
        previous, on=["month_idx", "loc_key"], how="left", validate="one_to_one"
    )


def add_momentum_features(x: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    anchor = out["anchor_tws"].to_numpy(np.float32)
    previous = pair["prev_tws"].to_numpy(np.float32)
    gap = pair["prev_gap"].to_numpy(np.float32)
    missing = ~np.isfinite(previous)
    previous = np.where(missing, anchor, previous)
    gap = np.where(np.isfinite(gap), np.clip(gap, 1, 24), 12).astype(np.float32)
    velocity = np.where(missing, 0.0, (anchor - previous) / gap).astype(np.float32)
    out["previous_anchor_tws"] = previous
    out["previous_anchor_gap"] = gap
    out["anchor_velocity"] = velocity
    out["anchor_change"] = np.where(missing, 0.0, anchor - previous)
    out["momentum_horizon"] = velocity * out["horizon"].to_numpy(np.float32)
    return out


def train(data_dir: Path, artifact_dir: Path, rows_per_horizon: int, estimators: int,
          n_jobs: int) -> None:
    frame, _, _ = load_data(data_dir, need_test=False)
    frame = augment(add_previous(frame))
    stats = fit_history_stats(frame)
    xs, ys = [], []
    for horizon in range(1, 8):
        pair = momentum_pair(frame, horizon)
        if len(pair) > rows_per_horizon:
            pair = pair.sample(rows_per_horizon, random_state=SEED + horizon)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        x = add_momentum_features(add_spatial_x(x, pair), pair)
        xs.append(x)
        ys.append(residual)
        print(f"h={horizon}: {len(pair):,} rows", flush=True)
        del pair
        gc.collect()
    x = pd.concat(xs, ignore_index=True)
    y = np.concatenate(ys)
    config = ModelConfig(n_jobs=n_jobs)
    model = lgb.LGBMRegressor(**lgb_params(config, estimators))
    model.fit(x, y, callbacks=[lgb.log_evaluation(20)])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"version": "momentum-v1", "stats": stats, "model": model,
         "features": x.columns.tolist()},
        artifact_dir / "momentum_model.joblib", compress=3,
    )


def test_observation_history(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    historical = train[["ID", "loc_key", "month_idx", "TWS_t"]]
    observed_test = test.loc[
        test["TWS_t"].notna(), ["ID", "loc_key", "month_idx", "TWS_t"]
    ]
    history = pd.concat([historical, observed_test], ignore_index=True)
    history = history.sort_values(["loc_key", "month_idx"])
    grouped = history.groupby("loc_key", sort=False)
    history["prev_tws"] = grouped["TWS_t"].shift()
    history["prev_month"] = grouped["month_idx"].shift()
    history["prev_gap"] = history["month_idx"] - history["prev_month"]
    history["velocity"] = (history["TWS_t"] - history["prev_tws"]) / history["prev_gap"]
    return history.loc[history["ID"].isin(test["ID"]), [
        "ID", "prev_tws", "prev_month", "prev_gap", "velocity"
    ]]


def predict(data_dir: Path, artifact_dir: Path, output: Path) -> None:
    bundle = joblib.load(artifact_dir / "momentum_model.joblib")
    train_frame, test, sample_path = load_data(data_dir, need_test=True)
    assert test is not None and sample_path is not None
    observation_features = test_observation_history(train_frame, test)
    test = augment(test).merge(observation_features, on="ID", how="left")
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    anchors = observed_anchor_months(test)
    by_month = {int(key): value for key, value in test.groupby("month_idx", sort=False)}
    outputs = []
    previous_columns = ["prev_tws", "prev_month", "prev_gap", "velocity"]

    for month in sorted(by_month):
        current = by_month[month].copy()
        anchor_month = max(value for value in anchors if value <= month)
        anchor = by_month[anchor_month][
            ["loc_key", *ANCHOR_COLUMNS, *previous_columns]
        ].rename(columns={
            **{column: f"anchor_{column}" for column in ANCHOR_COLUMNS},
            **{column: f"block_{column}" for column in previous_columns},
        })
        pair = current.merge(anchor, on="loc_key", how="left", validate="one_to_one")
        horizon = np.full(len(pair), min(7, month + 1 - anchor_month), np.int8)
        available = pair["TWS_t"].notna().to_numpy()
        for column in ANCHOR_COLUMNS:
            pair.loc[available, f"anchor_{column}"] = pair.loc[available, column]
        for column in previous_columns:
            pair.loc[available, f"block_{column}"] = pair.loc[available, column]
        horizon[available] = 1
        pair["prev_tws"] = pair["block_prev_tws"]
        pair["prev_gap"] = pair["block_prev_gap"]
        pair["velocity"] = pair["block_velocity"]

        anchor_frame = pd.DataFrame({"TWS_t": pair["anchor_TWS_t"].to_numpy()})
        for column in CLIMATE:
            anchor_frame[column] = pair[f"anchor_{column}"].to_numpy()
        x = build_feature_frame(pair, bundle["stats"], anchor=anchor_frame, horizon=horizon)
        x = add_momentum_features(add_spatial_x(x, pair), pair)
        prediction = x["anchor_tws"].to_numpy() + bundle["model"].predict(
            x[bundle["features"]]
        )
        outputs.append(pd.DataFrame({
            "_row_id": pair["_row_id"], "ID": pair["ID"],
            "Target": prediction.astype(np.float32), "horizon": horizon,
        }))
    raw = pd.concat(outputs).sort_values("_row_id")
    sample = pd.read_csv(sample_path)
    submission = sample[["ID"]].merge(raw[["ID", "Target"]], on="ID", validate="one_to_one")
    assert len(submission) == len(sample) and np.isfinite(submission["Target"]).all()
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Saved {len(submission):,} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_momentum"))
    parser.add_argument("--output", type=Path, default=Path("Submission_Momentum.csv"))
    parser.add_argument("--rows-per-horizon", type=int, default=240_000)
    parser.add_argument("--estimators", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    seed_everything()
    if args.command in {"train", "all"}:
        train(args.data_dir, args.artifact_dir, args.rows_per_horizon, args.estimators, args.n_jobs)
    if args.command in {"predict", "all"}:
        predict(args.data_dir, args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
