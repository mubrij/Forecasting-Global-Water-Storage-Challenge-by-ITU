#!/usr/bin/env python3
"""Retrain matched baseline/advanced models and build a gated V15 correction."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from advanced_features_experiment import add_advanced_features, build_training
from ensemble_solution import ANCHOR_COLUMNS, observed_anchor_months
from history_experiment import add_history
from spatial_experiment import add_spatial_x, augment
from winning_solution import (
    CLIMATE, ModelConfig, build_feature_frame, fit_history_stats, lgb_params,
    load_data, seed_everything,
)


GATED_HORIZONS = (4, 5, 7)


def fit_model(x, y, iterations: int, n_jobs: int):
    config = ModelConfig(
        direct_estimators=iterations, learning_rate=.03, num_leaves=72,
        min_child_samples=400, n_jobs=n_jobs,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, iterations))
    model.fit(x, y, callbacks=[lgb.log_evaluation(30)])
    return model


def test_feature_frames(train, test, stats):
    test = augment(test).copy()
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    anchors = observed_anchor_months(test)
    by_month = {int(month): rows for month, rows in test.groupby("month_idx", sort=False)}
    observed_test = test[test.TWS_t.notna()][["month_idx", "loc_key", "TWS_t"]]
    history = pd.concat([
        train[["month_idx", "loc_key", "TWS_t"]], observed_test
    ], ignore_index=True).drop_duplicates(["month_idx", "loc_key"], keep="last")
    baseline_parts, advanced_parts, metadata = [], [], []

    for group, month in enumerate(sorted(by_month)):
        current = by_month[month].copy()
        anchor_month = max(anchor for anchor in anchors if anchor <= month)
        anchor = by_month[anchor_month][["loc_key", *ANCHOR_COLUMNS]].rename(
            columns={column: f"anchor_{column}" for column in ANCHOR_COLUMNS}
        )
        pair = current.merge(anchor, on="loc_key", how="left", validate="one_to_one")
        horizon = np.full(len(pair), min(7, month + 1 - anchor_month), np.int8)
        observed = pair.TWS_t.notna().to_numpy()
        for column in ANCHOR_COLUMNS:
            pair.loc[observed, f"anchor_{column}"] = pair.loc[observed, column]
        horizon[observed] = 1
        anchor_frame = pd.DataFrame({"TWS_t": pair.anchor_TWS_t.to_numpy()})
        for column in CLIMATE:
            anchor_frame[column] = pair[f"anchor_{column}"].to_numpy()
        base = build_feature_frame(pair, stats, anchor=anchor_frame, horizon=horizon)
        base = add_spatial_x(base, pair)
        advanced = pd.DataFrame(index=pair.index)
        for value in np.unique(horizon):
            take = np.flatnonzero(horizon == value)
            part_pair = pair.iloc[take].reset_index(drop=True)
            part_x = base.iloc[take].reset_index(drop=True)
            part_x = add_history(part_x, part_pair, history, int(value))
            part_x = add_advanced_features(part_x, part_pair, stats, int(value))
            advanced = advanced.reindex(columns=part_x.columns)
            advanced.iloc[take] = part_x.to_numpy()
        baseline_parts.append(base)
        advanced_parts.append(advanced)
        metadata.append(pd.DataFrame({
            "_row_id": pair._row_id.to_numpy(), "ID": pair.ID.to_numpy(),
            "group": np.full(len(pair), group, np.int16), "horizon": horizon,
        }))
    order = pd.concat(metadata, ignore_index=True).sort_values("_row_id").index
    baseline = pd.concat(baseline_parts, ignore_index=True).iloc[order].reset_index(drop=True)
    advanced = pd.concat(advanced_parts, ignore_index=True).iloc[order].reset_index(drop=True)
    meta = pd.concat(metadata, ignore_index=True).iloc[order].reset_index(drop=True)
    return baseline, advanced, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_advanced_production"))
    parser.add_argument("--base", type=Path, default=Path("Submission_V15_MapMeanProbe.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V20_GatedHistoryCorrection.csv"))
    parser.add_argument("--rows-per-horizon", type=int, default=80000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--weight", type=float, default=.5)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    seed_everything()
    train, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    train = augment(train)
    stats = fit_history_stats(train)
    bx, by = build_training(train, stats, False, args.rows_per_horizon, False)
    ax, ay = build_training(train, stats, True, args.rows_per_horizon, False)
    baseline_model = fit_model(bx, by, args.iterations, args.n_jobs)
    advanced_model = fit_model(ax, ay, args.iterations, args.n_jobs)
    bx_test, ax_test, meta = test_feature_frames(train, test, stats)
    baseline = bx_test.anchor_tws.to_numpy(np.float64) + baseline_model.predict(bx_test)
    advanced = ax_test.anchor_tws.to_numpy(np.float64) + advanced_model.predict(ax_test)
    correction = advanced - baseline
    gate = np.isin(meta.horizon.to_numpy(), GATED_HORIZONS)
    correction[~gate] = 0.0
    # Preserve V15's successful map-mean calibration exactly. Center only the
    # gated rows so sparse horizon-1 cells inside a rollout map remain untouched.
    gated_mean = (
        pd.Series(np.where(gate, correction, np.nan))
        .groupby(meta.group)
        .transform("mean")
        .to_numpy()
    )
    correction[gate] -= gated_mean[gate]

    sample = pd.read_csv(sample_path)[["ID"]]
    base = sample.merge(pd.read_csv(args.base), on="ID", validate="one_to_one")
    output = sample.copy()
    output["Target"] = base.Target.to_numpy(np.float64) + args.weight * correction
    if len(output) != len(sample) or not output.ID.is_unique or not np.isfinite(output.Target).all():
        raise ValueError("invalid advanced-feature submission")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": sample.ID, "Target": baseline}).to_csv(
        args.artifact_dir / "Submission_MatchedBaseline.csv", index=False
    )
    pd.DataFrame({"ID": sample.ID, "Target": advanced}).to_csv(
        args.artifact_dir / "Submission_AdvancedHistory.csv", index=False
    )
    joblib.dump({"baseline": baseline_model, "advanced": advanced_model,
                 "baseline_features": list(bx.columns), "advanced_features": list(ax.columns)},
                args.artifact_dir / "models.joblib", compress=3)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output}: {len(output):,} rows")
    print(f"gated rows={gate.sum():,}; weight={args.weight}; correction_rms={np.sqrt(np.mean(correction**2)):.9f}")


if __name__ == "__main__":
    main()
