#!/usr/bin/env python3
"""Full-data production of the validated recency-regime correction."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ensemble_solution import ANCHOR_COLUMNS, observed_anchor_months
from recency_regime_experiment import training_set
from spatial_experiment import add_spatial_x, augment
from winning_solution import (
    CLIMATE, ModelConfig, build_feature_frame, fit_history_stats, lgb_params,
    load_data, seed_everything,
)


def fit_model(x, y, sample_weight, iterations, n_jobs):
    config = ModelConfig(direct_estimators=iterations, learning_rate=.03,
                         num_leaves=72, min_child_samples=400, n_jobs=n_jobs)
    model = lgb.LGBMRegressor(**lgb_params(config, iterations))
    model.fit(x, y, sample_weight=sample_weight, callbacks=[lgb.log_evaluation(30)])
    return model


def test_features(test, stats):
    test = augment(test).copy()
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    anchors = observed_anchor_months(test)
    by_month = {int(month): rows for month, rows in test.groupby("month_idx", sort=False)}
    features, metadata = [], []
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
        x = build_feature_frame(pair, stats, anchor=anchor_frame, horizon=horizon)
        features.append(add_spatial_x(x, pair))
        metadata.append(pd.DataFrame({
            "_row_id": pair._row_id.to_numpy(), "ID": pair.ID.to_numpy(),
            "group": np.full(len(pair), group, np.int16), "horizon": horizon,
        }))
    x = pd.concat(features, ignore_index=True)
    meta = pd.concat(metadata, ignore_index=True)
    order = np.argsort(meta._row_id.to_numpy(), kind="stable")
    return x.iloc[order].reset_index(drop=True), meta.iloc[order].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_recency_production"))
    parser.add_argument("--base", type=Path, default=Path("Submission_V15_MapMeanProbe.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V21_RecencyRegimeBlend.csv"))
    parser.add_argument("--rows-per-horizon", type=int, default=160000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--half-life", type=float, default=1.0)
    parser.add_argument("--weight", type=float, default=.4)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    seed_everything()
    train, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    train = augment(train)
    stats = fit_history_stats(train)
    tx, ty, months = training_set(train, stats, args.rows_per_horizon)
    uniform_model = fit_model(tx, ty, None, args.iterations, args.n_jobs)
    recent_weight = np.power(0.5, (months.max() - months) / args.half_life)
    recent_weight /= recent_weight.mean()
    recent_model = fit_model(tx, ty, recent_weight, args.iterations, args.n_jobs)
    test_x, meta = test_features(test, stats)
    anchor = test_x.anchor_tws.to_numpy(np.float64)
    uniform = anchor + uniform_model.predict(test_x)
    recent = anchor + recent_model.predict(test_x)
    correction = recent - uniform
    correction -= pd.Series(correction).groupby(meta.group).transform("mean").to_numpy()

    sample = pd.read_csv(sample_path)[["ID"]]
    base = sample.merge(pd.read_csv(args.base), on="ID", validate="one_to_one")
    output = sample.copy()
    output["Target"] = base.Target.to_numpy(np.float64) + args.weight * correction
    if len(output) != len(sample) or not output.ID.is_unique or not np.isfinite(output.Target).all():
        raise ValueError("invalid recency-regime submission")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": sample.ID, "Target": uniform}).to_csv(
        args.artifact_dir / "Submission_UniformMatched.csv", index=False
    )
    pd.DataFrame({"ID": sample.ID, "Target": recent}).to_csv(
        args.artifact_dir / "Submission_RecentStandalone.csv", index=False
    )
    joblib.dump({"uniform": uniform_model, "recent": recent_model,
                 "features": list(tx.columns), "half_life": args.half_life},
                args.artifact_dir / "models.joblib", compress=3)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output}: {len(output):,} rows")
    print(f"weight={args.weight}; half_life={args.half_life}; correction_rms={np.sqrt(np.mean(correction**2)):.9f}")


if __name__ == "__main__":
    main()
