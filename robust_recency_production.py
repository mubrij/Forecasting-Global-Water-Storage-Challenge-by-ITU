#!/usr/bin/env python3
"""Train the robust recency L1 model and blend its centered correction into V15."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from recency_regime_experiment import training_set
from recency_regime_production import test_features
from spatial_experiment import augment
from winning_solution import ModelConfig, fit_history_stats, lgb_params, load_data, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--uniform-artifact", type=Path,
                        default=Path("artifacts_recency_production/models.joblib"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_robust_recency_production"))
    parser.add_argument("--base", type=Path, default=Path("Submission_V15_MapMeanProbe.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V22_RobustRecencyBlend.csv"))
    parser.add_argument("--rows-per-horizon", type=int, default=160000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--weight", type=float, default=.5)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    seed_everything()
    train, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    train = augment(train)
    stats = fit_history_stats(train)
    tx, ty, months = training_set(train, stats, args.rows_per_horizon)
    recent_weight = np.power(0.5, months.max() - months)
    recent_weight /= recent_weight.mean()
    config = ModelConfig(direct_estimators=args.iterations, learning_rate=.03,
                         num_leaves=72, min_child_samples=400, n_jobs=args.n_jobs)
    params = lgb_params(config, args.iterations)
    params.update(objective="regression_l1", metric="l2")
    model = lgb.LGBMRegressor(**params)
    model.fit(tx, ty, sample_weight=recent_weight, callbacks=[lgb.log_evaluation(30)])

    uniform = joblib.load(args.uniform_artifact)["uniform"]
    test_x, meta = test_features(test, stats)
    anchor = test_x.anchor_tws.to_numpy(np.float64)
    uniform_prediction = anchor + uniform.predict(test_x)
    l1_prediction = anchor + model.predict(test_x)
    correction = l1_prediction - uniform_prediction
    correction -= pd.Series(correction).groupby(meta.group).transform("mean").to_numpy()
    sample = pd.read_csv(sample_path)[["ID"]]
    base = sample.merge(pd.read_csv(args.base), on="ID", validate="one_to_one")
    output = sample.copy()
    output["Target"] = base.Target.to_numpy(np.float64) + args.weight * correction
    if len(output) != len(sample) or not output.ID.is_unique or not np.isfinite(output.Target).all():
        raise ValueError("invalid robust recency submission")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": sample.ID, "Target": l1_prediction}).to_csv(
        args.artifact_dir / "Submission_RecentL1Standalone.csv", index=False
    )
    joblib.dump({"model": model, "features": list(tx.columns)},
                args.artifact_dir / "model.joblib", compress=3)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output}: {len(output):,} rows")
    print(f"weight={args.weight}; correction_rms={np.sqrt(np.mean(correction**2)):.9f}")


if __name__ == "__main__":
    main()
