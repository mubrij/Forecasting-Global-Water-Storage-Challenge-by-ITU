#!/usr/bin/env python3
"""Production spatial-tree + hierarchical-ridge ensemble for global TWS.

Examples
--------
python ensemble_solution.py train --data-dir . --artifact-dir artifacts_final
python ensemble_solution.py predict --data-dir . --artifact-dir artifacts_final \
    --output Submission_Ensemble_v1.csv
python ensemble_solution.py all --data-dir . --artifact-dir artifacts_final \
    --output Submission_Ensemble_v1.csv
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from winning_solution import (
    CLIMATE, DIRECT_FEATURES, SEED, ModelConfig, build_feature_frame,
    fit_history_stats, lgb_params, load_data, pair_to_xy, seed_everything,
)
from spatial_experiment import SPATIAL, add_spatial_x, augment, make_pair
from ridge_experiment import fit_local, ridge_matrix


# Raw OOF least-squares weights shrunk 20% toward an equal blend.  Horizons
# 1-3 have six validation maps; longer horizons are deliberately conservative.
BLEND_WEIGHTS = {
    1: 0.55532256, 2: 0.55548168, 3: 0.49333328, 4: 0.37235120,
    5: 0.74098128, 6: 0.24563552, 7: 0.43209168,
}
ANCHOR_COLUMNS = [
    "TWS_t", *CLIMATE,
    *[f"{column}_nbr{size}" for column in ["TWS_t", *CLIMATE] for size in (3, 7)],
]


def train(
    data_dir: Path,
    artifact_dir: Path,
    rows_per_horizon: int = 240_000,
    estimators: int = 36,
    n_jobs: int = -1,
) -> None:
    frame, _, _ = load_data(data_dir, need_test=False)
    frame = augment(frame)
    stats = fit_history_stats(frame)
    ridge_models: dict[int, np.ndarray] = {}
    tree_frames: list[pd.DataFrame] = []
    tree_targets: list[np.ndarray] = []

    for horizon in range(1, 8):
        pair = make_pair(frame, horizon)
        x_all, residual_all, y_all = pair_to_xy(pair, stats, horizon)
        ridge_models[horizon] = fit_local(
            pair, x_all, y_all.astype(np.float64), stats.max_loc_key, 100.0
        )[0].astype(np.float32)
        if len(pair) > rows_per_horizon:
            take = np.random.RandomState(SEED + horizon).choice(
                len(pair), rows_per_horizon, replace=False
            )
            pair = pair.iloc[take]
            x_all = x_all.iloc[take]
            residual_all = residual_all[take]
        tree_frames.append(add_spatial_x(x_all, pair))
        tree_targets.append(residual_all)
        print(
            f"h={horizon}: ridge={len(y_all):,}, tree={len(residual_all):,}",
            flush=True,
        )
        del pair, x_all, residual_all, y_all
        gc.collect()

    x_tree = pd.concat(tree_frames, ignore_index=True)
    y_tree = np.concatenate(tree_targets)
    config = ModelConfig(direct_estimators=estimators, n_jobs=n_jobs)
    tree_model = lgb.LGBMRegressor(**lgb_params(config, estimators))
    tree_model.fit(x_tree, y_tree, callbacks=[lgb.log_evaluation(12)])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": "spatial-ridge-v1",
        "seed": SEED,
        "stats": stats,
        "tree_model": tree_model,
        "ridge_models": ridge_models,
        "feature_names": x_tree.columns.tolist(),
        "blend_weights": BLEND_WEIGHTS,
    }
    joblib.dump(bundle, artifact_dir / "ensemble_models.joblib", compress=3)
    print(f"Saved {artifact_dir / 'ensemble_models.joblib'}", flush=True)


def observed_anchor_months(test: pd.DataFrame) -> list[int]:
    availability = test.groupby("month_idx")["TWS_t"].agg(["count", "size"])
    result = sorted(
        availability.index[(availability["count"] / availability["size"]) > 0.5]
    )
    if not result:
        raise ValueError("Test contains no mostly-observed TWS anchor map")
    return [int(value) for value in result]


def test_predictions(test: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    test = augment(test).copy()
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    anchors = observed_anchor_months(test)
    by_month = {int(key): value for key, value in test.groupby("month_idx", sort=False)}
    outputs: list[pd.DataFrame] = []

    for month in sorted(by_month):
        current = by_month[month].copy()
        anchor_month = max(value for value in anchors if value <= month)
        anchor = by_month[anchor_month][["loc_key", *ANCHOR_COLUMNS]].rename(
            columns={column: f"anchor_{column}" for column in ANCHOR_COLUMNS}
        )
        pair = current.merge(anchor, on="loc_key", how="left", validate="one_to_one")
        horizon = np.full(len(pair), min(7, month + 1 - anchor_month), np.int8)

        # Sparse finite values in otherwise masked maps are legal and get h=1.
        available = np.isfinite(pair["TWS_t"].to_numpy(np.float32))
        for column in ANCHOR_COLUMNS:
            pair.loc[available, f"anchor_{column}"] = pair.loc[available, column]
        horizon[available] = 1

        anchor_frame = pd.DataFrame({"TWS_t": pair["anchor_TWS_t"].to_numpy()})
        for column in CLIMATE:
            anchor_frame[column] = pair[f"anchor_{column}"].to_numpy()
        x = build_feature_frame(pair, bundle["stats"], anchor=anchor_frame, horizon=horizon)
        x = add_spatial_x(x, pair)
        tree = x["anchor_tws"].to_numpy() + bundle["tree_model"].predict(
            x[bundle["feature_names"]]
        )
        ridge = np.empty(len(pair), np.float32)
        keys = pair["loc_key"].to_numpy(np.int32)
        for value in range(1, 8):
            mask = horizon == value
            if mask.any():
                ridge[mask] = np.einsum(
                    "ij,ij->i",
                    ridge_matrix(x.loc[mask]),
                    bundle["ridge_models"][value][keys[mask]],
                )
        weight = np.array([bundle["blend_weights"][int(v)] for v in horizon], np.float32)
        prediction = weight * tree + (1.0 - weight) * ridge
        outputs.append(
            pd.DataFrame({
                "_row_id": pair["_row_id"].to_numpy(),
                "ID": pair["ID"].to_numpy(),
                "Target": prediction.astype(np.float32),
                "horizon": horizon,
            })
        )
    return pd.concat(outputs, ignore_index=True).sort_values("_row_id")


def predict(data_dir: Path, artifact_dir: Path, output: Path) -> None:
    bundle = joblib.load(artifact_dir / "ensemble_models.joblib")
    _, test, sample_path = load_data(data_dir, need_test=True)
    assert test is not None and sample_path is not None
    raw = test_predictions(test, bundle)
    sample = pd.read_csv(sample_path)
    submission = sample[["ID"]].merge(raw[["ID", "Target"]], on="ID", validate="one_to_one")
    if len(submission) != len(sample) or not np.isfinite(submission["Target"]).all():
        raise ValueError("Submission alignment or finiteness check failed")
    stats = bundle["stats"]
    submission["Target"] = np.clip(
        submission["Target"], stats.target_min - 0.25, stats.target_max + 0.25
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Saved {len(submission):,} rows to {output}")
    print("Horizons:", raw["horizon"].value_counts().sort_index().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_final"))
    parser.add_argument("--output", type=Path, default=Path("Submission_Ensemble_v1.csv"))
    parser.add_argument("--rows-per-horizon", type=int, default=240_000)
    parser.add_argument("--estimators", type=int, default=36)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    seed_everything()
    if args.command in {"train", "all"}:
        train(args.data_dir, args.artifact_dir, args.rows_per_horizon, args.estimators, args.n_jobs)
    if args.command in {"predict", "all"}:
        predict(args.data_dir, args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
