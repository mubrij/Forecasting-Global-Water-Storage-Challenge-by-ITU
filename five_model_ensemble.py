#!/usr/bin/env python3
"""Validate and train a diverse five-model TWS residual ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor

from map_unet_solution import VAL_BLOCKS, midx
from recency_regime_experiment import LATE_BLOCKS, training_set, validation_set
from recency_regime_production import test_features
from spatial_experiment import augment
from winning_solution import ModelConfig, fit_history_stats, lgb_params, load_data, rmse, seed_everything


MODEL_NAMES = ["lgb_l2", "lgb_l1", "catboost", "xgboost", "extra_trees"]


def model_factory(name: str, iterations: int, n_jobs: int):
    config = ModelConfig(direct_estimators=iterations, learning_rate=.035,
                         num_leaves=64, min_child_samples=350, n_jobs=n_jobs)
    if name.startswith("lgb"):
        params = lgb_params(config, iterations)
        if name == "lgb_l1":
            params.update(objective="regression_l1", metric="l2")
        return lgb.LGBMRegressor(**params)
    if name == "catboost":
        return CatBoostRegressor(
            iterations=max(120, iterations * 2), depth=8, learning_rate=.045,
            loss_function="RMSE", l2_leaf_reg=12, random_seed=20260817,
            verbose=False, allow_writing_files=False, thread_count=n_jobs,
        )
    if name == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=max(140, iterations * 2), max_depth=8,
            learning_rate=.04, subsample=.8, colsample_bytree=.8,
            min_child_weight=80, reg_lambda=12, reg_alpha=.02,
            objective="reg:squarederror", tree_method="hist",
            random_state=20260817, n_jobs=n_jobs,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=max(100, iterations), max_depth=22,
            min_samples_leaf=24, max_features=.75, bootstrap=False,
            random_state=20260817, n_jobs=n_jobs,
        )
    raise KeyError(name)


def fit_model(model, x, y):
    if isinstance(model, lgb.LGBMRegressor):
        return model.fit(x, y, callbacks=[lgb.log_evaluation(0)])
    return model.fit(x, y)


def center_maps(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    for group in np.unique(groups):
        take = groups == group
        out[take] -= out[take].mean()
    return out


def validate_schedule(frame, blocks, rows_per_horizon, iterations, n_jobs):
    cutoff = midx(blocks[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)
    tx, ty, _ = training_set(fit, stats, rows_per_horizon)
    vx, truth, horizon, group = validation_set(frame, stats, blocks)
    train_x = tx.to_numpy(np.float32)
    val_x = vx.to_numpy(np.float32)
    anchor = vx.anchor_tws.to_numpy(np.float64)
    predictions = []
    scores = {}
    for name in MODEL_NAMES:
        print(f"training {name} on {len(train_x):,} rows", flush=True)
        model = fit_model(model_factory(name, iterations, n_jobs), train_x, ty)
        prediction = anchor + model.predict(val_x)
        predictions.append(center_maps(prediction, group))
        scores[name] = rmse(truth, prediction)
        print(f"{name} rmse={scores[name]:.9f}", flush=True)
    return {
        "truth": center_maps(truth, group), "prediction": np.column_stack(predictions),
        "raw_truth": truth, "group": group, "horizon": horizon, "scores": scores,
    }


def select_weights(results):
    def objective(weights):
        values = []
        for result in results:
            error = result["truth"] - result["prediction"] @ weights
            values.append(np.mean(error * error))
        return float(np.mean(values))
    constraints = {"type": "eq", "fun": lambda weights: weights.sum() - 1.0}
    fitted = minimize(objective, np.full(5, .2), method="SLSQP",
                      bounds=[(.03, .88)] * 5, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-12})
    if not fitted.success:
        raise RuntimeError(f"weight optimization failed: {fitted.message}")
    weights = fitted.x
    # Blend only the model-diversity direction relative to the strongest common
    # reference, then choose an alpha that improves both regimes.
    grid = []
    for alpha in np.linspace(0, 1, 101):
        ratios, scores = [], []
        for result in results:
            base = result["prediction"][:, 0]
            ensemble = result["prediction"] @ weights
            candidate = base + alpha * (ensemble - base)
            score = rmse(result["truth"], candidate)
            base_score = rmse(result["truth"], base)
            ratios.append(score / base_score); scores.append(score)
        grid.append((max(ratios), float(alpha), scores))
    _, alpha, alpha_scores = min(grid, key=lambda row: row[0])
    return weights, alpha, alpha_scores


def validate(args):
    frame, _, _ = load_data(args.data_dir, need_test=False)
    frame = augment(frame)
    original = validate_schedule(frame, VAL_BLOCKS, args.validation_rows,
                                 args.iterations, args.n_jobs)
    late = validate_schedule(frame, LATE_BLOCKS, args.validation_rows,
                             args.iterations, args.n_jobs)
    weights, alpha, alpha_scores = select_weights([original, late])
    summary = {
        "models": MODEL_NAMES, "weights": dict(zip(MODEL_NAMES, map(float, weights))),
        "blend_alpha": float(alpha),
        "alpha_scores": {"original": alpha_scores[0], "late": alpha_scores[1]},
        "standalone_scores": {"original": original["scores"], "late": late["scores"]},
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def production(args):
    metadata = json.loads((args.artifact_dir / "validation.json").read_text())
    weights = np.asarray([metadata["weights"][name] for name in MODEL_NAMES])
    alpha = float(metadata["blend_alpha"])
    train, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    train = augment(train)
    stats = fit_history_stats(train)
    tx, ty, _ = training_set(train, stats, args.production_rows)
    train_x = tx.to_numpy(np.float32)
    test_x, meta = test_features(test, stats)
    raw_test_x = test_x.to_numpy(np.float32)
    anchor = test_x.anchor_tws.to_numpy(np.float64)
    models, predictions = {}, []
    for name in MODEL_NAMES:
        print(f"full training {name} on {len(train_x):,} rows", flush=True)
        model = fit_model(model_factory(name, args.iterations, args.n_jobs), train_x, ty)
        models[name] = model
        predictions.append(anchor + model.predict(raw_test_x))
    matrix = np.column_stack(predictions)
    ensemble = matrix @ weights
    correction = ensemble - matrix[:, 0]
    correction = center_maps(correction, meta.group.to_numpy())
    sample = pd.read_csv(sample_path)[["ID"]]
    base = sample.merge(pd.read_csv(args.base), on="ID", validate="one_to_one")
    output = sample.copy()
    output["Target"] = base.Target.to_numpy(np.float64) + alpha * correction
    standalone = sample.copy(); standalone["Target"] = ensemble
    if not np.isfinite(output.Target).all() or not output.ID.is_unique:
        raise ValueError("invalid five-model blend")
    standalone.to_csv(args.artifact_dir / "Submission_FiveModelStandalone.csv", index=False)
    output.to_csv(args.output, index=False)
    joblib.dump({"models": models, "features": list(tx.columns), "weights": weights,
                 "alpha": alpha}, args.artifact_dir / "models.joblib", compress=3)
    print(f"saved {args.output}: {len(output):,} rows; alpha={alpha:.3f}; "
          f"correction_rms={np.sqrt(np.mean(correction**2)):.9f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "production", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_five_model"))
    parser.add_argument("--base", type=Path, default=Path("Submission_V25_LeaderboardSurfaceOptimal.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V26_FiveModelBlend.csv"))
    parser.add_argument("--validation-rows", type=int, default=35000)
    parser.add_argument("--production-rows", type=int, default=70000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    seed_everything()
    if args.command in ("validate", "all"):
        validate(args)
    if args.command in ("production", "all"):
        production(args)


if __name__ == "__main__":
    main()
