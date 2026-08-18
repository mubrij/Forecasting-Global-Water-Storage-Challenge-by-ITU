#!/usr/bin/env python3
"""Build a robust validation-selected ML + deep forecasting stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader

from deep_forecast_ensemble import MODEL_NAMES as DEEP_NAMES, make_model, predict_model
from lstm_production import build_test
from lstm_solution import MapBatchDataset, TemporalStore
from winning_solution import load_data, rmse


STACK_NAMES = ["tree", "unet", "deep"]


def center_maps(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    for group in np.unique(groups):
        take = groups == group
        result[take] -= result[take].mean(axis=0)
    return result


def load_validation(map_artifacts: Path, deep_artifacts: Path):
    aligned = pd.read_pickle(map_artifacts / "aligned_validation.pkl")
    merged = aligned[["group", "key", "truth", "horizon", "tree", "tabular", "unet"]].copy()
    metadata = json.loads((deep_artifacts / "validation.json").read_text())
    for name in DEEP_NAMES:
        item = np.load(deep_artifacts / f"{name}_validation.npz")
        frame = pd.DataFrame({
            "group": item["group"], "key": item["key"], name: item["prediction"],
        })
        merged = merged.merge(frame, on=["group", "key"], validate="one_to_one")
    deep_weights = np.asarray([metadata["weights"][name] for name in DEEP_NAMES])
    deep = merged[DEEP_NAMES].to_numpy(np.float64) @ deep_weights
    groups = merged.group.to_numpy()
    truth = center_maps(merged.truth.to_numpy(), groups)
    matrix = center_maps(
        np.column_stack([merged.tree.to_numpy(), merged.unet.to_numpy(), deep]), groups
    )
    tabular = center_maps(merged.tabular.to_numpy(), groups)
    return merged, truth, matrix, tabular, deep_weights


def fit_weights(truth: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = minimize(
        lambda weights: np.mean((truth - matrix @ weights) ** 2),
        np.full(matrix.shape[1], 1 / matrix.shape[1]), method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-13},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def validation_report(args):
    merged, truth, matrix, tabular, deep_weights = load_validation(
        args.map_artifacts, args.deep_artifacts
    )
    weights = fit_weights(truth, matrix)
    stacked = matrix @ weights
    direction = stacked - tabular
    alpha = float(np.dot(truth - tabular, direction) / np.dot(direction, direction))
    alpha = float(np.clip(alpha, 0, 1.25))

    groups = merged.group.to_numpy()
    blocks = np.select(
        [groups <= 2, groups <= 5, groups <= 8, groups <= 11, groups <= 14],
        [0, 1, 2, 3, 4], default=5,
    )
    lobo = []
    lobo_weights = []
    for block in range(6):
        train = blocks != block
        test = ~train
        block_weights = fit_weights(truth[train], matrix[train])
        lobo.append(rmse(truth[test], matrix[test] @ block_weights))
        lobo_weights.append(dict(zip(STACK_NAMES, map(float, block_weights))))

    report = {
        "models": STACK_NAMES,
        "weights": dict(zip(STACK_NAMES, map(float, weights))),
        "deep_models": DEEP_NAMES,
        "deep_weights": dict(zip(DEEP_NAMES, map(float, deep_weights))),
        "centered_rmse": {
            "tabular": rmse(truth, tabular),
            "stack": rmse(truth, stacked),
            "tabular_to_stack_optimal": rmse(truth, tabular + alpha * direction),
        },
        "optimal_direction_alpha": alpha,
        "leave_one_block_out_rmse": lobo,
        "leave_one_block_out_weights": lobo_weights,
    }
    args.output_artifacts.mkdir(parents=True, exist_ok=True)
    (args.output_artifacts / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def aligned_values(sample: pd.DataFrame, path: Path) -> np.ndarray:
    frame = sample.merge(pd.read_csv(path), on="ID", validate="one_to_one")
    values = frame.Target.to_numpy(np.float64)
    if len(values) != len(sample) or not np.isfinite(values).all():
        raise ValueError(f"invalid component submission: {path}")
    return values


def predict_deep_ensemble(args, train, test, sample, deep_weights):
    cutoff = int(train.month_idx.max()) + 1
    train_store = TemporalStore(train, cutoff)
    test_store, examples = build_test(train, test)
    dataset = MapBatchDataset(
        test_store, examples, int(test.month_idx.min()), 999999,
        training=False, inference=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    for name in DEEP_NAMES:
        bundle = torch.load(
            args.deep_artifacts / f"{name}_full.pt", map_location=device, weights_only=True
        )
        dimensions = tuple(bundle["dimensions"])
        model = make_model(name, *dimensions, len(train_store.locs)).to(device)
        model.load_state_dict(bundle["state"])
        values = predict_model(model, dataset, test, device)
        prediction = pd.DataFrame({"ID": test.ID.to_numpy(), "Target": values})
        aligned = sample.merge(prediction, on="ID", validate="one_to_one")
        predictions.append(aligned.Target.to_numpy(np.float64))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.column_stack(predictions) @ deep_weights


def production(args, report):
    train, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    sample = pd.read_csv(sample_path)[["ID"]]
    weights = np.asarray([report["weights"][name] for name in STACK_NAMES])
    deep_weights = np.asarray([report["deep_weights"][name] for name in DEEP_NAMES])

    tree = aligned_values(sample, args.tree)
    unet = aligned_values(sample, args.unet)
    deep = predict_deep_ensemble(args, train, test, sample, deep_weights)
    stack = np.column_stack([tree, unet, deep]) @ weights
    base = aligned_values(sample, args.base)
    correction = stack - base
    months = sample.merge(test[["ID", "month_idx"]], on="ID", validate="one_to_one").month_idx
    correction -= pd.Series(correction).groupby(months).transform("mean").to_numpy()

    output = sample.copy()
    output["Target"] = base + args.blend * correction
    standalone = sample.copy()
    standalone["Target"] = stack
    if not output.ID.is_unique or not np.isfinite(output.Target).all():
        raise ValueError("invalid stacked submission")
    standalone.to_csv(args.output_artifacts / "Submission_MLDeepStackStandalone.csv", index=False)
    output.to_csv(args.output, index=False)
    print(
        f"saved {args.output}: {len(output):,} rows; blend={args.blend:.3f}; "
        f"raw_correction_rms={np.sqrt(np.mean(correction**2)):.9f}", flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "production", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--map-artifacts", type=Path, default=Path("artifacts_map_unet"))
    parser.add_argument("--deep-artifacts", type=Path, default=Path("artifacts_deep_ensemble"))
    parser.add_argument("--output-artifacts", type=Path, default=Path("artifacts_ml_deep_stack"))
    parser.add_argument("--tree", type=Path, default=Path("Submission_Tree_v1.csv"))
    parser.add_argument("--unet", type=Path, default=Path("artifacts_map_full/Submission_MapUNet_Standalone.csv"))
    parser.add_argument("--base", type=Path, default=Path("Submission_V27_CalibratedFiveModelBlend.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V29_MLDeepStackBlend.csv"))
    parser.add_argument("--blend", type=float, default=.25)
    args = parser.parse_args()
    if args.command in {"validate", "all"}:
        report = validation_report(args)
    else:
        report = json.loads((args.output_artifacts / "validation.json").read_text())
    if args.command in {"production", "all"}:
        production(args, report)


if __name__ == "__main__":
    main()
