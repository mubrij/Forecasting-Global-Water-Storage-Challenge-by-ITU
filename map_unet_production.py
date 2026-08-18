#!/usr/bin/env python3
"""Train full-data map U-Nets and create smoothed/blended submissions."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.utils.data import DataLoader

from ensemble_solution import observed_anchor_months
from map_unet_solution import (
    Config, Example, MapStore, ResidualUNet, TWSDataset, masked_mse,
)
from winning_solution import SEED, load_data, seed_everything


def train_one(frame: pd.DataFrame, output: Path, seed: int, epochs: int, lr: float,
              batch_size: int = 4, base: int = 24) -> None:
    seed_everything(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    store = MapStore(frame, fit_before=None)
    examples = []
    available = set(map(int, store.months))
    for current in sorted(available):
        for horizon in range(1, 8):
            anchor = current - horizon + 1
            if anchor in available:
                examples.append(Example(anchor, current, horizon, -1))
    dataset = TWSDataset(store, examples, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    channels = len(dataset[0][0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet(channels, base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for x, y, mask, _, _ in loader:
            x, y, mask = (v.to(device, non_blocking=True) for v in (x, y, mask))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = masked_mse(model(x), y, mask)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update()
            losses.append(loss.detach().item())
        print(f"seed={seed} epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "channels": channels, "base": base,
                "seed": seed, "epochs": epochs, "lr": lr}, output)


def build_test_store(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy(); test = test.copy()
    test["Target"] = np.nan
    combined = pd.concat([train, test[train.columns]], ignore_index=True)
    first_test = int(test.month_idx.min())
    store = MapStore(combined, fit_before=first_test)
    anchors = observed_anchor_months(test)
    examples = []
    for group, current in enumerate(sorted(map(int, test.month_idx.unique()))):
        anchor = max(value for value in anchors if value <= current)
        examples.append(Example(anchor, current, min(7, current - anchor + 1), group))
    observed = [int(train.month_idx.max()), *anchors]
    previous = {
        anchor: max(value for value in observed if value < anchor)
        for anchor in anchors
    }
    return store, examples, previous


@torch.inference_mode()
def predict_models(train, test, checkpoints: list[Path]) -> pd.DataFrame:
    store, examples, previous = build_test_store(train, test)
    dataset = TWSDataset(store, examples, augment=False, previous_by_anchor=previous)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum = None
    for checkpoint in checkpoints:
        bundle = torch.load(checkpoint, map_location=device, weights_only=True)
        model = ResidualUNet(bundle["channels"], bundle["base"]).to(device)
        model.load_state_dict(bundle["state"]); model.eval()
        maps = []
        for x, _, _, anchor, _ in loader:
            residual = model(x.to(device, non_blocking=True)).float().cpu().numpy() * 0.75
            maps.extend((residual + anchor.numpy())[:, 0])
        values = np.stack(maps)
        accum = values if accum is None else accum + values
    prediction_maps = accum / len(checkpoints)
    month_pos = {ex.current: i for i, ex in enumerate(examples)}
    ii = np.rint(test.lat.to_numpy() + 55.5).astype(np.int16)
    jj = np.rint(test.lon.to_numpy() + 179.5).astype(np.int16)
    tt = test.month_idx.map(month_pos).to_numpy(np.int16)
    prediction = prediction_maps[tt, ii, jj]
    return pd.DataFrame({"ID": test.ID.to_numpy(), "Target": prediction.astype(np.float32)})


def smooth_submission(test: pd.DataFrame, values: np.ndarray, sigma: float) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for _, index in test.groupby("month_idx", sort=False).groups.items():
        index = np.asarray(index); rows = test.iloc[index]
        ii = np.rint(rows.lat.to_numpy() + 55.5).astype(np.int16)
        jj = np.rint(rows.lon.to_numpy() + 179.5).astype(np.int16)
        grid = np.zeros((140, 360), np.float32); mask = np.zeros_like(grid)
        grid[ii, jj] = values[index]; mask[ii, jj] = 1
        num = gaussian_filter(grid, sigma, mode=("nearest", "wrap"))
        den = gaussian_filter(mask, sigma, mode=("nearest", "wrap"))
        result[index] = np.divide(num, den, out=grid, where=den > 1e-6)[ii, jj]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_map_full"))
    parser.add_argument("--baseline", type=Path, default=Path("Submission_V4_MomentumBlend.csv"))
    parser.add_argument("--validation-baseline", type=Path, default=Path("Submission_Ensemble_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V5_MapBlend.csv"))
    args = parser.parse_args()
    train, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    checkpoints = [args.artifact_dir / "unet_seed1.pt", args.artifact_dir / "unet_seed2.pt"]
    if args.command in {"train", "all"}:
        train_one(train, checkpoints[0], SEED, epochs=2, lr=2e-3)
        train_one(train, checkpoints[1], SEED + 12, epochs=3, lr=8e-4)
    if args.command in {"predict", "all"}:
        raw = predict_models(train, test, checkpoints)
        sample = pd.read_csv(sample_path)[["ID"]]
        unet = sample.merge(raw, on="ID", validate="one_to_one")
        baseline = sample.merge(pd.read_csv(args.baseline), on="ID", validate="one_to_one")
        validation_baseline = sample.merge(
            pd.read_csv(args.validation_baseline), on="ID", validate="one_to_one"
        )
        # Validation-selected spatial scales and a block-stable conservative
        # weight: held-one-anchor-out favored 43-47% neural contribution.
        unet_s = smooth_submission(test, unet.Target.to_numpy(), 1.0)
        base_s = smooth_submission(test, baseline.Target.to_numpy(), 2.0)
        submission = sample.copy()
        validation_base_s = smooth_submission(
            test, validation_baseline.Target.to_numpy(), 2.0
        )
        submission["Target"] = 0.45 * unet_s + 0.55 * base_s
        unet.to_csv(args.artifact_dir / "Submission_MapUNet_Standalone.csv", index=False)
        pd.DataFrame({"ID": sample.ID, "Target": base_s}).to_csv(
            args.artifact_dir / "Submission_V4_Smoothed.csv", index=False
        )
        submission.to_csv(args.output, index=False)
        print(f"saved {args.output} with {len(submission):,} finite rows", flush=True)

        pd.DataFrame({
            "ID": sample.ID,
            "Target": 0.45 * unet_s + 0.55 * validation_base_s,
        }).to_csv("Submission_V5_ValidationMirror.csv", index=False)

if __name__ == "__main__":
    main()
