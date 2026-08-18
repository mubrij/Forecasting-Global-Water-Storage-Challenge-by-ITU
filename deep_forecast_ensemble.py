#!/usr/bin/env python3
"""Train and ensemble five deep temporal forecasting architectures."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch import nn
from torch.utils.data import DataLoader

from lstm_production import build_test
from lstm_solution import (
    MapBatchDataset, RESIDUAL_SCALE, SEQ_LEN, TemporalExample, TemporalStore,
)
from map_unet_solution import VAL_BLOCKS, midx
from winning_solution import SEED, load_data, rmse, seed_everything


MODEL_NAMES = ["nbeats_mlp", "gru", "lstm", "tcn", "transformer"]


class Head(nn.Module):
    def __init__(self, encoded: int, context: int, locations: int):
        super().__init__()
        self.location = nn.Embedding(locations, 16)
        self.net = nn.Sequential(
            nn.Linear(encoded + context + 16, 192), nn.LayerNorm(192),
            nn.SiLU(), nn.Dropout(.12), nn.Linear(192, 96), nn.SiLU(),
            nn.Dropout(.08), nn.Linear(96, 1),
        )

    def forward(self, encoded, context, loc):
        return self.net(torch.cat([encoded, context, self.location(loc)], 1)).squeeze(1)


class NBeatsMLP(nn.Module):
    def __init__(self, seq_features, context_features, locations):
        super().__init__()
        width = 256
        self.input = nn.Linear(SEQ_LEN * seq_features, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                          nn.Dropout(.1), nn.Linear(width, width)) for _ in range(4)
        ])
        self.head = Head(width, context_features, locations)

    def forward(self, seq, context, loc):
        value = self.input(seq.flatten(1))
        for block in self.blocks:
            value = value + block(value)
        return self.head(value, context, loc)


class RNNModel(nn.Module):
    def __init__(self, kind, seq_features, context_features, locations):
        super().__init__()
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(seq_features, 96, num_layers=2, batch_first=True, dropout=.12)
        self.head = Head(96, context_features, locations)

    def forward(self, seq, context, loc):
        encoded, _ = self.rnn(seq)
        return self.head(encoded[:, -1], context, loc)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width), nn.SiLU(), nn.Dropout(.08),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width), nn.SiLU(),
        )

    def forward(self, value):
        return value + self.net(value)


class TCNModel(nn.Module):
    def __init__(self, seq_features, context_features, locations):
        super().__init__()
        self.input = nn.Conv1d(seq_features, 96, 1)
        self.blocks = nn.Sequential(*[TemporalBlock(96, dilation) for dilation in (1, 2, 4, 8)])
        self.head = Head(96, context_features, locations)

    def forward(self, seq, context, loc):
        encoded = self.blocks(self.input(seq.transpose(1, 2))).mean(2)
        return self.head(encoded, context, loc)


class TransformerModel(nn.Module):
    def __init__(self, seq_features, context_features, locations):
        super().__init__()
        width = 96
        self.input = nn.Linear(seq_features, width)
        self.position = nn.Parameter(torch.zeros(1, SEQ_LEN, width))
        layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=256, dropout=.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(width))
        self.head = Head(width, context_features, locations)

    def forward(self, seq, context, loc):
        encoded = self.encoder(self.input(seq) + self.position).mean(1)
        return self.head(encoded, context, loc)


def make_model(name, seq_features, context_features, locations):
    if name == "nbeats_mlp":
        return NBeatsMLP(seq_features, context_features, locations)
    if name in ("gru", "lstm"):
        return RNNModel(name, seq_features, context_features, locations)
    if name == "tcn":
        return TCNModel(seq_features, context_features, locations)
    if name == "transformer":
        return TransformerModel(seq_features, context_features, locations)
    raise KeyError(name)


def training_examples(store, cutoff):
    valid = np.isfinite(store.tws).any(axis=1) & np.isfinite(store.target).any(axis=1)
    observed = {int(store.months[index]) for index in np.flatnonzero(
        valid & (store.months < cutoff)
    )}
    return [TemporalExample(anchor, current, horizon, -1)
            for current in sorted(observed) for horizon in range(1, 8)
            if (anchor := current - horizon + 1) in observed]


def validation_examples(store):
    examples, visible, group = [], [], 0
    available = {int(store.months[index]) for index in
                 np.flatnonzero(np.isfinite(store.target).any(axis=1))}
    for date, max_horizon in VAL_BLOCKS:
        anchor = midx(date); visible.append(anchor)
        for horizon in range(1, max_horizon + 1):
            current = anchor + horizon - 1
            if current not in available:
                break
            examples.append(TemporalExample(anchor, current, horizon, group, tuple(visible)))
            group += 1
    return examples


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); truths=[]; predictions=[]; horizons=[]; groups=[]; keys=[]
    for seq, context, loc, target, anchor, meta in loader:
        seq, context, loc = seq[0].to(device), context[0].to(device), loc[0].to(device)
        pred = model(seq, context, loc).float().cpu().numpy() * RESIDUAL_SCALE + anchor[0].numpy()
        truth = target[0].numpy() * RESIDUAL_SCALE + anchor[0].numpy()
        truths.append(truth); predictions.append(pred)
        keys.append(meta[0, :, 0].numpy()); horizons.append(meta[0, :, 1].numpy())
        groups.append(meta[0, :, 2].numpy())
    return tuple(map(np.concatenate, (truths, predictions, horizons, groups, keys)))


def train_network(name, store, train_ds, val_loader, dimensions, artifact_dir,
                  epochs, device):
    seq_features, context_features = dimensions
    model = make_model(name, seq_features, context_features, len(store.locs)).to(device)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train(); losses=[]
        for seq, context, loc, target, _, _ in loader:
            seq, context, loc, target = (seq[0].to(device), context[0].to(device),
                                         loc[0].to(device), target[0].to(device))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                prediction = model(seq, context, loc)
                loss = (prediction - target).square().mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        scheduler.step()
        truth, prediction, horizon, group, key = evaluate(model, val_loader, device)
        score = rmse(truth, prediction)
        print(name, "epoch", epoch, "loss", round(float(np.mean(losses)), 5),
              "rmse", round(score, 6), flush=True)
        if score < best - 2e-4:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"state": model.state_dict(), "name": name,
                        "seq_features": seq_features, "context_features": context_features,
                        "locations": len(store.locs), "epoch": epoch},
                       artifact_dir / f"{name}_validation.pt")
            np.savez_compressed(artifact_dir / f"{name}_validation.npz", truth=truth,
                                prediction=prediction, horizon=horizon, group=group, key=key)
        else:
            stale += 1
            if stale >= 3:
                break
    return best, best_epoch


def center_maps(values, groups):
    out = values.astype(np.float64, copy=True)
    for group in np.unique(groups):
        take = groups == group; out[take] -= out[take].mean()
    return out


def ensemble_validation(artifact_dir):
    arrays = [np.load(artifact_dir / f"{name}_validation.npz") for name in MODEL_NAMES]
    truth, group = arrays[0]["truth"], arrays[0]["group"]
    matrix = np.column_stack([center_maps(item["prediction"], group) for item in arrays])
    target = center_maps(truth, group)
    result = minimize(lambda w: np.mean((target - matrix @ w) ** 2), np.full(5, .2),
                      method="SLSQP", bounds=[(.03, .88)] * 5,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                      options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    prediction = matrix @ result.x
    return result.x, rmse(target, prediction)


def validate(args):
    frame, _, _ = load_data(args.data_dir, need_test=False)
    cutoff = midx(VAL_BLOCKS[0][0]); store = TemporalStore(frame, cutoff)
    train_ds = MapBatchDataset(store, training_examples(store, cutoff), cutoff,
                               args.cells_per_map, training=True)
    val_ds = MapBatchDataset(store, validation_examples(store), cutoff, 999999, training=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    sample = train_ds[0]; dimensions = (sample[0].shape[-1], sample[1].shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    scores, epochs = {}, {}
    for name in MODEL_NAMES:
        scores[name], epochs[name] = train_network(
            name, store, train_ds, val_loader, dimensions, args.artifact_dir,
            args.epochs, device,
        )
        torch.cuda.empty_cache()
    weights, ensemble_score = ensemble_validation(args.artifact_dir)
    metadata = {"models": MODEL_NAMES, "scores": scores, "epochs": epochs,
                "weights": dict(zip(MODEL_NAMES, map(float, weights))),
                "ensemble_centered_rmse": ensemble_score,
                "seq_features": dimensions[0], "context_features": dimensions[1]}
    (args.artifact_dir / "validation.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def train_full_model(name, store, dataset, dimensions, epochs, device):
    model = make_model(name, *dimensions, len(store.locs)).to(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(epochs):
        model.train(); losses=[]
        for seq, context, loc, target, _, _ in loader:
            seq, context, loc, target = (seq[0].to(device), context[0].to(device),
                                         loc[0].to(device), target[0].to(device))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss = (model(seq, context, loc) - target).square().mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        print(name, "full epoch", epoch + 1, "loss", round(float(np.mean(losses)), 5), flush=True)
    return model


@torch.inference_mode()
def predict_model(model, dataset, test, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    rows=[]
    model.eval()
    for seq, context, loc, _, anchor, meta in loader:
        pred = model(seq[0].to(device), context[0].to(device), loc[0].to(device))
        pred = pred.float().cpu().numpy() * RESIDUAL_SCALE + anchor[0].numpy()
        rows.append(pd.DataFrame({"group": meta[0, :, 2].numpy(),
                                  "loc_key": meta[0, :, 0].numpy(), "Target": pred}))
    prediction = pd.concat(rows, ignore_index=True)
    month_group = {month: group for group, month in enumerate(sorted(test.month_idx.unique()))}
    lookup = test[["ID", "month_idx", "loc_key"]].copy()
    lookup["group"] = lookup.month_idx.map(month_group)
    return lookup.merge(prediction, on=["group", "loc_key"], validate="one_to_one").Target.to_numpy()


def production(args):
    metadata = json.loads((args.artifact_dir / "validation.json").read_text())
    weights = np.asarray([metadata["weights"][name] for name in MODEL_NAMES])
    frame, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    cutoff = int(frame.month_idx.max()) + 1; store = TemporalStore(frame, cutoff)
    dataset = MapBatchDataset(store, training_examples(store, cutoff), cutoff,
                              args.cells_per_map, training=True)
    test_store, examples = build_test(frame, test)
    test_dataset = MapBatchDataset(test_store, examples, int(test.month_idx.min()),
                                   999999, training=False, inference=True)
    sample_item = dataset[0]; dimensions = (sample_item[0].shape[-1], sample_item[1].shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions=[]
    for name in MODEL_NAMES:
        epochs = max(1, int(metadata["epochs"][name]))
        model = train_full_model(name, store, dataset, dimensions, epochs, device)
        predictions.append(predict_model(model, test_dataset, test, device))
        torch.save({"state": model.state_dict(), "name": name, "dimensions": dimensions,
                    "locations": len(store.locs), "epochs": epochs},
                   args.artifact_dir / f"{name}_full.pt")
        del model; torch.cuda.empty_cache()
    deep = np.column_stack(predictions) @ weights
    sample = pd.read_csv(sample_path)[["ID"]]
    base = sample.merge(pd.read_csv(args.base), on="ID", validate="one_to_one")
    correction = deep - base.Target.to_numpy(np.float64)
    months = sample.merge(test[["ID", "month_idx"]], on="ID", validate="one_to_one").month_idx
    correction -= pd.Series(correction).groupby(months).transform("mean").to_numpy()
    output = sample.copy(); output["Target"] = base.Target.to_numpy() + args.blend * correction
    standalone = sample.copy(); standalone["Target"] = deep
    standalone.to_csv(args.artifact_dir / "Submission_DeepEnsembleStandalone.csv", index=False)
    output.to_csv(args.output, index=False)
    print(f"saved {args.output}: {len(output):,}; blend={args.blend}; "
          f"correction_rms={np.sqrt(np.mean(correction**2)):.9f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "production", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_deep_ensemble"))
    parser.add_argument("--base", type=Path, default=Path("Submission_V27_CalibratedFiveModelBlend.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V28_DeepForecastBlend.csv"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cells-per-map", type=int, default=4096)
    parser.add_argument("--blend", type=float, default=.25)
    args = parser.parse_args()
    seed_everything(SEED); random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark = True
    if args.command in ("validate", "all"):
        validate(args)
    if args.command in ("production", "all"):
        production(args)


if __name__ == "__main__":
    main()
