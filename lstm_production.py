#!/usr/bin/env python3
"""Train full-history temporal LSTM and create raw/smoothed submissions."""
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
from lstm_solution import (
    MapBatchDataset, RESIDUAL_SCALE, TemporalExample, TemporalLSTM, TemporalStore,
)
from winning_solution import SEED, load_data, seed_everything


def train_full(frame: pd.DataFrame, checkpoint: Path, epochs: int = 5):
    cutoff = int(frame.month_idx.max()) + 1
    store = TemporalStore(frame, cutoff)
    observed = set(map(int, frame.month_idx.unique()))
    examples = []
    for current in sorted(observed):
        for horizon in range(1, 8):
            anchor = current - horizon + 1
            if anchor in observed:
                examples.append(TemporalExample(anchor, current, horizon, -1))
    dataset = MapBatchDataset(store, examples, cutoff, 2048, True)
    sample = dataset[0]
    model = TemporalLSTM(len(store.locs), sample[0].shape[-1], sample[1].shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for seq, ctx, loc, target, _, _ in loader:
            seq, ctx, loc, target = (
                seq[0].to(device), ctx[0].to(device), loc[0].to(device), target[0].to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                prediction = model(seq, ctx, loc)
                loss = (prediction - target).square().mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update(); losses.append(loss.detach().item())
        scheduler.step()
        print(f"epoch={epoch} loss={np.mean(losses):.6f}", flush=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": model.state_dict(), "seq_features": sample[0].shape[-1],
        "context_features": sample[1].shape[-1], "n_locations": len(store.locs),
        "epochs": epochs,
    }, checkpoint)


def build_test(frame: pd.DataFrame, test: pd.DataFrame):
    test_copy = test.copy(); test_copy["Target"] = np.nan
    combined = pd.concat([frame, test_copy[frame.columns]], ignore_index=True)
    cutoff = int(test.month_idx.min())
    store = TemporalStore(combined, cutoff)
    anchors = observed_anchor_months(test)
    examples, visible = [], []
    for group, current in enumerate(sorted(map(int, test.month_idx.unique()))):
        anchor = max(value for value in anchors if value <= current)
        if anchor not in visible: visible.append(anchor)
        examples.append(TemporalExample(
            anchor, current, min(7, current-anchor+1), group, tuple(visible)
        ))
    return store, examples


@torch.inference_mode()
def predict(frame, test, checkpoint):
    store, examples = build_test(frame, test)
    dataset = MapBatchDataset(store, examples, int(test.month_idx.min()), 999999,
                              training=False, inference=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(checkpoint, map_location=device, weights_only=True)
    model = TemporalLSTM(bundle["n_locations"], bundle["seq_features"], bundle["context_features"])
    model.load_state_dict(bundle["state"]); model.to(device).eval()
    rows = []
    for seq, ctx, loc, _, anchor, meta in loader:
        output = model(seq[0].to(device), ctx[0].to(device), loc[0].to(device))
        output = output.float().cpu().numpy() * RESIDUAL_SCALE + anchor[0].numpy()
        rows.append(pd.DataFrame({
            "group": meta[0, :, 2].numpy(), "loc_key": meta[0, :, 0].numpy(),
            "Target": output.astype(np.float32),
        }))
    predictions = pd.concat(rows, ignore_index=True)
    month_group = {month: group for group, month in enumerate(sorted(test.month_idx.unique()))}
    lookup = test[["ID", "month_idx", "loc_key"]].copy()
    lookup["group"] = lookup.month_idx.map(month_group)
    return lookup.merge(predictions, on=["group", "loc_key"], how="left", validate="one_to_one")


def smooth(test, values, sigma=1.5):
    result = np.empty_like(values, np.float32)
    for _, index in test.groupby("month_idx", sort=False).groups.items():
        index = np.asarray(index); rows = test.iloc[index]
        ii = np.rint(rows.lat.to_numpy()+55.5).astype(int)
        jj = np.rint(rows.lon.to_numpy()+179.5).astype(int)
        grid = np.zeros((140,360),np.float32); mask = np.zeros_like(grid)
        grid[ii,jj] = values[index]; mask[ii,jj] = 1
        num = gaussian_filter(grid,sigma,mode=("nearest","wrap"))
        den = gaussian_filter(mask,sigma,mode=("nearest","wrap"))
        result[index] = np.divide(num,den,out=grid,where=den>1e-6)[ii,jj]
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["train","predict","all"])
    p.add_argument("--data-dir",type=Path,default=Path("."));p.add_argument("--artifact-dir",type=Path,default=Path("artifacts_lstm_full"))
    p.add_argument("--output",type=Path,default=Path("Submission_LSTM_Smoothed.csv"));a=p.parse_args()
    seed_everything(SEED);random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.benchmark=True
    frame,test,sample_path=load_data(a.data_dir,True);assert test is not None and sample_path is not None
    checkpoint=a.artifact_dir/"lstm_full.pt"
    if a.command in {"train","all"}:train_full(frame,checkpoint)
    if a.command in {"predict","all"}:
        raw=predict(frame,test,checkpoint);sample=pd.read_csv(sample_path)[["ID"]]
        aligned=sample.merge(raw[["ID","Target"]],on="ID",validate="one_to_one")
        if not np.isfinite(aligned.Target).all():raise ValueError("non-finite LSTM predictions")
        a.artifact_dir.mkdir(parents=True,exist_ok=True)
        aligned.to_csv(a.artifact_dir/"Submission_LSTM_Raw.csv",index=False)
        aligned["Target"]=smooth(test,aligned.Target.to_numpy(),1.5)
        aligned.to_csv(a.output,index=False)
        print(f"saved {a.output}: {len(aligned):,} rows",flush=True)


if __name__=="__main__":main()
