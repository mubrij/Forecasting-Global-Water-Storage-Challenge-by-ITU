#!/usr/bin/env python3
"""Global residual-map U-Net for the Zindi TWS forecasting challenge.

The model uses only information available at each forecast time: the last
observed TWS map, the previous observed anchor, and the complete supplied
climate path up to the predictor month.  It predicts target-minus-anchor on the
native 1-degree grid and is trained with a loss mask over supplied land cells.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from winning_solution import CLIMATE, SEED, load_data, rmse, seed_everything

H, W = 140, 360
VARS = ["TWS_t", "Target", *CLIMATE]
VAL_BLOCKS = [
    ("2012-07-01", 3), ("2012-12-01", 3), ("2013-05-01", 3),
    ("2013-11-01", 3), ("2014-04-01", 3), ("2014-09-01", 7),
]


def midx(date: str | pd.Timestamp) -> int:
    value = pd.Timestamp(date)
    return value.year * 12 + value.month - 1


@dataclass
class Config:
    base: int = 24
    batch_size: int = 4
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 2e-3
    weight_decay: float = 2e-4
    num_workers: int = 4
    max_horizon: int = 7
    amp: bool = True


class MapStore:
    def __init__(self, frame: pd.DataFrame, fit_before: int | None):
        self.frame = frame
        self.months = np.sort(frame.month_idx.unique()).astype(np.int32)
        self.pos = {int(m): i for i, m in enumerate(self.months)}
        n = len(self.months)
        self.data = {c: np.zeros((n, H, W), np.float32) for c in VARS}
        self.present = np.zeros((n, H, W), bool)
        ii = np.rint(frame.lat.to_numpy() + 55.5).astype(np.int16)
        jj = np.rint(frame.lon.to_numpy() + 179.5).astype(np.int16)
        tt = frame.month_idx.map(self.pos).to_numpy(np.int16)
        self.present[tt, ii, jj] = True
        for c in VARS:
            self.data[c][tt, ii, jj] = frame[c].to_numpy(np.float32)
        self.land = self.present.any(axis=0)

        fit_mask = self.months < fit_before if fit_before is not None else np.ones(n, bool)
        fit_pos = np.flatnonzero(fit_mask)
        counts = self.present[fit_pos].sum(axis=0)
        tws_sum = (self.data["TWS_t"][fit_pos] * self.present[fit_pos]).sum(axis=0)
        self.loc_mean = np.divide(
            tws_sum, counts, out=np.zeros((H, W), np.float32), where=counts > 0
        )
        self.climatology = np.empty((12, H, W), np.float32)
        for month in range(12):
            choose = fit_pos[self.months[fit_pos] % 12 == month]
            count = self.present[choose].sum(axis=0)
            total = (self.data["TWS_t"][choose] * self.present[choose]).sum(axis=0)
            raw = np.divide(total, count, out=self.loc_mean.copy(), where=count > 0)
            self.climatology[month] = (count * raw + 3.0 * self.loc_mean) / (count + 3.0)

        # Normalize climate from fit-period land observations only.
        self.climate_mean, self.climate_std = {}, {}
        fit_rows = frame[frame.month_idx < fit_before] if fit_before is not None else frame
        for c in CLIMATE:
            self.climate_mean[c] = float(fit_rows[c].mean())
            self.climate_std[c] = max(float(fit_rows[c].std()), 1e-4)

    def field(self, variable: str, month: int, fill: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        p = self.pos.get(int(month))
        if p is None:
            value = np.broadcast_to(np.asarray(fill, np.float32), (H, W)).copy()
            return value, np.zeros((H, W), bool)
        mask = self.present[p] & np.isfinite(self.data[variable][p])
        value = np.where(mask, self.data[variable][p], fill).astype(np.float32)
        return value, mask

    def climate(self, variable: str, month: int) -> tuple[np.ndarray, np.ndarray]:
        raw, mask = self.field(variable, month, self.climate_mean[variable])
        return ((raw - self.climate_mean[variable]) / self.climate_std[variable]).astype(np.float32), mask


def previous_available(months: np.ndarray, anchor: int) -> int:
    candidates = months[months < anchor]
    return int(candidates[-1]) if len(candidates) else anchor


@dataclass(frozen=True)
class Example:
    anchor: int
    current: int
    horizon: int
    group: int


def train_examples(store: MapStore, cutoff: int) -> list[Example]:
    available = set(map(int, store.months[store.months < cutoff]))
    examples = []
    for current in sorted(available):
        for horizon in range(1, 8):
            anchor = current - horizon + 1
            if anchor in available:
                examples.append(Example(anchor, current, horizon, -1))
    return examples


def validation_examples(store: MapStore) -> list[Example]:
    examples, group = [], 0
    available = set(map(int, store.months))
    for date, max_h in VAL_BLOCKS:
        anchor = midx(date)
        for horizon in range(1, max_h + 1):
            current = anchor + horizon - 1
            if current not in available:
                break
            examples.append(Example(anchor, current, horizon, group))
            group += 1
    return examples


class TWSDataset(Dataset):
    def __init__(self, store: MapStore, examples: list[Example], augment: bool = False,
                 previous_by_anchor: dict[int, int] | None = None):
        self.store, self.examples, self.augment = store, examples, augment
        self.previous_by_anchor = previous_by_anchor or {}
        lat = np.linspace(-55.5, 83.5, H, dtype=np.float32)[:, None]
        lon = np.linspace(-179.5, 179.5, W, dtype=np.float32)[None, :]
        self.static = np.stack([
            np.broadcast_to(lat / 90.0, (H, W)),
            np.broadcast_to(np.sin(np.deg2rad(lat)), (H, W)),
            np.broadcast_to(np.cos(np.deg2rad(lat)), (H, W)),
            np.broadcast_to(np.sin(np.deg2rad(lon)), (H, W)),
            np.broadcast_to(np.cos(np.deg2rad(lon)), (H, W)),
            np.broadcast_to(np.sin(2 * np.deg2rad(lon)), (H, W)),
            np.broadcast_to(np.cos(2 * np.deg2rad(lon)), (H, W)),
            store.land.astype(np.float32),
        ]).astype(np.float32)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ex = self.examples[index]
        store = self.store
        anchor, anchor_mask = store.field("TWS_t", ex.anchor, store.loc_mean)
        prev_month = self.previous_by_anchor.get(
            ex.anchor, previous_available(store.months, ex.anchor)
        )
        previous, previous_mask = store.field("TWS_t", prev_month, store.loc_mean)
        gap = max(1, ex.anchor - prev_month)
        target, target_mask = store.field("Target", ex.current, 0.0)
        target_month = (ex.current + 1) % 12
        clim = store.climatology[target_month]

        channels = [
            anchor, previous, (anchor - previous) / gap, store.loc_mean, clim,
            anchor - clim, anchor_mask.astype(np.float32), previous_mask.astype(np.float32),
        ]
        path_months = list(range(ex.anchor, ex.current + 1))
        for variable in CLIMATE:
            sequence = np.stack([store.climate(variable, m)[0] for m in path_months])
            weights = np.arange(len(sequence), dtype=np.float32)
            weights -= weights.mean()
            slope = np.zeros((H, W), np.float32) if len(sequence) == 1 else (
                np.tensordot(weights, sequence, axes=(0, 0)) / float(np.dot(weights, weights))
            ).astype(np.float32)
            channels.extend([
                sequence[0], sequence[-1], sequence.mean(axis=0),
                sequence.std(axis=0), sequence.min(axis=0), sequence.max(axis=0), slope,
            ])

        current_angle = 2 * math.pi * (ex.current % 12) / 12
        target_angle = 2 * math.pi * target_month / 12
        constants = [
            math.sin(current_angle), math.cos(current_angle),
            math.sin(target_angle), math.cos(target_angle),
            ex.horizon / 7.0, min(gap, 24) / 24.0,
        ]
        channels.extend([np.full((H, W), v, np.float32) for v in constants])
        channels.extend(list(self.static))
        x = np.stack(channels).astype(np.float32)
        y = ((target - anchor) / 0.75).astype(np.float32)[None]
        mask = target_mask.astype(np.float32)[None]
        if self.augment and random.random() < 0.5:
            # Circular longitude roll is physically valid when every channel,
            # coordinate field, target, and mask is shifted together.
            shift = random.randrange(-W // 2, W // 2)
            x, y, mask = (np.roll(v, shift, axis=-1).copy() for v in (x, y, mask))
        meta = np.array([ex.anchor, ex.current, ex.horizon, ex.group], np.int32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(anchor[None]), meta


class Block(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.body(x) + self.skip(x)


class ResidualUNet(nn.Module):
    def __init__(self, channels: int, base: int):
        super().__init__()
        self.e1 = Block(channels, base)
        self.e2 = Block(base, base * 2)
        self.e3 = Block(base * 2, base * 4)
        self.middle = nn.Sequential(Block(base * 4, base * 6, 2), Block(base * 6, base * 6, 4))
        self.d3 = Block(base * 10, base * 4)
        self.d2 = Block(base * 6, base * 2)
        self.d1 = Block(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        z = self.middle(c)
        z = self.d3(torch.cat([z, c], 1))
        z = F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False)
        z = self.d2(torch.cat([z, b], 1))
        z = F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.d1(torch.cat([z, a], 1)))


def masked_mse(pred, target, mask):
    return ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    truths, preds, horizons, groups = [], [], [], []
    for x, y, mask, anchor, meta in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).float().cpu().numpy() * 0.75 + anchor.numpy()
        truth = y.numpy() * 0.75 + anchor.numpy()
        valid = mask.numpy().astype(bool)
        for i in range(len(x)):
            truths.append(truth[i][valid[i]])
            preds.append(pred[i][valid[i]])
            horizons.append(np.full(valid[i].sum(), int(meta[i, 2]), np.int8))
            groups.append(np.full(valid[i].sum(), int(meta[i, 3]), np.int16))
    truth, pred = np.concatenate(truths), np.concatenate(preds)
    horizon, group = np.concatenate(horizons), np.concatenate(groups)
    return truth, pred, horizon, group


def fit_model(frame, artifact_dir: Path, config: Config):
    cutoff = midx(VAL_BLOCKS[0][0])
    store = MapStore(frame, cutoff)
    train_ds = TWSDataset(store, train_examples(store, cutoff), augment=True)
    val_ds = TWSDataset(store, validation_examples(store), augment=False)
    sample_x = train_ds[0][0]
    print(f"maps: train={len(train_ds)}, val={len(val_ds)}, channels={len(sample_x)}", flush=True)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
        pin_memory=True, persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet(len(sample_x), config.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    best, stale = float("inf"), 0
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, mask, _, _ in train_loader:
            x, y, mask = (v.to(device, non_blocking=True) for v in (x, y, mask))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"):
                output = model(x)
                loss = masked_mse(output, y, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss))
        scheduler.step()
        truth, pred, horizon, group = evaluate(model, val_loader, device)
        score = rmse(truth, pred)
        detail = {h: round(rmse(truth[horizon == h], pred[horizon == h]), 5) for h in range(1, 8)}
        print(f"epoch={epoch:02d} loss={np.mean(losses):.5f} val={score:.6f} {detail}", flush=True)
        if score < best - 2e-4:
            best, stale = score, 0
            torch.save({"state": model.state_dict(), "channels": len(sample_x), "config": asdict(config)}, artifact_dir / "map_unet_val.pt")
            np.savez_compressed(artifact_dir / "map_unet_val_predictions.npz", truth=truth, prediction=pred, horizon=horizon, group=group)
        else:
            stale += 1
            if stale >= config.patience:
                break
    print(f"best validation RMSE: {best:.6f}", flush=True)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_map_unet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    config = Config(
        epochs=args.epochs, batch_size=args.batch_size, base=args.base,
        learning_rate=args.learning_rate,
    )
    frame, _, _ = load_data(args.data_dir, need_test=False)
    fit_model(frame, args.artifact_dir, config)
    (args.artifact_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))


if __name__ == "__main__":
    main()
