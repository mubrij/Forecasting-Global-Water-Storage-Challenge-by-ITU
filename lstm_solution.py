#!/usr/bin/env python3
"""Leakage-safe per-cell temporal LSTM for global TWS direct forecasting."""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from map_unet_solution import VAL_BLOCKS, midx
from winning_solution import CLIMATE, SEED, load_data, rmse, seed_everything

SEQ_LEN = 24
RESIDUAL_SCALE = .75


@dataclass(frozen=True)
class TemporalExample:
    anchor: int
    current: int
    horizon: int
    group: int
    visible_anchors: tuple[int, ...] = ()


class TemporalStore:
    def __init__(self, frame: pd.DataFrame, fit_before: int):
        self.month0 = int(frame.month_idx.min()) - SEQ_LEN
        self.month1 = int(frame.month_idx.max()) + 1
        self.months = np.arange(self.month0, self.month1 + 1, dtype=np.int32)
        self.month_pos = {int(m): i for i, m in enumerate(self.months)}
        self.locs = np.sort(frame.loc_key.unique()).astype(np.int32)
        self.loc_pos = np.full(int(self.locs.max()) + 1, -1, np.int32)
        self.loc_pos[self.locs] = np.arange(len(self.locs), dtype=np.int32)
        shape = (len(self.months), len(self.locs))
        self.tws = np.full(shape, np.nan, np.float32)
        self.target = np.full(shape, np.nan, np.float32)
        self.present = np.zeros(shape, bool)
        self.climate = np.full((len(CLIMATE), *shape), np.nan, np.float32)
        ti = frame.month_idx.map(self.month_pos).to_numpy(np.int16)
        li = self.loc_pos[frame.loc_key.to_numpy(np.int32)]
        self.tws[ti, li] = frame.TWS_t.to_numpy(np.float32)
        self.target[ti, li] = frame.Target.to_numpy(np.float32)
        self.present[ti, li] = True
        for ci, name in enumerate(CLIMATE):
            self.climate[ci, ti, li] = frame[name].to_numpy(np.float32)

        fit_t = self.months < fit_before
        tws_fit = self.tws[fit_t]
        self.loc_mean = np.nanmean(tws_fit, axis=0).astype(np.float32)
        self.loc_std = np.nanstd(tws_fit, axis=0).astype(np.float32)
        self.loc_std = np.maximum(self.loc_std, .08)
        self.global_mean = float(np.nanmean(tws_fit))
        self.global_std = float(np.nanstd(tws_fit))
        self.loc_mean = np.nan_to_num(self.loc_mean, nan=self.global_mean)
        self.loc_std = np.nan_to_num(self.loc_std, nan=self.global_std)
        self.climate_mean = np.array([
            np.nanmean(self.climate[i, fit_t]) for i in range(len(CLIMATE))
        ], np.float32)
        self.climate_std = np.array([
            np.nanstd(self.climate[i, fit_t]) for i in range(len(CLIMATE))
        ], np.float32)
        self.climate_std = np.maximum(self.climate_std, 1e-4)
        lat = self.locs // 360 - 55.5
        lon = self.locs % 360 - 179.5
        self.static = np.column_stack([
            lat / 90, np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
            np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)),
            np.sin(2*np.deg2rad(lon)), np.cos(2*np.deg2rad(lon)),
        ]).astype(np.float32)

    def pos(self, month: int) -> int:
        return self.month_pos[int(month)]


class MapBatchDataset(Dataset):
    def __init__(self, store: TemporalStore, examples: list[TemporalExample],
                 fit_before: int, cells_per_map: int = 2048, training: bool = True,
                 inference: bool = False):
        self.store, self.examples = store, examples
        self.fit_before, self.cells_per_map, self.training = fit_before, cells_per_map, training
        self.inference = inference

    def __len__(self): return len(self.examples)

    def __getitem__(self, index):
        s, ex = self.store, self.examples[index]
        ai, ci = s.pos(ex.anchor), s.pos(ex.current)
        current_valid = s.present[ci] if self.inference else np.isfinite(s.target[ci])
        valid = np.flatnonzero(current_valid if self.inference else (
            np.isfinite(s.tws[ai]) & current_valid
        ))
        if self.training and len(valid) > self.cells_per_map:
            rng = np.random.RandomState(SEED + index * 7919)
            loc = rng.choice(valid, self.cells_per_map, replace=False)
        else:
            loc = valid

        seq_pos = np.arange(ai - SEQ_LEN + 1, ai + 1)
        tws = s.tws[np.ix_(seq_pos, loc)].T.astype(np.float32)
        tws_mask = np.isfinite(tws)
        # Validation exposes only pre-cutoff observations and declared anchor maps.
        if not self.training:
            legal = (s.months[seq_pos] < self.fit_before) | np.isin(
                s.months[seq_pos], np.asarray(ex.visible_anchors, np.int32)
            )
            tws_mask &= legal[None, :]
        else:
            # Match test anchor gaps: preserve the final anchor and hide the most
            # recent intermediate TWS states for a deterministic gap augmentation.
            gap_choices = np.array([1, 1, 3, 4, 5, 6, 12, 19])
            gap = int(gap_choices[(index * 17 + ex.horizon) % len(gap_choices)])
            if gap > 1:
                tws_mask[:, max(0, SEQ_LEN-gap):-1] = False
        tws_z = (tws - s.loc_mean[loc, None]) / s.loc_std[loc, None]
        tws_z = np.where(tws_mask, tws_z, 0).astype(np.float32)

        climate = s.climate[:, seq_pos][:, :, loc].transpose(2, 1, 0)
        climate = (climate - s.climate_mean[None, None]) / s.climate_std[None, None]
        climate = np.nan_to_num(climate, nan=0, posinf=0, neginf=0).astype(np.float32)
        seq_month = s.months[seq_pos] % 12
        sin = np.broadcast_to(np.sin(2*np.pi*seq_month/12)[None, :, None], (len(loc), SEQ_LEN, 1))
        cos = np.broadcast_to(np.cos(2*np.pi*seq_month/12)[None, :, None], (len(loc), SEQ_LEN, 1))
        seq = np.concatenate([
            tws_z[:, :, None], tws_mask[:, :, None].astype(np.float32),
            climate, sin.astype(np.float32), cos.astype(np.float32),
        ], axis=2).astype(np.float32)

        anchor = s.tws[ai, loc]
        anchor = np.where(np.isfinite(anchor), anchor, s.loc_mean[loc]).astype(np.float32)
        anchor_c = s.climate[:, ai, loc].T
        current_c = s.climate[:, ci, loc].T
        anchor_c = np.nan_to_num((anchor_c-s.climate_mean)/s.climate_std, nan=0)
        current_c = np.nan_to_num((current_c-s.climate_mean)/s.climate_std, nan=0)
        path = s.climate[:, ai:ci+1, :][:, :, loc].transpose(2, 1, 0)
        path = np.nanmean((path-s.climate_mean[None, None])/s.climate_std[None, None], axis=1)
        path = np.nan_to_num(path, nan=0)
        target_month = (ex.current + 1) % 12
        constants = np.tile(np.array([
            math.sin(2*np.pi*target_month/12), math.cos(2*np.pi*target_month/12),
            ex.horizon/7,
        ], np.float32), (len(loc), 1))
        context = np.column_stack([
            anchor, s.loc_mean[loc], s.loc_std[loc], anchor-s.loc_mean[loc],
            anchor_c, current_c, current_c-anchor_c, path,
            constants, s.static[loc],
        ]).astype(np.float32)
        residual = ((s.target[ci, loc] - anchor) / RESIDUAL_SCALE).astype(np.float32)
        if self.inference:
            residual.fill(0)
        keys = s.locs[loc]
        meta = np.column_stack([
            keys, np.full(len(loc), ex.horizon), np.full(len(loc), ex.group)
        ]).astype(np.int32)
        return tuple(torch.from_numpy(v) for v in (seq, context, loc, residual, anchor, meta))


class TemporalLSTM(nn.Module):
    def __init__(self, n_locations: int, seq_features: int, context_features: int):
        super().__init__()
        self.lstm = nn.LSTM(seq_features, 72, num_layers=2, batch_first=True,
                            dropout=.15, bidirectional=False)
        self.location = nn.Embedding(n_locations, 16)
        self.head = nn.Sequential(
            nn.Linear(72 + 16 + context_features, 192), nn.LayerNorm(192), nn.SiLU(), nn.Dropout(.15),
            nn.Linear(192, 96), nn.SiLU(), nn.Dropout(.1), nn.Linear(96, 1),
        )

    def forward(self, seq, context, loc):
        encoded, _ = self.lstm(seq)
        return self.head(torch.cat([encoded[:, -1], self.location(loc), context], 1)).squeeze(1)


def examples_for_validation(store):
    result, visible, group = [], [], 0
    available = set(np.flatnonzero(np.isfinite(store.tws).any(axis=1)))
    available_months = {int(store.months[i]) for i in available}
    for date, max_h in VAL_BLOCKS:
        anchor = midx(date); visible.append(anchor)
        for horizon in range(1, max_h+1):
            current = anchor+horizon-1
            if current not in available_months: break
            result.append(TemporalExample(anchor,current,horizon,group,tuple(visible)))
            group += 1
    return result


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; hs=[]; gs=[]; ks=[]
    for seq,ctx,loc,y,anchor,meta in loader:
        seq,ctx,loc = seq[0].to(device),ctx[0].to(device),loc[0].to(device)
        pred=model(seq,ctx,loc).float().cpu().numpy()*RESIDUAL_SCALE+anchor[0].numpy()
        truth=y[0].numpy()*RESIDUAL_SCALE+anchor[0].numpy()
        ys.append(truth);ps.append(pred);ks.append(meta[0,:,0].numpy())
        hs.append(meta[0,:,1].numpy());gs.append(meta[0,:,2].numpy())
    return tuple(map(np.concatenate,(ys,ps,hs,gs,ks)))


def validate(data_dir: Path, artifact_dir: Path, epochs: int):
    frame,_,_=load_data(data_dir,False);cut=midx(VAL_BLOCKS[0][0]);store=TemporalStore(frame,cut)
    observed=set(map(int,frame.loc[frame.month_idx<cut,'month_idx'].unique()))
    train_examples=[]
    for current in sorted(observed):
        for horizon in range(1,8):
            anchor=current-horizon+1
            if anchor in observed:train_examples.append(TemporalExample(anchor,current,horizon,-1))
    train_ds=MapBatchDataset(store,train_examples,cut,2048,True)
    val_ds=MapBatchDataset(store,examples_for_validation(store),cut,999999,False)
    sample=train_ds[0];seq_f=sample[0].shape[-1];ctx_f=sample[1].shape[-1]
    print('examples',len(train_ds),len(val_ds),'features',seq_f,ctx_f,flush=True)
    train_loader=DataLoader(train_ds,batch_size=1,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TemporalLSTM(len(store.locs),seq_f,ctx_f).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-3,weight_decay=2e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');best=9.;stale=0
    artifact_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train();losses=[]
        for seq,ctx,loc,y,_,_ in train_loader:
            seq,ctx,loc,y=seq[0].to(device),ctx[0].to(device),loc[0].to(device),y[0].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                pred=model(seq,ctx,loc);loss=(pred-y).square().mean()
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),2.)
            scaler.step(opt);scaler.update();losses.append(loss.detach().item())
        scheduler.step();truth,pred,h,g,k=evaluate(model,val_loader,device);score=rmse(truth,pred)
        detail={z:round(rmse(truth[h==z],pred[h==z]),5) for z in range(1,8)}
        print('epoch',epoch,'loss',round(float(np.mean(losses)),5),'val',round(score,6),detail,flush=True)
        if score<best-2e-4:
            best=score;stale=0;torch.save({'state':model.state_dict(),'seq_features':seq_f,'context_features':ctx_f},artifact_dir/'lstm_val.pt')
            np.savez_compressed(artifact_dir/'lstm_val_predictions.npz',truth=truth,prediction=pred,horizon=h,group=g,key=k)
        else:
            stale+=1
            if stale>=4:break
    print('best',best)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate']);p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--artifact-dir',type=Path,default=Path('artifacts_lstm'));p.add_argument('--epochs',type=int,default=12);a=p.parse_args()
    seed_everything(SEED);random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED);torch.backends.cudnn.benchmark=True
    validate(a.data_dir,a.artifact_dir,a.epochs)


if __name__=='__main__':main()
