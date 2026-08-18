#!/usr/bin/env python3
"""Validate public NOAA water-balance covariates for TWS forecasting.

Source: NOAA/PSL NCEP-NCAR Reanalysis 1 monthly surface Gaussian fields.
Only forcing through the current predictor month is used.  No GRACE/TWS data is
read from the external source.
"""
from __future__ import annotations

import gc
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED, ModelConfig, fit_history_stats, lgb_params, load_data, pair_to_xy, rmse,
)

NOAA_FILES = {
    "prate": "external_noaa/prate.nc",
    "pevpr": "external_noaa/pevpr.nc",
    "runof": "external_noaa/runof.nc",
    "weasd": "external_noaa/weasd.nc",
}


def build_cache(frame: pd.DataFrame, path: Path) -> dict[str, np.ndarray]:
    if path.exists():
        return dict(np.load(path))
    locs = np.sort(frame.loc_key.unique()).astype(np.int32)
    start, end = int(frame.month_idx.min()) - 6, int(frame.month_idx.max()) + 1
    months = np.arange(start, end + 1, dtype=np.int32)
    dates = pd.to_datetime([f"{m//12:04d}-{m%12+1:02d}-01" for m in months])
    lat = (locs // 360 - 55.5).astype(np.float32)
    lon = ((locs % 360 - 179.5) % 360).astype(np.float32)
    lat_points = xr.DataArray(lat, dims="point")
    lon_points = xr.DataArray(lon, dims="point")
    result: dict[str, np.ndarray] = {"months": months, "locs": locs}
    for name, filename in NOAA_FILES.items():
        source = xr.open_dataset(filename)[name]
        # Linear spatial interpolation, exact monthly selection.
        raw = source.sel(time=xr.DataArray(dates, dims="date")).interp(
            lat=lat_points, lon=lon_points
        ).transpose("date", "point").values.astype(np.float32)
        reference = source.sel(time=slice("1981-01-01", "2010-12-01"))
        clim_mean = reference.groupby("time.month").mean("time").interp(
            lat=lat_points, lon=lon_points
        ).transpose("month", "point").values.astype(np.float32)
        clim_std = reference.groupby("time.month").std("time").interp(
            lat=lat_points, lon=lon_points
        ).transpose("month", "point").values.astype(np.float32)
        month_of_year = months % 12
        anomaly = raw - clim_mean[month_of_year]
        zscore = anomaly / np.maximum(clim_std[month_of_year], 1e-5)
        result[f"{name}_raw"] = raw
        result[f"{name}_anom"] = anomaly.astype(np.float32)
        result[f"{name}_z"] = np.clip(zscore, -8, 8).astype(np.float32)
        print(name, raw.shape, "z std", float(np.nanstd(zscore)), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)
    return result


class NOAAFeatures:
    def __init__(self, cache):
        self.cache = cache
        self.month0 = int(cache["months"][0])
        max_key = int(cache["locs"].max())
        self.loc_pos = np.full(max_key + 1, -1, np.int32)
        self.loc_pos[cache["locs"].astype(np.int32)] = np.arange(len(cache["locs"]))

    def add(self, x: pd.DataFrame, pair: pd.DataFrame, horizon: int) -> pd.DataFrame:
        out = x.copy()
        ti = pair.month_idx.to_numpy(np.int32) - self.month0
        li = self.loc_pos[pair.loc_key.to_numpy(np.int32)]
        path_index = ti[:, None] - np.arange(horizon - 1, -1, -1)[None, :]
        gathered = {}
        for name in NOAA_FILES:
            z = self.cache[f"{name}_z"][path_index, li[:, None]]
            anom = self.cache[f"{name}_anom"][path_index, li[:, None]]
            gathered[name] = anom
            out[f"noaa_{name}_current_z"] = z[:, -1]
            out[f"noaa_{name}_path_mean_z"] = z.mean(axis=1)
            out[f"noaa_{name}_path_min_z"] = z.min(axis=1)
            out[f"noaa_{name}_path_max_z"] = z.max(axis=1)
            out[f"noaa_{name}_path_sum_anom"] = anom.sum(axis=1)
        # Reanalysis units differ, so expose both components and an approximate
        # standardized water-balance direction rather than imposing exact units.
        p = self.cache["prate_z"][path_index, li[:, None]]
        e = self.cache["pevpr_z"][path_index, li[:, None]]
        r = self.cache["runof_z"][path_index, li[:, None]]
        snow = self.cache["weasd_z"][path_index, li[:, None]]
        out["noaa_balance_z_sum"] = (p - e - r).sum(axis=1)
        out["noaa_snow_z_change"] = snow[:, -1] - snow[:, 0]
        out["noaa_wet_input_z_sum"] = (p + snow).sum(axis=1)
        return out


def main():
    frame, _, _ = load_data(Path("."), need_test=False)
    cache = build_cache(frame, Path("external_noaa/features_train.npz"))
    noaa = NOAAFeatures(cache)
    frame = augment(frame)
    cutoff = midx(VAL_BLOCKS[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)

    val_x, val_r, truth, horizons, groups, keys = [], [], [], [], [], []
    group = 0
    for date, max_h in VAL_BLOCKS:
        anchor = midx(date)
        for horizon in range(1, max_h + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
            if pair.empty:
                break
            x, residual, target = pair_to_xy(pair, stats, horizon)
            x = noaa.add(add_spatial_x(x, pair), pair, horizon)
            val_x.append(x); val_r.append(residual); truth.append(target)
            horizons.append(np.full(len(pair), horizon, np.int8))
            groups.append(np.full(len(pair), group, np.int16))
            keys.append(pair.loc_key.to_numpy(np.int32))
            group += 1
    vx = pd.concat(val_x, ignore_index=True)
    vr, y, vh = np.concatenate(val_r), np.concatenate(truth), np.concatenate(horizons)

    xs, ys = [], []
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        if len(pair) > 100_000:
            take = np.random.RandomState(SEED + horizon).choice(len(pair), 100_000, False)
            pair, x, residual = pair.iloc[take], x.iloc[take], residual[take]
        x = noaa.add(add_spatial_x(x, pair), pair, horizon)
        xs.append(x); ys.append(residual)
        print("h", horizon, len(x), flush=True)
        del pair, x
        gc.collect()
    tx, ty = pd.concat(xs, ignore_index=True), np.concatenate(ys)
    config = ModelConfig(direct_estimators=700, learning_rate=.03, num_leaves=80,
                         min_child_samples=300, n_jobs=20)
    model = lgb.LGBMRegressor(**lgb_params(config, 700))
    model.fit(tx, ty, eval_set=[(vx, vr)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
    pred = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
    print("NOAA overall", rmse(y, pred))
    print({h: rmse(y[vh == h], pred[vh == h]) for h in range(1, 8)})

    np.savez_compressed(
        "external_noaa/validation_predictions.npz", truth=y, prediction=pred,
        horizon=vh, group=np.concatenate(groups), key=np.concatenate(keys),
    )

if __name__ == "__main__":
    main()
