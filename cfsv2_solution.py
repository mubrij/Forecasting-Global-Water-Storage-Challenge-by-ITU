#!/usr/bin/env python3
"""Leakage-safe NOAA CFSv2 forecast covariates for global TWS forecasting.

The important distinction from reanalysis is that every CFSv2 field used here
was initialized during predictor month ``t`` and forecasts target month ``t+1``.
It is therefore information that was operationally available at prediction
time, not a hindsight observation of the target month.

The script downloads the archived late-month CFSv2 surface forecast, samples
water-budget fields to the challenge grid, validates on the established
chronological rollout blocks, and can train a full-data production model.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from ensemble_solution import ANCHOR_COLUMNS, observed_anchor_months
from map_blend_experiment import smooth
from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED, ModelConfig, build_feature_frame, fit_history_stats, lgb_params,
    load_data, pair_to_xy, rmse, seed_everything,
)


CFS_START = 2011 * 12 + 3  # operational archive begins April 2011
SURFACE_VARS = [
    "prate", "cpr", "watr", "ssrun", "avg_slhtf", "evbs", "evcw",
    "trans", "sbsno", "srweq", "sdwe", "sde", "cnwat", "avg_t",
]
SOIL_VARS = ["soilw0", "soilw1", "soilw2", "soilw3", "ssw"]
CFS_FEATURES = SURFACE_VARS + SOIL_VARS
USE_STATE_CHANGE = False  # rejected by chronological validation (0.63802 vs 0.63662)


def ym(month_idx: int) -> tuple[int, int]:
    return month_idx // 12, month_idx % 12 + 1


def next_ym(year: int, month: int) -> tuple[int, int]:
    return (year + (month == 12), month % 12 + 1)


def archive_url(month_idx: int, init_day: int = 25, lead: int = 1) -> str:
    year, month = ym(month_idx)
    day = min(init_day, calendar.monthrange(year, month)[1])
    target_year, target_month = ym(month_idx + lead)
    init = f"{year:04d}{month:02d}{day:02d}00"
    target = f"{target_year:04d}{target_month:02d}"
    root = "https://www.ncei.noaa.gov/thredds/fileServer/model-cfs_v2_for_mm"
    return (
        f"{root}/{year:04d}/{year:04d}{month:02d}/{year:04d}{month:02d}{day:02d}/"
        f"{init}/flxf.01.{init}.{target}.avrg.grib.grb2"
    )


def cache_path(cache_dir: Path, month_idx: int, lead: int = 1) -> Path:
    year, month = ym(month_idx)
    suffix = "current" if lead == 0 else "to_next"
    return cache_dir / "grib" / f"cfsv2_{year:04d}{month:02d}_{suffix}.grb2"


def required_months(train: pd.DataFrame, test: pd.DataFrame | None) -> list[int]:
    values = set(map(int, train.loc[train.month_idx >= CFS_START, "month_idx"].unique()))
    if test is not None:
        values.update(map(int, test.loc[test.month_idx >= CFS_START, "month_idx"].unique()))
    return sorted(values)


def download_one(month_idx: int, cache_dir: Path, lead: int = 1, retries: int = 4) -> tuple[int, str]:
    destination = cache_path(cache_dir, month_idx, lead)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return month_idx, "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    # A few archive days are missing. Fall back to an earlier operational
    # initialization in the same predictor month, which remains leakage-safe.
    errors = []
    init_days = (1,) if lead == 0 else (25, 20, 15, 10, 5, 1)
    for init_day in init_days:
        url = archive_url(month_idx, init_day, lead)
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "tws-cfsv2/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                if temporary.stat().st_size < 1_000_000:
                    raise IOError(f"short download: {temporary.stat().st_size} bytes")
                os.replace(temporary, destination)
                return month_idx, f"downloaded-d{init_day:02d}"
            except urllib.error.HTTPError as exc:
                if temporary.exists():
                    temporary.unlink()
                errors.append(f"d{init_day:02d}:{exc.code}")
                if exc.code == 404:
                    break
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
            except (OSError, urllib.error.URLError) as exc:
                if temporary.exists():
                    temporary.unlink()
                errors.append(f"d{init_day:02d}:{exc}")
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
    raise RuntimeError(f"no CFSv2 file for month {month_idx}: {errors}")


def download_all(months: list[int], cache_dir: Path, workers: int, lead: int = 1) -> None:
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, month, cache_dir, lead) for month in months]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                month, status = future.result()
            except RuntimeError as exc:
                print(f"[{done:02d}/{len(months):02d}] unavailable: {exc}", flush=True)
                continue
            counts[status] = counts.get(status, 0) + 1
            year, mon = ym(month)
            print(f"[{done:02d}/{len(months):02d}] {year:04d}-{mon:02d} {status}", flush=True)
    print(counts, flush=True)


def nearest_indices(coordinate: np.ndarray, query: np.ndarray, circular: bool = False) -> np.ndarray:
    coordinate = np.asarray(coordinate, np.float64)
    query = np.asarray(query, np.float64)
    if circular:
        distance = np.abs(((coordinate[:, None] - query[None, :] + 180) % 360) - 180)
    else:
        distance = np.abs(coordinate[:, None] - query[None, :])
    return distance.argmin(axis=0).astype(np.int32)


def decode_month(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    common = {"indexpath": "", "errors": "ignore"}
    surface = xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={**common, "filter_by_keys": {"typeOfLevel": "surface"}},
    )
    lat_idx = nearest_indices(surface.latitude.values, lat)
    lon_idx = nearest_indices(surface.longitude.values, lon % 360, circular=True)
    result = []
    for name in SURFACE_VARS:
        if name not in surface:
            raise KeyError(f"{path.name} lacks {name}")
        result.append(surface[name].values[lat_idx, lon_idx].astype(np.float32))
    surface.close()

    soil = xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={**common, "filter_by_keys": {"shortName": "soilw"}},
    )
    for layer in range(4):
        result.append(soil.soilw.values[layer, lat_idx, lon_idx].astype(np.float32))
    soil.close()
    total = xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={**common, "filter_by_keys": {"shortName": "ssw"}},
    )
    result.append(total.ssw.values[lat_idx, lon_idx].astype(np.float32))
    total.close()
    return np.column_stack(result).astype(np.float32)


def build_cache(frame: pd.DataFrame, months: list[int], cache_dir: Path) -> Path:
    output = cache_dir / "features_cfsv2.npz"
    months = [month for month in months if cache_path(cache_dir, month).exists()]
    locations = (
        frame[["loc_key", "lat", "lon"]].drop_duplicates("loc_key")
        .sort_values("loc_key").reset_index(drop=True)
    )
    values = np.empty((len(months), len(locations), len(CFS_FEATURES)), np.float32)
    current_values = np.full_like(values, np.nan)
    for index, month in enumerate(months):
        path = cache_path(cache_dir, month)
        values[index] = decode_month(
            path, locations.lat.to_numpy(), locations.lon.to_numpy()
        )
        current_path = cache_path(cache_dir, month, lead=0)
        if current_path.exists():
            current_values[index] = decode_month(
                current_path, locations.lat.to_numpy(), locations.lon.to_numpy()
            )
        year, mon = ym(month)
        print(f"decoded {index + 1:02d}/{len(months):02d}: {year:04d}-{mon:02d}", flush=True)
    np.savez_compressed(
        output, months=np.asarray(months, np.int32),
        locs=locations.loc_key.to_numpy(np.int32),
        features=np.asarray(CFS_FEATURES), values=values,
        current_values=current_values,
    )
    print(f"saved {output} {values.shape}", flush=True)
    return output


class CFSFeatures:
    def __init__(self, filename: Path):
        raw = np.load(filename)
        self.months = raw["months"].astype(np.int32)
        self.locs = raw["locs"].astype(np.int32)
        self.names = list(map(str, raw["features"]))
        self.values = raw["values"].astype(np.float32)
        self.current_values = raw["current_values"].astype(np.float32) if "current_values" in raw else np.full_like(self.values, np.nan)
        self.spatial_mean = np.nanmean(self.values, axis=1)
        self.spatial_std = np.maximum(np.nanstd(self.values, axis=1), 1e-6)
        self.current_spatial_mean = np.nanmean(self.current_values, axis=1)
        self.current_spatial_std = np.maximum(np.nanstd(self.current_values, axis=1), 1e-6)
        self.month_pos = {int(value): i for i, value in enumerate(self.months)}
        self.loc_pos = np.full(int(self.locs.max()) + 1, -1, np.int32)
        self.loc_pos[self.locs] = np.arange(len(self.locs), dtype=np.int32)

    def add(self, x: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        mi = np.array([self.month_pos.get(int(v), -1) for v in pair.month_idx], np.int32)
        li = self.loc_pos[pair.loc_key.to_numpy(np.int32)]
        valid = (mi >= 0) & (li >= 0)
        data = np.full((len(pair), len(self.names)), np.nan, np.float32)
        current_data = np.full_like(data, np.nan)
        data[valid] = self.values[mi[valid], li[valid]]
        current_data[valid] = self.current_values[mi[valid], li[valid]]
        for index, name in enumerate(self.names):
            raw = data[:, index]
            out[f"cfs_{name}"] = raw
            if USE_STATE_CHANGE:
                out[f"cfs_current_{name}"] = current_data[:, index]
                out[f"cfs_change_{name}"] = raw - current_data[:, index]
            if name in {"prate", "cpr", "watr", "ssrun", "srweq"}:
                out[f"cfs_log_{name}"] = np.sign(raw) * np.log1p(np.abs(raw))
            zscore = np.full(len(pair), np.nan, np.float32)
            zscore[valid] = (raw[valid] - self.spatial_mean[mi[valid], index]) / self.spatial_std[mi[valid], index]
            out[f"cfs_spatial_z_{name}"] = np.clip(zscore, -8, 8)
        # Forecast water-balance directions; latent heat is an evaporation proxy.
        by = {name: data[:, i] for i, name in enumerate(self.names)}
        out["cfs_evap_flux"] = by["evbs"] + by["evcw"] + by["trans"] + by["sbsno"]
        out["cfs_liquid_input"] = by["prate"] - by["srweq"]
        out["cfs_runoff_total"] = by["watr"] + by["ssrun"]
        out["cfs_soil_mean"] = np.nanmean(
            np.column_stack([by[f"soilw{i}"] for i in range(4)]), axis=1
        ).astype(np.float32)
        out["cfs_soil_deep_minus_surface"] = by["soilw3"] - by["soilw0"]
        current_soil = pair["SOIL_MOISTURE_t"].to_numpy(np.float32)
        out["cfs_surface_soil_minus_input"] = by["soilw0"] - current_soil
        out["cfs_total_soil_per_depth"] = by["ssw"] / 4.0
        out["cfs_wetness_forcing"] = out["cfs_spatial_z_prate"] - out["cfs_spatial_z_avg_slhtf"]
        return out

    def add_hydrologic_path(
        self,
        x: pd.DataFrame,
        pair: pd.DataFrame,
        horizon: int | np.ndarray,
    ) -> pd.DataFrame:
        """Add legal CFS storage evolution from the TWS anchor to current t."""
        out = self.add(x, pair)
        current_month = pair.month_idx.to_numpy(np.int32)
        h = np.broadcast_to(np.asarray(horizon, np.int32), len(pair))
        anchor_month = current_month - h + 1
        loc = self.loc_pos[pair.loc_key.to_numpy(np.int32)]
        cur_pos = np.array([self.month_pos.get(int(v), -1) for v in current_month], np.int32)
        anc_pos = np.array([self.month_pos.get(int(v), -1) for v in anchor_month], np.int32)
        valid_cur = (cur_pos >= 0) & (loc >= 0)
        valid_anc = (anc_pos >= 0) & (loc >= 0)
        storage_names = ["ssw", "soilw0", "soilw1", "soilw2", "soilw3", "srweq", "sdwe", "cnwat"]
        denom = np.maximum(h - 1, 1).astype(np.float32)
        for name in storage_names:
            index = self.names.index(name)
            current = np.full(len(pair), np.nan, np.float32)
            anchor = np.full(len(pair), np.nan, np.float32)
            future = np.full(len(pair), np.nan, np.float32)
            current[valid_cur] = self.current_values[cur_pos[valid_cur], loc[valid_cur], index]
            future[valid_cur] = self.values[cur_pos[valid_cur], loc[valid_cur], index]
            anchor[valid_anc] = self.current_values[anc_pos[valid_anc], loc[valid_anc], index]

            current_z = np.full(len(pair), np.nan, np.float32)
            anchor_z = np.full(len(pair), np.nan, np.float32)
            current_z[valid_cur] = (
                current[valid_cur] - self.current_spatial_mean[cur_pos[valid_cur], index]
            ) / self.current_spatial_std[cur_pos[valid_cur], index]
            anchor_z[valid_anc] = (
                anchor[valid_anc] - self.current_spatial_mean[anc_pos[valid_anc], index]
            ) / self.current_spatial_std[anc_pos[valid_anc], index]

            out[f"cfs_path_current_{name}"] = current
            out[f"cfs_path_anchor_{name}"] = anchor
            out[f"cfs_path_delta_{name}"] = current - anchor
            out[f"cfs_path_velocity_{name}"] = (current - anchor) / denom
            out[f"cfs_path_next_delta_{name}"] = future - current
            out[f"cfs_path_anchor_to_next_{name}"] = future - anchor
            out[f"cfs_path_current_z_{name}"] = np.clip(current_z, -8, 8)
            out[f"cfs_path_delta_z_{name}"] = np.clip(current_z - anchor_z, -12, 12)
        return out


def validation_rows(frame: pd.DataFrame, stats, cfs: CFSFeatures):
    xs, residuals, truths, horizons, groups, keys = [], [], [], [], [], []
    group = 0
    for date, max_h in VAL_BLOCKS:
        anchor = midx(date)
        for horizon in range(1, max_h + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
            if pair.empty:
                break
            x, residual, truth = pair_to_xy(pair, stats, horizon)
            x = cfs.add(add_spatial_x(x, pair), pair)
            xs.append(x); residuals.append(residual); truths.append(truth)
            horizons.append(np.full(len(pair), horizon, np.int8))
            groups.append(np.full(len(pair), group, np.int16))
            keys.append(pair.loc_key.to_numpy(np.int32)); group += 1
    return (
        pd.concat(xs, ignore_index=True), np.concatenate(residuals),
        np.concatenate(truths), np.concatenate(horizons),
        np.concatenate(groups), np.concatenate(keys),
    )


def sampled_training(frame, stats, cfs: CFSFeatures, rows_per_horizon: int):
    xs, ys = [], []
    available = set(map(int, cfs.months))
    for horizon in range(1, 8):
        pair = make_pair(frame, horizon)
        pair = pair[pair.month_idx.isin(available)].reset_index(drop=True)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        if len(pair) > rows_per_horizon:
            take = np.random.RandomState(SEED + horizon).choice(
                len(pair), rows_per_horizon, replace=False
            )
            pair, x, residual = pair.iloc[take], x.iloc[take], residual[take]
        x = cfs.add(add_spatial_x(x, pair), pair)
        xs.append(x); ys.append(residual)
        print(f"h={horizon}: {len(pair):,}", flush=True)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def validate(frame: pd.DataFrame, cache_dir: Path, artifact_dir: Path, n_jobs: int) -> int:
    frame = augment(frame)
    cutoff = midx(VAL_BLOCKS[0][0])
    fit = frame[frame.month_idx < cutoff].copy()
    stats = fit_history_stats(fit)
    cfs = CFSFeatures(cache_dir / "features_cfsv2.npz")
    vx, vr, truth, horizons, groups, keys = validation_rows(frame, stats, cfs)
    tx, ty = sampled_training(fit, stats, cfs, 120_000)
    config = ModelConfig(
        direct_estimators=900, learning_rate=0.025, num_leaves=72,
        min_child_samples=350, n_jobs=n_jobs,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, config.direct_estimators))
    model.fit(
        tx, ty, eval_set=[(vx, vr)],
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(50)],
    )
    pred = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
    print("CFS validation", rmse(truth, pred), flush=True)
    print({h: rmse(truth[horizons == h], pred[horizons == h]) for h in range(1, 8)})
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "validation_predictions.npz", truth=truth, prediction=pred,
        horizon=horizons, group=groups, key=keys,
    )
    detail = select_validation_blend(truth, pred, horizons, groups, keys)
    detail["best_iteration"] = int(model.best_iteration_)
    (artifact_dir / "validation.json").write_text(json.dumps(detail, indent=2))
    print(json.dumps(detail, indent=2), flush=True)
    return int(model.best_iteration_)


def select_validation_blend(truth, cfs_pred, horizons, groups, keys) -> dict:
    aligned = pd.read_pickle("artifacts_map_unet/aligned_validation.pkl")
    cframe = pd.DataFrame({
        "group": groups, "key": keys, "cfs": cfs_pred,
        "cfs_truth": truth, "horizon_cfs": horizons,
    })
    data = aligned.merge(cframe, on=["group", "key"], validate="one_to_one")
    y = data.truth.to_numpy(np.float64)
    group = data.group.to_numpy(); key = data.key.to_numpy()
    tab_s = smooth(data.tabular.to_numpy(), key, group, 2.0)
    unet_s = smooth(data.unet.to_numpy(), key, group, 1.0)
    base = .55 * tab_s + .45 * unet_s
    cfs = data.cfs.to_numpy(np.float64)
    direction = cfs - base
    raw_weight = float(np.dot(y - base, direction) / max(np.dot(direction, direction), 1e-12))
    weights = np.linspace(0, .5, 21)
    rows = []
    for weight in weights:
        scores = []
        for value in np.unique(group):
            mask = group == value
            scores.append(rmse(y[mask], base[mask] + weight * direction[mask]))
        rows.append({
            "weight": float(weight), "rmse": rmse(y, base + weight * direction),
            "worst_block": float(max(scores)), "median_block": float(np.median(scores)),
        })
    # Robust choice: best mean among weights whose worst block is no worse than
    # the base by more than 0.5%; this guards against one lucky validation era.
    base_worst = rows[0]["worst_block"]
    eligible = [row for row in rows if row["worst_block"] <= base_worst * 1.005]
    chosen = min(eligible, key=lambda row: row["rmse"])
    return {
        "cfs_standalone_rmse": rmse(y, cfs), "reference_base_rmse": rmse(y, base),
        "raw_optimal_cfs_weight": raw_weight, "selected": chosen, "grid": rows,
    }


def build_test_pairs(test: pd.DataFrame, stats, cfs: CFSFeatures):
    test = augment(test).copy(); test["_row_id"] = np.arange(len(test), dtype=np.int64)
    anchors = observed_anchor_months(test)
    by_month = {int(key): value for key, value in test.groupby("month_idx", sort=False)}
    pairs, features = [], []
    for month in sorted(by_month):
        current = by_month[month].copy()
        anchor_month = max(value for value in anchors if value <= month)
        anchor = by_month[anchor_month][["loc_key", *ANCHOR_COLUMNS]].rename(
            columns={column: f"anchor_{column}" for column in ANCHOR_COLUMNS}
        )
        pair = current.merge(anchor, on="loc_key", how="left", validate="one_to_one")
        horizon = np.full(len(pair), min(7, month + 1 - anchor_month), np.int8)
        observed = np.isfinite(pair.TWS_t.to_numpy(np.float32))
        for column in ANCHOR_COLUMNS:
            pair.loc[observed, f"anchor_{column}"] = pair.loc[observed, column]
        horizon[observed] = 1
        anchor_frame = pd.DataFrame({"TWS_t": pair.anchor_TWS_t.to_numpy()})
        for column in ["SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"]:
            anchor_frame[column] = pair[f"anchor_{column}"].to_numpy()
        x = build_feature_frame(pair, stats, anchor=anchor_frame, horizon=horizon)
        features.append(cfs.add(add_spatial_x(x, pair), pair))
        pair["horizon"] = horizon; pairs.append(pair)
    return pd.concat(pairs, ignore_index=True), pd.concat(features, ignore_index=True)


def production(frame, test, sample_path: Path, cache_dir: Path, artifact_dir: Path,
               baseline: Path, output: Path, n_jobs: int) -> None:
    train_aug = augment(frame)
    stats = fit_history_stats(train_aug)
    cfs = CFSFeatures(cache_dir / "features_cfsv2.npz")
    tx, ty = sampled_training(train_aug, stats, cfs, 180_000)
    metadata = json.loads((artifact_dir / "validation.json").read_text())
    estimators = max(30, int(metadata["best_iteration"]))
    config = ModelConfig(
        direct_estimators=estimators, learning_rate=.025, num_leaves=72,
        min_child_samples=350, n_jobs=n_jobs,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, estimators))
    model.fit(tx, ty, callbacks=[lgb.log_evaluation(50)])
    pairs, x_test = build_test_pairs(test, stats, cfs)
    standalone = x_test.anchor_tws.to_numpy(np.float32) + model.predict(x_test)
    sample = pd.read_csv(sample_path)[["ID"]]
    raw = pd.DataFrame({"ID": pairs.ID, "Target": standalone, "_row_id": pairs._row_id})
    raw = raw.sort_values("_row_id")[["ID", "Target"]]
    raw = sample.merge(raw, on="ID", validate="one_to_one")
    raw.to_csv(artifact_dir / "Submission_CFSv2_Standalone.csv", index=False)
    base = sample.merge(pd.read_csv(baseline), on="ID", validate="one_to_one")
    weight = float(metadata["selected"]["weight"])
    submission = sample.copy()
    submission["Target"] = (1 - weight) * base.Target.to_numpy() + weight * raw.Target.to_numpy()
    if not np.isfinite(submission.Target).all() or len(submission) != len(sample):
        raise ValueError("invalid production submission")
    submission.to_csv(output, index=False)
    joblib.dump({
        "model": model, "stats": stats, "features": tx.columns.tolist(),
        "cfs_weight": weight, "cfs_feature_names": CFS_FEATURES,
    }, artifact_dir / "cfsv2_model.joblib", compress=3)
    print(f"saved {output}: {len(submission):,} rows; CFS weight={weight:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "download-current", "features", "validate", "production", "all"])
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("external_cfsv2"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_cfsv2"))
    parser.add_argument("--baseline", type=Path, default=Path("Submission_V10_RegularizedMultiBlend.csv"))
    parser.add_argument("--output", type=Path, default=Path("Submission_V11_CFSv2Blend.csv"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    seed_everything(SEED)
    frame, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    months = required_months(frame, test)
    combined = pd.concat([frame[["loc_key", "lat", "lon"]], test[["loc_key", "lat", "lon"]]])
    if args.command in {"download", "all"}:
        download_all(months, args.cache_dir, args.workers)
    if args.command in {"download-current", "all"}:
        download_all(months, args.cache_dir, args.workers, lead=0)
    if args.command in {"features", "all"}:
        build_cache(combined, months, args.cache_dir)
    if args.command in {"validate", "all"}:
        validate(frame, args.cache_dir, args.artifact_dir, args.n_jobs)
    if args.command in {"production", "all"}:
        production(
            frame, test, sample_path, args.cache_dir, args.artifact_dir,
            args.baseline, args.output, args.n_jobs,
        )


if __name__ == "__main__":
    main()
