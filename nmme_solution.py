#!/usr/bin/env python3
"""NOAA NMME multi-model next-month forecast covariates."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

from cfsv2_solution import CFSFeatures, nearest_indices
from ensemble_solution import ANCHOR_COLUMNS, observed_anchor_months
from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    CLIMATE, SEED, ModelConfig, build_feature_frame, fit_history_stats,
    lgb_params, load_data, pair_to_xy, rmse, seed_everything,
)


NMME_START = 2011 * 12 + 7  # first real-time issue: August 2011
NMME_VARIABLES = {"prate": "prate", "soilm": "sm"}


def issue_stamp(month_idx: int) -> str:
    return f"{month_idx // 12:04d}{month_idx % 12 + 1:02d}0800"


def forecast_url(month_idx: int, variable: str) -> str:
    stamp = issue_stamp(month_idx)
    name = f"{variable}.{stamp}.ENSMEAN.ensmean.anom.1x1.grb"
    return f"https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/{stamp}/{name}"


def cache_path(cache_dir: Path, month_idx: int, variable: str) -> Path:
    return cache_dir / "grib" / f"{variable}_{issue_stamp(month_idx)}.grb"


def download_one(month_idx: int, variable: str, cache_dir: Path) -> tuple[int, str, str]:
    destination = cache_path(cache_dir, month_idx, variable)
    if destination.exists() and destination.stat().st_size > 10_000:
        return month_idx, variable, "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".grb.part")
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                forecast_url(month_idx, variable), headers={"User-Agent": "tws-nmme/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            temporary.replace(destination)
            return month_idx, variable, "downloaded"
        except (OSError, urllib.error.URLError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                return month_idx, variable, f"unavailable:{exc}"
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def download_all(months: list[int], cache_dir: Path, workers: int) -> None:
    jobs = [(month, variable) for month in months for variable in NMME_VARIABLES]
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, month, variable, cache_dir) for month, variable in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            month, variable, status = future.result()
            counts[status.split(":", 1)[0]] = counts.get(status.split(":", 1)[0], 0) + 1
            print(f"[{index:03d}/{len(jobs):03d}] {issue_stamp(month)} {variable} {status}", flush=True)
    print(counts, flush=True)


def decode(path: Path, variable: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    data = xr.open_dataset(
        path, engine="cfgrib", backend_kwargs={"indexpath": "", "errors": "ignore"}
    )
    name = NMME_VARIABLES[variable]
    lat_index = nearest_indices(data.latitude.values, lat)
    lon_index = nearest_indices(data.longitude.values, lon % 360, circular=True)
    # The first 30-day interval in these CPC files is encoded as hours 720--1440:
    # it is the month immediately following the issue month.
    result = data[name].values[0, lat_index, lon_index].astype(np.float32)
    data.close()
    return result


def build_cache(frame: pd.DataFrame, months: list[int], cache_dir: Path) -> Path:
    locations = frame[["loc_key", "lat", "lon"]].drop_duplicates("loc_key").sort_values("loc_key")
    valid_months, values = [], []
    for month in months:
        paths = {variable: cache_path(cache_dir, month, variable) for variable in NMME_VARIABLES}
        if not paths["prate"].exists():
            continue
        columns = []
        for variable in NMME_VARIABLES:
            if paths[variable].exists():
                columns.append(decode(
                    paths[variable], variable,
                    locations.lat.to_numpy(), locations.lon.to_numpy(),
                ))
            else:
                columns.append(np.full(len(locations), np.nan, np.float32))
        values.append(np.column_stack(columns))
        valid_months.append(month)
        print(f"decoded {issue_stamp(month)}", flush=True)
    output = cache_dir / "features_nmme.npz"
    np.savez_compressed(
        output, months=np.asarray(valid_months, np.int32),
        locs=locations.loc_key.to_numpy(np.int32),
        names=np.asarray(list(NMME_VARIABLES)), values=np.asarray(values, np.float32),
    )
    print(f"saved {output}: {np.asarray(values).shape}")
    return output


class NMMEFeatures:
    def __init__(self, path: Path):
        raw = np.load(path)
        self.months = raw["months"].astype(np.int32)
        self.locs = raw["locs"].astype(np.int32)
        self.names = list(map(str, raw["names"]))
        self.values = raw["values"].astype(np.float32)
        self.mean = np.nanmean(self.values, axis=1)
        self.std = np.maximum(np.nanstd(self.values, axis=1), 1e-6)
        self.month_pos = {int(value): index for index, value in enumerate(self.months)}
        self.loc_pos = np.full(int(self.locs.max()) + 1, -1, np.int32)
        self.loc_pos[self.locs] = np.arange(len(self.locs), dtype=np.int32)

    def add(self, x: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        mi = np.asarray([self.month_pos.get(int(value), -1) for value in pair.month_idx], np.int32)
        li = self.loc_pos[pair.loc_key.to_numpy(np.int32)]
        valid = (mi >= 0) & (li >= 0)
        data = np.full((len(pair), len(self.names)), np.nan, np.float32)
        data[valid] = self.values[mi[valid], li[valid]]
        for index, name in enumerate(self.names):
            out[f"nmme_{name}_anom"] = data[:, index]
            z = np.full(len(pair), np.nan, np.float32)
            z[valid] = (data[valid, index] - self.mean[mi[valid], index]) / self.std[mi[valid], index]
            out[f"nmme_{name}_spatial_z"] = np.clip(z, -8, 8)
        out["nmme_wet_storage"] = out["nmme_prate_spatial_z"] + out["nmme_soilm_spatial_z"]
        return out


def add_forecasts(x, pair, nmme: NMMEFeatures, cfs: CFSFeatures | None):
    out = nmme.add(add_spatial_x(x, pair), pair)
    return cfs.add(out, pair) if cfs is not None else out


def build_validation(frame, stats, nmme, cfs):
    xs, residuals, truths, horizons, groups, keys = [], [], [], [], [], []
    group = 0
    for date, max_horizon in VAL_BLOCKS:
        anchor = midx(date)
        for horizon in range(1, max_horizon + 1):
            pair = make_pair(frame, horizon)
            pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
            x, residual, truth = pair_to_xy(pair, stats, horizon)
            xs.append(add_forecasts(x, pair, nmme, cfs))
            residuals.append(residual); truths.append(truth)
            horizons.append(np.full(len(pair), horizon, np.int8))
            groups.append(np.full(len(pair), group, np.int16))
            keys.append(pair.loc_key.to_numpy(np.int32))
            group += 1
    return (
        pd.concat(xs, ignore_index=True), np.concatenate(residuals),
        np.concatenate(truths), np.concatenate(horizons),
        np.concatenate(groups), np.concatenate(keys),
    )


def training_rows(frame, stats, nmme, cfs, rows_per_horizon):
    available = set(map(int, nmme.months))
    if cfs is not None:
        available &= set(map(int, cfs.months))
    xs, ys = [], []
    for horizon in range(1, 8):
        pair = make_pair(frame, horizon)
        pair = pair[pair.month_idx.isin(available)].reset_index(drop=True)
        x, residual, _ = pair_to_xy(pair, stats, horizon)
        if len(pair) > rows_per_horizon:
            take = np.random.RandomState(SEED + horizon).choice(len(pair), rows_per_horizon, False)
            pair, x, residual = pair.iloc[take], x.iloc[take], residual[take]
        xs.append(add_forecasts(x, pair, nmme, cfs)); ys.append(residual)
        print(f"h={horizon}: {len(pair):,}", flush=True)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def validate(data_dir: Path, cache_dir: Path, cfs_cache: Path | None,
             artifact_dir: Path, n_jobs: int) -> None:
    frame, _, _ = load_data(data_dir, need_test=False)
    frame = augment(frame)
    fit = frame[frame.month_idx < midx(VAL_BLOCKS[0][0])].copy()
    stats = fit_history_stats(fit)
    nmme = NMMEFeatures(cache_dir / "features_nmme.npz")
    cfs = CFSFeatures(cfs_cache) if cfs_cache else None
    vx, vr, truth, horizons, groups, keys = build_validation(frame, stats, nmme, cfs)
    tx, ty = training_rows(fit, stats, nmme, cfs, 60_000)
    config = ModelConfig(
        direct_estimators=500, learning_rate=.025, num_leaves=72,
        min_child_samples=350, n_jobs=n_jobs,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, 500))
    model.fit(tx, ty, eval_set=[(vx, vr)], callbacks=[
        lgb.early_stopping(100), lgb.log_evaluation(25),
    ])
    prediction = vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)
    result = {
        "standalone_rmse": rmse(truth, prediction),
        "best_iteration": int(model.best_iteration_),
        "by_horizon": {str(h): rmse(truth[horizons == h], prediction[horizons == h]) for h in range(1, 8)},
        "with_cfs": cfs is not None,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "validation_predictions.npz", truth=truth,
        prediction=prediction, horizon=horizons, group=groups, key=keys,
    )
    (artifact_dir / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

def build_test(frame: pd.DataFrame, stats, nmme: NMMEFeatures,
               cfs: CFSFeatures | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe test rows using the most recent observed TWS anchor."""
    test = augment(frame).copy()
    test["_row_id"] = np.arange(len(test), dtype=np.int64)
    anchors = observed_anchor_months(test)
    by_month = {int(month): rows for month, rows in test.groupby("month_idx", sort=False)}
    pairs, features = [], []
    for month in sorted(by_month):
        current = by_month[month].copy()
        anchor_month = max(value for value in anchors if value <= month)
        anchor = by_month[anchor_month][["loc_key", *ANCHOR_COLUMNS]].rename(
            columns={column: f"anchor_{column}" for column in ANCHOR_COLUMNS}
        )
        pair = current.merge(anchor, on="loc_key", how="left", validate="one_to_one")
        horizon = np.full(len(pair), min(7, month + 1 - anchor_month), np.int8)
        observed = np.isfinite(pair["TWS_t"].to_numpy(np.float32))
        for column in ANCHOR_COLUMNS:
            pair.loc[observed, f"anchor_{column}"] = pair.loc[observed, column]
        horizon[observed] = 1

        anchor_frame = pd.DataFrame({"TWS_t": pair["anchor_TWS_t"].to_numpy()})
        for column in CLIMATE:
            anchor_frame[column] = pair[f"anchor_{column}"].to_numpy()
        x = build_feature_frame(pair, stats, anchor=anchor_frame, horizon=horizon)
        features.append(add_forecasts(x, pair, nmme, cfs))
        pair["horizon"] = horizon
        pairs.append(pair)
    return pd.concat(pairs, ignore_index=True), pd.concat(features, ignore_index=True)


def production(train: pd.DataFrame, test: pd.DataFrame, sample_path: Path,
               cache_dir: Path, cfs_cache: Path | None, artifact_dir: Path,
               output: Path, n_jobs: int) -> None:
    """Fit the validation-selected NMME model on all labels and predict test."""
    train = augment(train)
    stats = fit_history_stats(train)
    nmme = NMMEFeatures(cache_dir / "features_nmme.npz")
    cfs = CFSFeatures(cfs_cache) if cfs_cache else None
    tx, ty = training_rows(train, stats, nmme, cfs, 60_000)
    metadata = json.loads((artifact_dir / "validation.json").read_text())
    estimators = max(20, int(metadata["best_iteration"]))
    config = ModelConfig(
        direct_estimators=estimators, learning_rate=.025, num_leaves=72,
        min_child_samples=350, n_jobs=n_jobs,
    )
    model = lgb.LGBMRegressor(**lgb_params(config, estimators))
    model.fit(tx, ty, callbacks=[lgb.log_evaluation(25)])
    pairs, x_test = build_test(test, stats, nmme, cfs)
    prediction = x_test.anchor_tws.to_numpy(np.float32) + model.predict(x_test)
    raw = pd.DataFrame({
        "ID": pairs.ID.to_numpy(), "Target": prediction,
        "_row_id": pairs._row_id.to_numpy(),
    }).sort_values("_row_id")
    sample = pd.read_csv(sample_path)[["ID"]]
    submission = sample.merge(raw[["ID", "Target"]], on="ID", validate="one_to_one")
    if len(submission) != len(sample) or not np.isfinite(submission.Target).all():
        raise ValueError("invalid NMME production submission")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"saved {output}: {len(submission):,} rows; estimators={estimators}", flush=True)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["download", "cache", "validate", "prepare", "production"]
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("external_nmme"))
    parser.add_argument("--cfs-cache", type=Path, default=Path("external_cfsv2/features_cfsv2.npz"))
    parser.add_argument("--without-cfs", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts_nmme"))
    parser.add_argument("--output", type=Path, default=Path("Submission_NMME_Standalone.csv"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()
    seed_everything(SEED)
    train, test, sample_path = load_data(
        args.data_dir,
        need_test=args.command in {"download", "cache", "prepare", "production"},
    )
    months = sorted(set(map(int, train.loc[train.month_idx >= NMME_START, "month_idx"])))
    if test is not None:
        months = sorted(set(months) | set(map(int, test.loc[test.month_idx >= NMME_START, "month_idx"])))
    if args.command in {"download", "prepare"}:
        download_all(months, args.cache_dir, args.workers)
    if args.command in {"cache", "prepare"}:
        build_cache(pd.concat([train, test]) if test is not None else train, months, args.cache_dir)
    if args.command == "validate":
        validate(
            args.data_dir, args.cache_dir,
            None if args.without_cfs else args.cfs_cache,
            args.artifact_dir, args.n_jobs,
        )
    if args.command == "production":
        assert test is not None and sample_path is not None
        production(
            train, test, sample_path, args.cache_dir,
            None if args.without_cfs else args.cfs_cache,
            args.artifact_dir, args.output, args.n_jobs,
        )



if __name__ == "__main__":
    main()
