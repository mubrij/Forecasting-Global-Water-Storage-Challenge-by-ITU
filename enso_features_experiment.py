#!/usr/bin/env python3
"""Walk-forward A/B test of lagged NOAA Niño 3.4 teleconnection features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from map_unet_solution import VAL_BLOCKS, midx
from spatial_experiment import add_spatial_x, augment, make_pair
from winning_solution import (
    SEED, ModelConfig, fit_history_stats, lgb_params, load_data, pair_to_xy, rmse,
)


def load_nino34(path: Path) -> dict[int, float]:
    values = {}
    for line in path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) < 13 or not fields[0].isdigit():
            continue
        year = int(fields[0])
        for month, raw in enumerate(fields[1:13], 1):
            value = float(raw)
            if value > -90:
                values[year * 12 + month - 1] = value
    # Remove the large deterministic SST seasonal cycle using a pre-validation
    # 1981-2010 climatology. This baseline is fixed before all validation/test dates.
    climatology = {
        month: np.mean([value for key, value in values.items()
                        if 1981 <= key // 12 <= 2010 and key % 12 == month])
        for month in range(12)
    }
    return {key: value - climatology[key % 12] for key, value in values.items()}


def add_enso(x: pd.DataFrame, pair: pd.DataFrame, index: dict[int, float]) -> pd.DataFrame:
    out = x.copy()
    current = pair.month_idx.to_numpy(np.int32)
    lagged = {}
    # Lag one month is the newest operationally safe value. Longer lags expose
    # ENSO persistence and phase change without using future information.
    for lag in (1, 2, 3, 6, 12):
        values = np.asarray([index.get(int(month - lag), np.nan) for month in current], np.float32)
        lagged[lag] = values
        out[f"nino34_lag{lag}"] = values
    out["nino34_recent_mean"] = np.nanmean(np.column_stack([lagged[1], lagged[2], lagged[3]]), axis=1)
    out["nino34_phase_change"] = lagged[1] - lagged[3]
    out["nino34_annual_change"] = lagged[1] - lagged[12]
    enso = lagged[1]
    # Explicit teleconnection interactions help a shallow tree express that
    # ENSO has opposite effects in different regions.
    for spatial in ("lat_sin", "lat_cos", "lon_sin", "lon_cos", "lon2_sin", "lon2_cos"):
        out[f"nino34_x_{spatial}"] = enso * out[spatial].to_numpy(np.float32)
    out["nino34_x_month_sin"] = enso * out.target_month_sin.to_numpy(np.float32)
    out["nino34_x_month_cos"] = enso * out.target_month_cos.to_numpy(np.float32)
    out["nino34_x_horizon"] = enso * out.horizon.to_numpy(np.float32)
    return out


def feature_frame(pair, stats, horizon: int, enso: dict[int, float] | None):
    x, residual, truth = pair_to_xy(pair, stats, horizon)
    x = add_spatial_x(x, pair)
    if enso is not None:
        x = add_enso(x, pair, enso)
    return x, residual, truth


def training_set(fit, stats, enso, rows_per_horizon):
    xs, ys = [], []
    for horizon in range(1, 8):
        pair = make_pair(fit, horizon)
        if len(pair) > rows_per_horizon:
            pair = pair.sample(rows_per_horizon, random_state=SEED + horizon).reset_index(drop=True)
        x, residual, _ = feature_frame(pair, stats, horizon, enso)
        xs.append(x); ys.append(residual)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def validation_set(frame, stats, anchor, max_horizon, enso):
    xs, ys, hs = [], [], []
    for horizon in range(1, max_horizon + 1):
        pair = make_pair(frame, horizon)
        pair = pair[pair.month_idx == anchor + horizon - 1].reset_index(drop=True)
        if pair.empty:
            break
        x, _, truth = feature_frame(pair, stats, horizon, enso)
        xs.append(x); ys.append(truth); hs.append(np.full(len(pair), horizon, np.int8))
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(hs)


def train_predict(tx, ty, vx, iterations, n_jobs):
    config = ModelConfig(direct_estimators=iterations, learning_rate=.03, num_leaves=72,
                         min_child_samples=400, n_jobs=n_jobs)
    model = lgb.LGBMRegressor(**lgb_params(config, iterations))
    model.fit(tx, ty, callbacks=[lgb.log_evaluation(0)])
    return vx.anchor_tws.to_numpy(np.float32) + model.predict(vx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("external_noaa/nina34.data"))
    parser.add_argument("--rows-per-horizon", type=int, default=40000)
    parser.add_argument("--iterations", type=int, default=90)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_enso_features"))
    args = parser.parse_args()
    enso = load_nino34(args.index)
    frame, _, _ = load_data(Path("."), need_test=False)
    frame = augment(frame)
    arrays = {name: [] for name in ("truth", "horizon", "block", "baseline", "enso")}
    records = []
    for block, (date, max_horizon) in enumerate(VAL_BLOCKS):
        anchor = midx(date)
        fit = frame[frame.month_idx < anchor].copy()
        stats = fit_history_stats(fit)
        block_result = {}
        truth = horizons = None
        for name, index in (("baseline", None), ("enso", enso)):
            tx, ty = training_set(fit, stats, index, args.rows_per_horizon)
            vx, truth, horizons = validation_set(frame, stats, anchor, max_horizon, index)
            block_result[name] = train_predict(tx, ty, vx, args.iterations, args.n_jobs)
            print(f"block={block} {date} {name} rmse={rmse(truth, block_result[name]):.9f}", flush=True)
        records.append({"block": block, "date": date,
                        **{name: rmse(truth, pred) for name, pred in block_result.items()}})
        arrays["truth"].append(truth); arrays["horizon"].append(horizons)
        arrays["block"].append(np.full(len(truth), block, np.int8))
        for name in ("baseline", "enso"):
            arrays[name].append(block_result[name])
    final = {name: np.concatenate(parts) for name, parts in arrays.items()}
    summary = {
        "overall": {name: rmse(final["truth"], final[name]) for name in ("baseline", "enso")},
        "blocks": records,
        "horizon": {str(h): {name: rmse(final["truth"][final["horizon"] == h],
                                          final[name][final["horizon"] == h])
                             for name in ("baseline", "enso")}
                    for h in range(1, 8) if np.any(final["horizon"] == h)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "walkforward_predictions.npz", **final)
    (args.output_dir / "validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
