#!/usr/bin/env python3
"""Build the horizon-aware block-robust forecast ensemble."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_solution import observed_anchor_months
from map_unet_production import smooth_submission
from robust_ensemble import aligned_values
from winning_solution import load_data


GLOBAL_WEIGHTS = np.asarray([0.42, 0.28, 0.30], np.float64)
WEIGHTS_BY_HORIZON = {
    1: np.asarray([0.145, 0.495, 0.360]),
    2: np.asarray([0.480, 0.290, 0.230]),
    3: np.asarray([0.710, 0.190, 0.100]),
    **{horizon: GLOBAL_WEIGHTS for horizon in range(4, 8)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--unet", type=Path,
        default=Path("artifacts_map_full/Submission_MapUNet_Standalone.csv"),
    )
    parser.add_argument(
        "--nmme", type=Path,
        default=Path("artifacts_nmme_only/Submission_NMME_Standalone.csv"),
    )
    parser.add_argument(
        "--ridge", type=Path,
        default=Path("artifacts_global_ridge/Submission_GlobalRidge_Sigma2.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("Submission_V17_HorizonRobustForecastBlend.csv"),
    )
    args = parser.parse_args()

    _, raw_test, sample_path = load_data(args.data_dir, need_test=True)
    assert raw_test is not None and sample_path is not None
    sample = pd.read_csv(sample_path)[["ID"]]
    test = sample.merge(raw_test, on="ID", validate="one_to_one")
    components = np.column_stack([
        smooth_submission(test, aligned_values(sample, args.unet), sigma=1.0),
        smooth_submission(test, aligned_values(sample, args.nmme), sigma=1.0),
        aligned_values(sample, args.ridge),
    ])

    anchors = observed_anchor_months(test)
    month_anchor = {
        int(month): max(anchor for anchor in anchors if anchor <= int(month))
        for month in test.month_idx.unique()
    }
    horizon = np.asarray([
        min(7, int(month) + 1 - month_anchor[int(month)])
        for month in test.month_idx
    ], np.int8)
    horizon[np.isfinite(test.TWS_t.to_numpy(np.float32))] = 1
    row_weights = np.stack([WEIGHTS_BY_HORIZON[int(value)] for value in horizon])

    submission = sample.copy()
    submission["Target"] = np.einsum("ij,ij->i", components, row_weights)
    if not np.isfinite(submission.Target).all() or not submission.ID.is_unique:
        raise ValueError("invalid horizon-aware submission")
    submission.to_csv(args.output, index=False)
    print(
        f"saved {args.output}: {len(submission):,} rows; "
        f"mean={submission.Target.mean():.6f}; std={submission.Target.std():.6f}; "
        f"horizons={dict(zip(*np.unique(horizon, return_counts=True)))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
