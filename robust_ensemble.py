#!/usr/bin/env python3
"""Build the block-robust spatial/NMME ensemble selected on chronological maps."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from map_unet_production import smooth_submission
from winning_solution import load_data


WEIGHTS = {
    "unet": 0.42,
    "nmme": 0.28,
    "ridge": 0.30,
}


def aligned_values(sample: pd.DataFrame, path: Path) -> np.ndarray:
    frame = sample.merge(pd.read_csv(path), on="ID", validate="one_to_one")
    values = frame["Target"].to_numpy(np.float64)
    if len(values) != len(sample) or not np.isfinite(values).all():
        raise ValueError(f"invalid component submission: {path}")
    return values


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
        "--output", type=Path, default=Path("Submission_V16_BlockRobustForecastBlend.csv")
    )
    args = parser.parse_args()

    _, test, sample_path = load_data(args.data_dir, need_test=True)
    assert test is not None and sample_path is not None
    sample = pd.read_csv(sample_path)[["ID"]]
    unet = smooth_submission(test, aligned_values(sample, args.unet), sigma=1.0)
    nmme = smooth_submission(test, aligned_values(sample, args.nmme), sigma=1.0)
    ridge = aligned_values(sample, args.ridge)

    submission = sample.copy()
    submission["Target"] = (
        WEIGHTS["unet"] * unet
        + WEIGHTS["nmme"] * nmme
        + WEIGHTS["ridge"] * ridge
    )
    if not np.isfinite(submission.Target).all():
        raise ValueError("ensemble contains non-finite predictions")
    submission.to_csv(args.output, index=False)
    print(
        f"saved {args.output}: {len(submission):,} rows; "
        f"mean={submission.Target.mean():.6f}; std={submission.Target.std():.6f}; "
        f"weights={WEIGHTS}",
        flush=True,
    )


if __name__ == "__main__":
    main()
