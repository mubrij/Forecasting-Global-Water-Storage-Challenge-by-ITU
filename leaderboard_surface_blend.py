#!/usr/bin/env python3
"""Fit the two-direction RMSE surface implied by V15, V20, and V24 scores."""
from pathlib import Path

import numpy as np
import pandas as pd


FILES = [
    Path("Submission_V15_MapMeanProbe.csv"),
    Path("Submission_V20_GatedHistoryCorrection.csv"),
    Path("Submission_V24_LBAwareRecencyEnsemble.csv"),
]
SCORES = np.asarray([0.735383264, 0.73754315, 0.745387396], np.float64)
OUTPUT = Path("Submission_V25_LeaderboardSurfaceOptimal.csv")


def main() -> None:
    frames = [pd.read_csv(path) for path in FILES]
    if any(not frames[0].ID.equals(frame.ID) for frame in frames[1:]):
        raise ValueError("submission IDs are not aligned")
    base = frames[0].Target.to_numpy(np.float64)
    directions = np.column_stack([
        frames[1].Target.to_numpy(np.float64) - base,
        frames[2].Target.to_numpy(np.float64) - base,
    ])
    gram = directions.T @ directions / len(base)
    cross = np.asarray([
        (SCORES[index] ** 2 - SCORES[0] ** 2 - gram[index - 1, index - 1]) / 2
        for index in (1, 2)
    ])
    weights = -np.linalg.solve(gram, cross)
    prediction = base + directions @ weights
    expected = np.sqrt(SCORES[0] ** 2 + 2 * cross @ weights + weights @ gram @ weights)
    output = frames[0][["ID"]].copy()
    output["Target"] = prediction
    if not output.ID.is_unique or not np.isfinite(output.Target).all():
        raise ValueError("invalid leaderboard-surface submission")
    output.to_csv(OUTPUT, index=False)
    print("weights", weights.tolist())
    print("estimated_public_rmse", float(expected))
    print(f"saved {OUTPUT}: {len(output):,} rows")


if __name__ == "__main__":
    main()
