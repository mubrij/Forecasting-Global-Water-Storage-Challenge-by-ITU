#!/usr/bin/env python3
"""Apply the leaderboard-optimal weight to the V25-to-V26 direction."""
import numpy as np
import pandas as pd


WEIGHT = 0.42348292774123003


def main():
    base = pd.read_csv("Submission_V25_LeaderboardSurfaceOptimal.csv")
    model = pd.read_csv("Submission_V26_FiveModelBlend.csv")
    if not base.ID.equals(model.ID):
        raise ValueError("IDs are not aligned")
    output = base[["ID"]].copy()
    output["Target"] = base.Target.to_numpy(np.float64) + WEIGHT * (
        model.Target.to_numpy(np.float64) - base.Target.to_numpy(np.float64)
    )
    output.to_csv("Submission_V27_CalibratedFiveModelBlend.csv", index=False)
    print(f"saved {len(output):,} rows; weight={WEIGHT}")


if __name__ == "__main__":
    main()
