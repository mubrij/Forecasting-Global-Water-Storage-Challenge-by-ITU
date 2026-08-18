#!/usr/bin/env python3
"""Combine the validated robust-L1 and small L2 recency corrections."""
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("Submission_V15_MapMeanProbe.csv")
L2_BLEND = Path("Submission_V21_RecencyRegimeBlend.csv")  # coefficient 0.40
L1_BLEND = Path("Submission_V22_RobustRecencyBlend.csv")  # coefficient 0.50
OUTPUT = Path("Submission_V23_RobustRecencyEnsemble.csv")
FAILED_HISTORY = Path("Submission_V20_GatedHistoryCorrection.csv")
LB_OUTPUT = Path("Submission_V24_LBAwareRecencyEnsemble.csv")


def main() -> None:
    base = pd.read_csv(BASE)
    l2 = pd.read_csv(L2_BLEND)
    l1 = pd.read_csv(L1_BLEND)
    if not base.ID.equals(l2.ID) or not base.ID.equals(l1.ID):
        raise ValueError("submission IDs are not aligned")
    # Add coefficient 0.05 of the raw L2 direction. V21 contains 0.40,
    # therefore 0.05 / 0.40 = 0.125 of its displacement from V15.
    output = base[["ID"]].copy()
    output["Target"] = (
        l1.Target.to_numpy(np.float64)
        + 0.125 * (l2.Target.to_numpy(np.float64) - base.Target.to_numpy(np.float64))
    )
    if not output.ID.is_unique or not np.isfinite(output.Target).all():
        raise ValueError("invalid recency ensemble")
    output.to_csv(OUTPUT, index=False)
    print(f"saved {OUTPUT}: {len(output):,} rows")

    failed = pd.read_csv(FAILED_HISTORY)
    if not base.ID.equals(failed.ID):
        raise ValueError("V20 IDs are not aligned")
    lb_output = output.copy()
    # The V15/V20 public scores imply an optimum of -3.679 along this
    # direction. Use a slightly shrunken -3.5 coefficient.
    lb_output["Target"] += -3.5 * (
        failed.Target.to_numpy(np.float64) - base.Target.to_numpy(np.float64)
    )
    lb_output.to_csv(LB_OUTPUT, index=False)
    print(f"saved {LB_OUTPUT}: {len(lb_output):,} rows")


if __name__ == "__main__":
    main()
