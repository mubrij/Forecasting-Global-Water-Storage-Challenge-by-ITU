#!/usr/bin/env python3
"""Leakage-safe reconstruction of historical TWS maps from training labels.

For a training row dated month t, Target is the exact TWS value at t+1.  The
shifted labels therefore recover some historical maps that are absent from the
TWS_t column.  This module exposes those maps only as history; it never creates
synthetic climate/model-training rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HISTORY_COLUMNS = ["time", "month_idx", "loc_key", "TWS_t", "Target"]


def reconstruct_tws_history(
    frame: pd.DataFrame,
    before_month: int | None = None,
) -> pd.DataFrame:
    """Return deduplicated observed and label-recovered historical TWS rows.

    Parameters
    ----------
    frame:
        Original labelled training rows.  Target at t is known to equal TWS at
        t+1 in this competition.
    before_month:
        Optional exclusive calendar-month cutoff.  Applying it after shifting
        prevents a label at t from exposing the validation month t+1.
    """
    original = frame[HISTORY_COLUMNS].copy()
    original["_source_priority"] = np.int8(0)

    recovered = frame[HISTORY_COLUMNS].copy()
    recovered["time"] = recovered["time"] + pd.offsets.MonthBegin(1)
    recovered["month_idx"] = recovered["month_idx"].to_numpy(np.int32) + 1
    recovered["TWS_t"] = recovered["Target"].to_numpy(np.float32)
    # fit_history_stats uses this only to establish a broad clipping range.
    recovered["Target"] = recovered["TWS_t"]
    recovered["_source_priority"] = np.int8(1)

    history = pd.concat([original, recovered], ignore_index=True)
    if before_month is not None:
        history = history[history["month_idx"] < int(before_month)]
    history = history[np.isfinite(history["TWS_t"].to_numpy(np.float32))]
    history = history.sort_values(
        ["month_idx", "loc_key", "_source_priority"], kind="stable"
    ).drop_duplicates(["month_idx", "loc_key"], keep="first")
    return history.drop(columns="_source_priority").reset_index(drop=True)


def append_observed_anchor(history: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Add one currently observed TWS map to a historical lookup table."""
    extra = anchor[["time", "month_idx", "loc_key", "TWS_t", "Target"]].copy()
    extra = extra[np.isfinite(extra["TWS_t"].to_numpy(np.float32))]
    out = pd.concat([history, extra], ignore_index=True)
    return out.drop_duplicates(["month_idx", "loc_key"], keep="last")
