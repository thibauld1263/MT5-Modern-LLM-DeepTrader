"""
DeepTrader — Temporal Alignment
Aligns M30, H1, and H4 candles so that at any M30 bar,
we know exactly which H1 and H4 bars are available as context.

This is critical for Phase 2 where the model receives multi-TF context.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def align_timeframes(
    m30_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensure all three DataFrames cover the same date range.
    Returns trimmed copies.
    """
    # Find common date range
    start = max(m30_df.index[0], h1_df.index[0], h4_df.index[0])
    end = min(m30_df.index[-1], h1_df.index[-1], h4_df.index[-1])

    m30 = m30_df.loc[start:end].copy()
    h1 = h1_df.loc[start:end].copy()
    h4 = h4_df.loc[start:end].copy()

    return m30, h1, h4


def get_htf_context_indices(
    m30_timestamps: pd.DatetimeIndex,
    htf_timestamps: pd.DatetimeIndex,
    context_window: int
) -> np.ndarray:
    """
    For each M30 bar, find the indices of the last `context_window`
    higher-timeframe bars that closed BEFORE this M30 bar.

    Returns: (len(m30), context_window) array of integer indices into htf_timestamps.
             Value -1 means padding (not enough history yet).
    """
    n_m30 = len(m30_timestamps)
    indices = np.full((n_m30, context_window), -1, dtype=np.int64)

    # For each M30 bar, binary search to find the last HTF bar <= this M30 time
    htf_times = htf_timestamps.values.astype(np.int64)
    m30_times = m30_timestamps.values.astype(np.int64)

    for i in range(n_m30):
        # Find the last HTF bar that closed at or before this M30 bar
        pos = np.searchsorted(htf_times, m30_times[i], side="right") - 1
        if pos < 0:
            continue

        # Take the last `context_window` bars ending at `pos`
        start_pos = max(0, pos - context_window + 1)
        n_bars = pos - start_pos + 1

        # Fill from the right (pad left with -1 if not enough history)
        indices[i, context_window - n_bars:] = np.arange(start_pos, pos + 1)

    return indices


def build_aligned_dataset(
    m30_features: np.ndarray,
    h1_features: np.ndarray,
    h4_features: np.ndarray,
    m30_timestamps: pd.DatetimeIndex,
    h1_timestamps: pd.DatetimeIndex,
    h4_timestamps: pd.DatetimeIndex,
    h1_context_window: int = 100,
    h4_context_window: int = 50
) -> Dict[str, np.ndarray]:
    """
    Build the aligned dataset for Phase 2 training.

    Returns dict with:
        - m30_features: (N, 17) full M30 feature array
        - h1_indices: (N, h1_context_window) indices to slice H1 context
        - h4_indices: (N, h4_context_window) indices to slice H4 context
    """
    print("  Computing H1 context alignment...")
    h1_indices = get_htf_context_indices(m30_timestamps, h1_timestamps, h1_context_window)

    print("  Computing H4 context alignment...")
    h4_indices = get_htf_context_indices(m30_timestamps, h4_timestamps, h4_context_window)

    return {
        "m30_features": m30_features,
        "h1_features": h1_features,
        "h4_features": h4_features,
        "h1_indices": h1_indices,
        "h4_indices": h4_indices,
    }
