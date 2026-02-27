"""
DeepTrader — Feature Engine
Converts raw OHLCV candles into 17-dimensional feature vectors.
Each candle becomes a normalized vector that the Transformer can consume.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# Feature indices (for reference)
FEATURE_NAMES = [
    "norm_open", "norm_high", "norm_low", "norm_close",
    "body_ratio", "upper_wick", "lower_wick", "close_position",
    "log_return", "atr_ratio", "rel_volume", "vol_price_corr",
    "gap", "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]

N_FEATURES = len(FEATURE_NAMES)  # 17


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period, min_periods=1).mean()


def encode_candles(df: pd.DataFrame) -> np.ndarray:
    """
    Transform a cleaned OHLCV DataFrame into a (N, 17) feature matrix.

    The 17 features capture:
    - Price structure (normalized OHLC)
    - Candle shape (body, wicks, close position)
    - Momentum (log returns)
    - Volatility (ATR-normalized range)
    - Volume dynamics (relative volume, vol-price correlation)
    - Gaps
    - Time cyclicality (hour/day sin/cos)

    Price normalization is NOT done here — it's done per-window in the Dataset.
    We store raw OHLC so the Dataset can z-score normalize per window.
    """
    eps = 1e-10
    n = len(df)

    features = np.zeros((n, N_FEATURES), dtype=np.float32)

    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    v = df["tick_volume"].values.astype(np.float64)

    # --- Raw OHLC (to be z-score normalized per-window later) ---
    features[:, 0] = o  # norm_open (raw for now)
    features[:, 1] = h  # norm_high
    features[:, 2] = l  # norm_low
    features[:, 3] = c  # norm_close

    # --- Candle shape features (already scale-invariant) ---
    candle_range = h - l + eps

    # Body ratio: how much of the candle is body vs wick (-1 to +1)
    features[:, 4] = (c - o) / candle_range

    # Upper wick ratio (0 to 1)
    max_oc = np.maximum(o, c)
    features[:, 5] = (h - max_oc) / candle_range

    # Lower wick ratio (0 to 1)
    min_oc = np.minimum(o, c)
    features[:, 6] = (min_oc - l) / candle_range

    # Close position in range (0 = closed at low, 1 = closed at high)
    features[:, 7] = (c - l) / candle_range

    # --- Momentum ---
    # Log return
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    features[:, 8] = np.log(c / (prev_close + eps))

    # --- Volatility ---
    atr = compute_atr(df, period=14).values
    atr_safe = np.where(atr > eps, atr, eps)
    features[:, 9] = (h - l) / atr_safe  # ATR-normalized range

    # --- Volume dynamics ---
    # Relative volume (current / rolling mean)
    vol_sma = pd.Series(v).rolling(20, min_periods=1).mean().values
    vol_sma_safe = np.where(vol_sma > eps, vol_sma, eps)
    features[:, 10] = v / vol_sma_safe

    # Volume-price correlation (rolling 20-bar)
    abs_returns = np.abs(features[:, 8])
    vol_series = pd.Series(v)
    ret_series = pd.Series(abs_returns)
    features[:, 11] = vol_series.rolling(20, min_periods=5).corr(ret_series).fillna(0).values

    # --- Gap ---
    features[:, 12] = (o - prev_close) / atr_safe

    # --- Time cyclicality ---
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour + df.index.minute / 60.0
        dow = df.index.dayofweek  # 0=Mon, 4=Fri
    else:
        hours = np.zeros(n)
        dow = np.zeros(n)

    features[:, 13] = np.sin(2 * np.pi * hours / 24.0)  # hour_sin
    features[:, 14] = np.cos(2 * np.pi * hours / 24.0)  # hour_cos
    features[:, 15] = np.sin(2 * np.pi * dow / 5.0)     # dow_sin
    features[:, 16] = np.cos(2 * np.pi * dow / 5.0)     # dow_cos

    # Replace any NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def get_direction(features: np.ndarray) -> np.ndarray:
    """
    Extract direction labels from feature matrix.
    direction = 1 if close > open, else 0
    Uses the raw OHLC in features[:, 0:4].
    """
    return (features[:, 3] > features[:, 0]).astype(np.float32)
