"""
DeepTrader — Preprocessor
Loads raw CSV exports from MT5, cleans, and normalizes data.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional
import yaml


def load_config(config_dir: str = "config") -> dict:
    """Load all YAML configs into a single dict."""
    cfg = {}
    for name in ["data", "model", "training"]:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                cfg[name] = yaml.safe_load(f)
    return cfg


def load_raw_csv(filepath: str) -> Optional[pd.DataFrame]:
    """Load a single raw CSV exported by the MQ5 script."""
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return None

    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Keep core columns
    required = ["open", "high", "low", "close", "tick_volume"]
    for col in required:
        if col not in df.columns:
            print(f"  [WARN] Missing column '{col}' in {filepath}")
            return None

    # Optional: spread
    if "spread" not in df.columns:
        df["spread"] = 0

    return df[required + ["spread"]].copy()


def load_all_raw(raw_dir: str, symbols: list, timeframes: list) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all raw CSVs into a nested dict: {symbol: {tf: DataFrame}}.
    """
    data = {}
    for symbol in symbols:
        data[symbol] = {}
        for tf in timeframes:
            filename = f"{symbol}_{tf}.csv"
            filepath = os.path.join(raw_dir, filename)
            df = load_raw_csv(filepath)
            if df is not None:
                data[symbol][tf] = df
                print(f"  [OK] Loaded {symbol} {tf}: {len(df):,} bars")
            else:
                print(f"  [FAIL] Failed {symbol} {tf}")
    return data


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw OHLCV DataFrame:
    - Remove zero-volume bars (market closed)
    - Remove bars with zero range (frozen price)
    - Forward-fill any NaN gaps
    - Remove duplicates
    """
    df = df.copy()

    # Remove zero-volume (market was closed)
    df = df[df["tick_volume"] > 0]

    # Remove zero-range (frozen/broken data)
    df = df[(df["high"] - df["low"]) > 0]

    # Remove duplicate indices
    df = df[~df.index.duplicated(keep="first")]

    # Forward-fill small gaps (max 3 bars)
    df = df.asfreq(pd.infer_freq(df.index[:100]) if len(df) > 100 else None)
    if df is not None:
        df = df.ffill(limit=3)
        df = df.dropna()

    return df


def normalize_window(window: np.ndarray, price_cols: list = [0, 1, 2, 3]) -> np.ndarray:
    """
    Z-score normalize price columns within a single window.
    Volume and other features are handled separately.

    Args:
        window: shape (seq_len, n_features)
        price_cols: indices of OHLC columns
    Returns:
        Normalized window (same shape)
    """
    out = window.copy()

    # Price normalization: z-score within window
    prices = window[:, price_cols]
    mu = prices.mean()
    sigma = prices.std()
    if sigma < 1e-10:
        sigma = 1e-10
    out[:, price_cols] = (prices - mu) / sigma

    return out


def process_and_save(
    raw_dir: str,
    processed_dir: str,
    symbols: list,
    timeframes: list
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Full preprocessing pipeline:
    1. Load raw CSVs
    2. Clean each DataFrame
    3. Save cleaned DataFrames as parquet (fast I/O)
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Load
    raw_data = load_all_raw(raw_dir, symbols, timeframes)

    # Clean and save
    cleaned = {}
    for symbol in symbols:
        cleaned[symbol] = {}
        for tf in timeframes:
            if tf not in raw_data.get(symbol, {}):
                continue
            df = raw_data[symbol][tf]
            df_clean = clean_dataframe(df)

            # Save
            out_path = os.path.join(processed_dir, f"{symbol}_{tf}.parquet")
            df_clean.to_parquet(out_path)
            cleaned[symbol][tf] = df_clean
            print(f"  [OK] Cleaned {symbol} {tf}: {len(df_clean):,} bars -> {out_path}")

    return cleaned


def load_processed(processed_dir: str, symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """Load a single processed parquet file."""
    path = os.path.join(processed_dir, f"{symbol}_{tf}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
