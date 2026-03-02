"""
DeepTrader — Run Pre-Training (Phase 1)
Launch script for self-supervised next-candle prediction.

Usage:
    python -m scripts.run_pretrain
    python -m scripts.run_pretrain --config config/
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessor import load_config, load_processed, process_and_save
from src.data.features import encode_candles, N_FEATURES
from src.data.dataset import PretrainDataset
from src.training.pretrain import run_pretraining
from src.training.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="DeepTrader Phase 1: Pre-Training")
    parser.add_argument("--config", default=None, help="Config directory")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    args = parser.parse_args()

    # Resolve config dir relative to project root
    config_dir = args.config if args.config else os.path.join(PROJECT_ROOT, "config")

    # Load configs
    cfg = load_config(config_dir)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(train_cfg.get("seed", 42))

    symbols = data_cfg["symbols"]
    timeframes = list(data_cfg["timeframes"].keys())
    raw_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["raw_dir"])
    processed_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["processed_dir"])

    print("=" * 60)
    print("  DeepTrader -- Phase 1: Pre-Training")
    print(f"  Symbols: {symbols}")
    print(f"  Timeframes: {timeframes}")
    print("=" * 60)

    # Step 1: Preprocess data (if needed)
    if not args.skip_preprocess:
        print("\n  Step 1: Preprocessing raw data...")
        process_and_save(raw_dir, processed_dir, symbols, timeframes)
    else:
        print("\n  Step 1: Skipping preprocessing")

    # Step 2: Encode features and create datasets
    print("\n  Step 2: Encoding features...")
    seq_len = model_cfg["architecture"]["max_seq_len"]
    predict_len = model_cfg["pretrain"]["predict_candles"]

    # We use a flat 85/15 split per asset

    train_datasets = []
    val_datasets = []

    for symbol in symbols:
        for tf in timeframes:
            df = load_processed(processed_dir, symbol, tf)
            if df is None:
                print(f"  [WARN] No data for {symbol} {tf}, skipping")
                continue

            # Encode to 17-feature vectors
            features = encode_candles(df)
            print(f"  {symbol} {tf}: {len(features):,} candles -> {N_FEATURES} features")

            # Chronological 85/15 split
            total_bars = len(df)
            train_idx = int(total_bars * 0.85)
            val_idx = total_bars

            # Create datasets
            train_ds = PretrainDataset(
                features[:train_idx], seq_len=seq_len,
                predict_len=predict_len, stride=1
            )
            val_ds = PretrainDataset(
                features[train_idx:val_idx], seq_len=seq_len,
                predict_len=predict_len, stride=1
            )

            if len(train_ds) > 0:
                train_datasets.append(train_ds)
            if len(val_ds) > 0:
                val_datasets.append(val_ds)

            print(f"    Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

    if not train_datasets:
        print("\n  [ERROR] No training data found! Export data from MT5 first.")
        print("  Place CSV files in data/raw/ (e.g., EURUSD_M30.csv)")
        sys.exit(1)

    # Step 3: Run pre-training
    print(f"\n  Step 3: Starting pre-training...")
    model = run_pretraining(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        model_config=model_cfg,
        training_config=train_cfg,
        output_dir=os.path.join(PROJECT_ROOT, data_cfg["paths"]["models_dir"]),
        log_dir=os.path.join(PROJECT_ROOT, "runs"),
    )

    print("\n  [OK] Pre-training complete! Next step: python -m scripts.run_finetune")

if __name__ == "__main__":
    main()
