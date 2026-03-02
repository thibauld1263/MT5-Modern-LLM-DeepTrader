"""
DeepTrader — Run Fine-Tuning (Phase 2)
Launch script for contextual multi-TF fine-tuning.

Usage:
    python -m scripts.run_finetune
    python -m scripts.run_finetune --pretrained models/pretrain_best.pt
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessor import load_config, load_processed
from src.data.features import encode_candles, N_FEATURES
from src.data.alignment import align_timeframes, build_aligned_dataset
from src.data.dataset import FinetuneDataset
from src.model.deep_trader import DeepTrader
from src.training.finetune import run_finetuning
from src.training.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="DeepTrader Phase 2: Fine-Tuning")
    parser.add_argument("--config", default=None, help="Config directory")
    parser.add_argument("--pretrained", default=None, help="Pre-trained checkpoint")
    args = parser.parse_args()

    config_dir = args.config if args.config else os.path.join(PROJECT_ROOT, "config")
    cfg = load_config(config_dir)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(train_cfg.get("seed", 42))

    symbols = data_cfg["symbols"]
    processed_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["processed_dir"])
    models_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["models_dir"])
    primary_tf = data_cfg.get("primary_tf", "M30")

    pretrained_path = args.pretrained if args.pretrained else os.path.join(models_dir, "pretrain_best.pt")

    arch = model_cfg["architecture"]
    ctx = model_cfg.get("context", {})
    h1_window = ctx.get("h1_window", 100)
    h4_window = ctx.get("h4_window", 50)
    seq_len = arch["max_seq_len"]
    predict_len = model_cfg["finetune"]["predict_candles"]
    freeze_layers = model_cfg["finetune"].get("freeze_layers", 4)

    split = data_cfg["split"]
    train_end = pd.Timestamp(split["train_end"])
    val_end = pd.Timestamp(split["val_end"])

    print("=" * 60)
    print("  DeepTrader -- Phase 2: Contextual Fine-Tuning")
    print(f"  Symbols: {symbols}")
    print(f"  Primary TF: {primary_tf}")
    print(f"  Context: H1 ({h1_window} bars) + H4 ({h4_window} bars)")
    print(f"  Pre-trained model: {pretrained_path}")
    print("=" * 60)

    # Load pre-trained model
    print("\n  Loading pre-trained model...")
    model, checkpoint = DeepTrader.load_checkpoint(
        pretrained_path,
        n_features=arch["n_features"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        predict_len=predict_len,
    )

    # Build multi-TF aligned datasets per symbol
    train_datasets = []
    val_datasets = []

    for symbol in symbols:
        print(f"\n  Processing {symbol}...")

        # Load all 3 TFs
        m30_df = load_processed(processed_dir, symbol, "M30")
        h1_df = load_processed(processed_dir, symbol, "H1")
        h4_df = load_processed(processed_dir, symbol, "H4")

        if m30_df is None or h1_df is None or h4_df is None:
            print(f"  [WARN] Missing TF data for {symbol}, skipping")
            continue

        # Align timeframes
        m30_df, h1_df, h4_df = align_timeframes(m30_df, h1_df, h4_df)

        # Encode features
        m30_feat = encode_candles(m30_df)
        h1_feat = encode_candles(h1_df)
        h4_feat = encode_candles(h4_df)

        # Build aligned dataset
        aligned = build_aligned_dataset(
            m30_feat, h1_feat, h4_feat,
            m30_df.index, h1_df.index, h4_df.index,
            h1_context_window=h1_window,
            h4_context_window=h4_window,
        )

        # Chronological 85/15 split
        total_bars = len(m30_df)
        train_idx = int(total_bars * 0.85)
        val_idx = total_bars

        # Train dataset
        train_ds = FinetuneDataset(
            m30_features=aligned["m30_features"][:train_idx],
            h1_features=aligned["h1_features"],
            h4_features=aligned["h4_features"],
            h1_indices=aligned["h1_indices"][:train_idx],
            h4_indices=aligned["h4_indices"][:train_idx],
            seq_len=seq_len,
            predict_len=predict_len,
        )

        # Val dataset
        val_ds = FinetuneDataset(
            m30_features=aligned["m30_features"][train_idx:val_idx],
            h1_features=aligned["h1_features"],
            h4_features=aligned["h4_features"],
            h1_indices=aligned["h1_indices"][train_idx:val_idx],
            h4_indices=aligned["h4_indices"][train_idx:val_idx],
            seq_len=seq_len,
            predict_len=predict_len,
        )

        if len(train_ds) > 0:
            train_datasets.append(train_ds)
        if len(val_ds) > 0:
            val_datasets.append(val_ds)

        print(f"  {symbol}: Train {len(train_ds):,} | Val {len(val_ds):,} samples")

    if not train_datasets:
        print("\n  [ERROR] No training data available!")
        sys.exit(1)

    # Run fine-tuning
    model = run_finetuning(
        model=model,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        training_config=train_cfg,
        model_config=model_cfg,
        output_dir=models_dir,
        log_dir=os.path.join(PROJECT_ROOT, "runs"),
        freeze_layers=freeze_layers,
    )

    print("\n  [OK] Fine-tuning complete! Next step: python -m scripts.run_export")


if __name__ == "__main__":
    main()
