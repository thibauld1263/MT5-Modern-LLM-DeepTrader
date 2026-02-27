"""
DeepTrader — Run Backtest
Evaluate the trained model on out-of-sample data.

Usage:
    python -m scripts.run_backtest
    python -m scripts.run_backtest --checkpoint models/dpo_best.pt
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessor import load_config, load_processed
from src.data.features import encode_candles
from src.data.alignment import align_timeframes, build_aligned_dataset
from src.data.dataset import FinetuneDataset
from src.model.deep_trader import DeepTrader
from src.evaluation.metrics import evaluate_model, print_evaluation
from src.training.utils import get_device, set_seed

import torch
from torch.utils.data import DataLoader, ConcatDataset


def main():
    parser = argparse.ArgumentParser(description="DeepTrader Backtest")
    parser.add_argument("--config", default=None, help="Config directory")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate")
    args = parser.parse_args()

    config_dir = args.config if args.config else os.path.join(PROJECT_ROOT, "config")
    cfg = load_config(config_dir)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))

    arch = model_cfg["architecture"]
    ctx = model_cfg.get("context", {})
    split = data_cfg["split"]
    val_end = pd.Timestamp(split["val_end"])
    processed_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["processed_dir"])
    models_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["models_dir"])

    # Auto-detect checkpoint
    if args.checkpoint is None:
        for name in ["dpo_best.pt", "finetune_best.pt", "pretrain_best.pt"]:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                args.checkpoint = path
                break

    print("=" * 60)
    print("  DeepTrader -- Backtest (Out-of-Sample)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test date: after {split['val_end']}")
    print("=" * 60)

    # Load model
    model, _ = DeepTrader.load_checkpoint(
        args.checkpoint,
        n_features=arch["n_features"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        predict_len=model_cfg["finetune"]["predict_candles"],
    )
    model = model.to(device)
    model.eval()

    # Build test datasets (after val_end)
    test_datasets = []
    for symbol in data_cfg["symbols"]:
        m30_df = load_processed(processed_dir, symbol, "M30")
        h1_df = load_processed(processed_dir, symbol, "H1")
        h4_df = load_processed(processed_dir, symbol, "H4")

        if m30_df is None or h1_df is None or h4_df is None:
            continue

        m30_df, h1_df, h4_df = align_timeframes(m30_df, h1_df, h4_df)
        m30_feat = encode_candles(m30_df)
        h1_feat = encode_candles(h1_df)
        h4_feat = encode_candles(h4_df)

        aligned = build_aligned_dataset(
            m30_feat, h1_feat, h4_feat,
            m30_df.index, h1_df.index, h4_df.index,
            h1_context_window=ctx.get("h1_window", 100),
            h4_context_window=ctx.get("h4_window", 50),
        )

        val_idx = (m30_df.index <= val_end).sum()

        test_ds = FinetuneDataset(
            m30_features=aligned["m30_features"][val_idx:],
            h1_features=aligned["h1_features"],
            h4_features=aligned["h4_features"],
            h1_indices=aligned["h1_indices"][val_idx:],
            h4_indices=aligned["h4_indices"][val_idx:],
            seq_len=arch["max_seq_len"],
            predict_len=model_cfg["finetune"]["predict_candles"],
        )
        if len(test_ds) > 0:
            test_datasets.append(test_ds)
            print(f"  {symbol}: {len(test_ds):,} test samples")

    if not test_datasets:
        print("  [ERROR] No test data!")
        sys.exit(1)

    test_data = ConcatDataset(test_datasets)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Generate predictions
    print(f"\n  Generating predictions on {len(test_data):,} samples...")
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch in test_loader:
            m30 = batch["m30_input"].to(device)
            h1 = batch["h1_context"].to(device)
            h4 = batch["h4_context"].to(device)
            target = batch["target"]

            h1_mask = (h1.abs().sum(dim=-1) > 0).float()
            h4_mask = (h4.abs().sum(dim=-1) > 0).float()

            traj, _ = model.finetune_forward(
                m30, h1, h4, h1_mask, h4_mask,
                predict_len=model_cfg["finetune"]["predict_candles"]
            )

            all_preds.append(traj.cpu().numpy())
            all_actuals.append(target.numpy())

    predictions = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    print(f"  Generated {len(predictions):,} trajectory predictions")

    # Evaluate
    results = evaluate_model(predictions, actuals)
    print_evaluation(results)


if __name__ == "__main__":
    main()
