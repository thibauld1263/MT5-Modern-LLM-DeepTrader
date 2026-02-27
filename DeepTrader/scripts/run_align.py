"""
DeepTrader — Run DPO Alignment (Phase 3)
Launch script for outcome-aligned training.

Usage:
    python -m scripts.run_align
    python -m scripts.run_align --finetuned models/finetune_best.pt
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessor import load_config, load_processed
from src.data.features import encode_candles
from src.data.alignment import align_timeframes, build_aligned_dataset
from src.data.dataset import FinetuneDataset
from src.model.deep_trader import DeepTrader
from src.training.align import run_dpo_alignment
from src.training.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="DeepTrader Phase 3: DPO Alignment")
    parser.add_argument("--config", default=None, help="Config directory")
    parser.add_argument("--finetuned", default=None, help="Fine-tuned checkpoint")
    parser.add_argument("--n-pairs", type=int, default=5000, help="Number of preference pairs")
    args = parser.parse_args()

    config_dir = args.config if args.config else os.path.join(PROJECT_ROOT, "config")
    cfg = load_config(config_dir)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    set_seed(train_cfg.get("seed", 42))

    arch = model_cfg["architecture"]
    ctx = model_cfg.get("context", {})
    split = data_cfg["split"]
    val_end = pd.Timestamp(split["val_end"])
    train_end = pd.Timestamp(split["train_end"])
    processed_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["processed_dir"])
    models_dir = os.path.join(PROJECT_ROOT, data_cfg["paths"]["models_dir"])

    finetuned_path = args.finetuned if args.finetuned else os.path.join(models_dir, "finetune_best.pt")

    print("=" * 60)
    print("  DeepTrader -- Phase 3: DPO Outcome Alignment")
    print("=" * 60)

    # Load fine-tuned model
    model, _ = DeepTrader.load_checkpoint(
        finetuned_path,
        n_features=arch["n_features"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        predict_len=model_cfg["finetune"]["predict_candles"],
    )

    # Build validation datasets (used for generating preference pairs)
    val_datasets = []
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

        train_idx = (m30_df.index <= train_end).sum()
        val_idx = (m30_df.index <= val_end).sum()

        val_ds = FinetuneDataset(
            m30_features=aligned["m30_features"][train_idx:val_idx],
            h1_features=aligned["h1_features"],
            h4_features=aligned["h4_features"],
            h1_indices=aligned["h1_indices"][train_idx:val_idx],
            h4_indices=aligned["h4_indices"][train_idx:val_idx],
            seq_len=arch["max_seq_len"],
            predict_len=model_cfg["finetune"]["predict_candles"],
        )
        if len(val_ds) > 0:
            val_datasets.append(val_ds)

    # Run DPO
    model = run_dpo_alignment(
        model=model,
        val_datasets=val_datasets,
        training_config=train_cfg,
        output_dir=models_dir,
        log_dir=os.path.join(PROJECT_ROOT, "runs"),
        n_preference_pairs=args.n_pairs,
    )

    print("\n  [OK] DPO alignment complete! Next step: python -m scripts.run_export")


if __name__ == "__main__":
    main()
