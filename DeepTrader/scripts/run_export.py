"""
DeepTrader — Export to ONNX
Exports the trained model to ONNX format for MT5 deployment.

Usage:
    python -m scripts.run_export
    python -m scripts.run_export --checkpoint models/dpo_best.pt
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessor import load_config
from src.model.deep_trader import DeepTrader
from src.export.onnx_export import export_to_onnx, verify_onnx


def main():
    parser = argparse.ArgumentParser(description="DeepTrader ONNX Export")
    parser.add_argument("--config", default=None, help="Config directory")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to export (auto-detect best)")
    parser.add_argument("--output", default=None, help="ONNX output path")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify ONNX output")
    args = parser.parse_args()

    config_dir = args.config if args.config else os.path.join(PROJECT_ROOT, "config")
    cfg = load_config(config_dir)
    model_cfg = cfg["model"]
    arch = model_cfg["architecture"]
    ctx = model_cfg.get("context", {})
    models_dir = os.path.join(PROJECT_ROOT, cfg["data"]["paths"]["models_dir"])

    output_path = args.output if args.output else os.path.join(models_dir, "deep_trader.onnx")

    # Auto-detect best checkpoint
    if args.checkpoint is None:
        for name in ["align_best.pt", "finetune_best.pt", "pretrain_best.pt"]:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                args.checkpoint = path
                break
        if args.checkpoint is None:
            print("  [ERROR] No checkpoint found! Train a model first.")
            sys.exit(1)

    print("=" * 60)
    print("  DeepTrader -- ONNX Export")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {output_path}")
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

    # Export
    export_to_onnx(
        model,
        output_path=output_path,
        seq_len=arch["max_seq_len"],
        h1_window=ctx.get("h1_window", 100),
        h4_window=ctx.get("h4_window", 50),
        n_features=arch["n_features"],
        predict_len=model_cfg["finetune"]["predict_candles"],
    )

    # Verify
    if args.verify:
        verify_onnx(
            output_path, model,
            seq_len=arch["max_seq_len"],
            h1_window=ctx.get("h1_window", 100),
            h4_window=ctx.get("h4_window", 50),
            n_features=arch["n_features"],
            predict_len=model_cfg["finetune"]["predict_candles"],
        )

    print("\n  [OK] Export complete!")
    print(f"  Copy {output_path} to MT5's MQL5/Files/ folder")
    print(f"  Then compile and attach DeepTrader_EA.mq5")


if __name__ == "__main__":
    main()
