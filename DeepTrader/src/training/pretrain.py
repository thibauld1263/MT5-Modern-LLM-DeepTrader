"""
DeepTrader — Phase 1: Pre-Training
Autoregressive next-candle prediction across all symbols and timeframes.
This is the "learn the language of markets" phase.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from typing import List, Dict, Optional

from ..model.deep_trader import DeepTrader
from ..data.dataset import PretrainDataset
from .utils import (
    get_device, CosineWarmupScheduler, EarlyStopping,
    TrainingLogger, set_seed
)

def train_one_epoch(
    model: DeepTrader,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    device: torch.device,
    grad_clip: float = 1.0,
    log_every: int = 100,
) -> Dict[str, float]:
    """Train for one epoch, return average metrics."""
    model.train()
    total_loss = 0.0
    total_feat = 0.0
    total_dir = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        input_seq = batch["input"].to(device)
        target = batch["target"].to(device)
        direction = batch["direction"].to(device)

        # Forward + loss
        losses = model.pretrain_loss(input_seq, target, direction)
        loss = losses["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        lr = scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_feat += losses["feature_loss"].item()
        total_dir += losses["direction_loss"].item()
        n_batches += 1

        if batch_idx % log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "feat": f"{losses['feature_loss'].item():.4f}",
                "dir": f"{losses['direction_loss'].item():.4f}",
                "lr": f"{lr:.2e}",
            })

    return {
        "loss": total_loss / max(1, n_batches),
        "feature_loss": total_feat / max(1, n_batches),
        "direction_loss": total_dir / max(1, n_batches),
    }

@torch.no_grad()
def validate(
    model: DeepTrader,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate and return average metrics."""
    model.eval()
    total_loss = 0.0
    total_feat = 0.0
    total_dir = 0.0
    n_correct_dir = 0
    n_total_dir = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="  Validating", leave=False):
        input_seq = batch["input"].to(device)
        target = batch["target"].to(device)
        direction = batch["direction"].to(device)

        losses = model.pretrain_loss(input_seq, target, direction)

        total_loss += losses["loss"].item()
        total_feat += losses["feature_loss"].item()
        total_dir += losses["direction_loss"].item()
        n_batches += 1

        # Direction accuracy
        _, dir_logits = model.pretrain_forward(input_seq)
        dir_pred = (torch.sigmoid(dir_logits[:, -1, 0]) > 0.5).float()
        target_dir = direction.squeeze(1) if direction.dim() > 1 else direction
        n_correct_dir += (dir_pred == target_dir).sum().item()
        n_total_dir += target_dir.shape[0]

    return {
        "loss": total_loss / max(1, n_batches),
        "feature_loss": total_feat / max(1, n_batches),
        "direction_loss": total_dir / max(1, n_batches),
        "direction_accuracy": n_correct_dir / max(1, n_total_dir),
    }

def run_pretraining(
    train_datasets: List[PretrainDataset],
    val_datasets: List[PretrainDataset],
    model_config: dict,
    training_config: dict,
    output_dir: str = "models",
    log_dir: str = "runs",
) -> DeepTrader:
    """
    Full Phase 1 pre-training pipeline.

    Args:
        train_datasets: list of PretrainDataset (one per symbol-tf combo)
        val_datasets: list of PretrainDataset for validation
        model_config: architecture hyperparameters
        training_config: training hyperparameters
        output_dir: where to save checkpoints
        log_dir: TensorBoard log directory
    Returns:
        Trained DeepTrader model
    """
    cfg = training_config["pretrain"]
    set_seed(training_config.get("seed", 42))

    # Device
    device = get_device(training_config.get("device", "auto"))

    # Combine all datasets
    train_data = ConcatDataset(train_datasets)
    val_data = ConcatDataset(val_datasets)

    print(f"\n{'='*60}")
    print(f"  Phase 1: Pre-Training")
    print(f"  Train samples: {len(train_data):,}")
    print(f"  Val samples:   {len(val_data):,}")
    print(f"{'='*60}\n")

    # DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    arch = model_config["architecture"]
    model = DeepTrader(
        n_features=arch["n_features"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        predict_len=model_config["pretrain"]["predict_candles"],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Scheduler
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = len(train_loader) * cfg.get("warmup_epochs", 5)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    # Early stopping
    early_stopping = EarlyStopping(patience=cfg.get("early_stopping_patience", 15))

    # Logger
    logger = TrainingLogger(log_dir, "pretrain")

    # Output dir
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"\n  Epoch {epoch}/{cfg['epochs']}")
        print(f"  {'-'*50}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=cfg.get("grad_clip", 1.0),
            log_every=training_config.get("log_every", 100),
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Log
        logger.log_epoch(epoch, train_metrics, val_metrics)

        elapsed = time.time() - t0
        print(f"  Train loss: {train_metrics['loss']:.4f} | "
              f"Val loss: {val_metrics['loss']:.4f} | "
              f"Dir acc: {val_metrics['direction_accuracy']:.2%} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save_checkpoint(
                os.path.join(output_dir, "pretrain_best.pt"),
                optimizer=optimizer,
                epoch=epoch,
                extra={"val_loss": best_val_loss, "val_metrics": val_metrics},
            )

        # Periodic checkpoint
        if epoch % training_config.get("checkpoint_every", 5) == 0:
            model.save_checkpoint(
                os.path.join(output_dir, f"pretrain_epoch_{epoch}.pt"),
                optimizer=optimizer,
                epoch=epoch,
            )

        # Early stopping
        if early_stopping(val_metrics["loss"]):
            print(f"\n  [STOP] Early stopping at epoch {epoch} (patience={early_stopping.patience})")
            break

    logger.close()
    print(f"\n{'='*60}")
    print(f"  Pre-training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir}/pretrain_best.pt")
    print(f"{'='*60}\n")

    # Load best model
    model, _ = DeepTrader.load_checkpoint(
        os.path.join(output_dir, "pretrain_best.pt"),
        n_features=arch["n_features"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
    )

    return model.to(device)
