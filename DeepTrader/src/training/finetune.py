"""
DeepTrader — Phase 2: Contextual Fine-Tuning
Multi-TF context + trajectory generation.
This is the "instruction tuning" equivalent for markets.
"""

import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from typing import List, Dict

from ..model.deep_trader import DeepTrader
from ..data.dataset import FinetuneDataset
from .utils import (
    get_device, CosineWarmupScheduler, EarlyStopping,
    TrainingLogger, set_seed
)

def train_one_epoch_ft(
    model: DeepTrader,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    device: torch.device,
    grad_clip: float = 1.0,
    log_every: int = 100,
) -> Dict[str, float]:
    """Train one epoch of fine-tuning."""
    model.train()
    totals = {"loss": 0, "trajectory_loss": 0, "direction_loss": 0}
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Fine-tuning", leave=False)
    for batch_idx, batch in enumerate(pbar):
        m30 = batch["m30_input"].to(device)
        h1 = batch["h1_context"].to(device)
        h4 = batch["h4_context"].to(device)
        target_probs = batch["target_probs"].to(device)

        # Context masks (valid = non-zero vectors)
        h1_mask = (h1.abs().sum(dim=-1) > 0).float()
        h4_mask = (h4.abs().sum(dim=-1) > 0).float()

        # Forward + loss
        losses = model.finetune_loss(
            m30, h1, h4, target_probs, h1_mask, h4_mask
        )
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr = scheduler.step()

        for k, v in losses.items():
            totals[k] += v.item()
        n_batches += 1

        if batch_idx % log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr:.2e}",
            })

    return {"loss": totals["loss"] / max(1, n_batches)}

@torch.no_grad()
def validate_ft(
    model: DeepTrader,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate fine-tuning."""
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="  Validating", leave=False):
        m30 = batch["m30_input"].to(device)
        h1 = batch["h1_context"].to(device)
        h4 = batch["h4_context"].to(device)
        target_probs = batch["target_probs"].to(device)

        h1_mask = (h1.abs().sum(dim=-1) > 0).float()
        h4_mask = (h4.abs().sum(dim=-1) > 0).float()

        losses = model.finetune_loss(m30, h1, h4, target_probs, h1_mask, h4_mask)
        total_loss += losses["loss"].item()
        n_batches += 1

    return {"loss": total_loss / max(1, n_batches)}

def run_finetuning(
    model: DeepTrader,
    train_datasets: List[FinetuneDataset],
    val_datasets: List[FinetuneDataset],
    training_config: dict,
    model_config: dict,
    output_dir: str = "models",
    log_dir: str = "runs",
    freeze_layers: int = 4,
) -> DeepTrader:
    """
    Full Phase 2 fine-tuning pipeline.
    Takes a pre-trained model and fine-tunes it with multi-TF context.
    """
    cfg = training_config["finetune"]
    set_seed(training_config.get("seed", 42))
    device = get_device(training_config.get("device", "auto"))

    model = model.to(device)

    # Prepare for fine-tuning (freeze layers, init context encoder)
    model.prepare_for_finetune(freeze_bottom_layers=freeze_layers)

    # Combine datasets
    train_data = ConcatDataset(train_datasets)
    val_data = ConcatDataset(val_datasets)

    print(f"\n{'='*60}")
    print(f"  Phase 2: Contextual Fine-Tuning")
    print(f"  Train samples: {len(train_data):,}")
    print(f"  Val samples:   {len(val_data):,}")
    print(f"  Frozen layers: {freeze_layers}")
    print(f"{'='*60}\n")

    train_loader = DataLoader(
        train_data, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True,
    )

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = len(train_loader) * cfg.get("warmup_epochs", 3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    early_stopping = EarlyStopping(patience=cfg.get("early_stopping_patience", 10))
    logger = TrainingLogger(log_dir, "finetune")
    os.makedirs(output_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"\n  Epoch {epoch}/{cfg['epochs']}")
        print(f"  {'-'*50}")

        train_metrics = train_one_epoch_ft(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=cfg.get("grad_clip", 1.0),
            log_every=training_config.get("log_every", 100),
        )

        val_metrics = validate_ft(model, val_loader, device)
        logger.log_epoch(epoch, train_metrics, val_metrics)

        elapsed = time.time() - t0
        print(f"  Train loss: {train_metrics['loss']:.4f} | "
              f"Val loss: {val_metrics['loss']:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save_checkpoint(
                os.path.join(output_dir, "finetune_best.pt"),
                optimizer=optimizer, epoch=epoch,
                extra={"val_loss": best_val_loss},
            )

        if epoch % training_config.get("checkpoint_every", 5) == 0:
            model.save_checkpoint(
                os.path.join(output_dir, f"finetune_epoch_{epoch}.pt"),
                optimizer=optimizer, epoch=epoch,
            )

        if early_stopping(val_metrics["loss"]):
            print(f"\n  [STOP] Early stopping at epoch {epoch}")
            break

    logger.close()
    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete! Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    return model
