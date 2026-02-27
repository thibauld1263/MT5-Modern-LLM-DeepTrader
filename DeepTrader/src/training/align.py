"""
DeepTrader — Phase 3: DPO Outcome Alignment
Direct Preference Optimization — the RLHF equivalent for trading.
Aligns the model to prefer trajectories that match actual market outcomes.
"""

import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List

from ..model.deep_trader import DeepTrader
from ..data.dataset import AlignDataset, FinetuneDataset
from .utils import (
    get_device, CosineWarmupScheduler, EarlyStopping,
    TrainingLogger, set_seed
)


def generate_preference_pairs(
    model: DeepTrader,
    val_datasets: List[FinetuneDataset],
    device: torch.device,
    n_samples: int = 5000,
    predict_len: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Generate (chosen, rejected) trajectory pairs from the fine-tuned model.

    For each context:
    1. Run the model to generate a predicted trajectory
    2. Compare with the actual trajectory
    3. The actual trajectory is "chosen" (ground truth)
    4. The model's prediction is "rejected" (to be improved upon)

    In more sophisticated setups, you'd sample multiple trajectories
    and rank them. This is the simplified version.
    """
    model.eval()
    all_contexts = []
    all_h1 = []
    all_h4 = []
    all_chosen = []
    all_rejected = []

    print(f"  Generating {n_samples} preference pairs...")

    total_collected = 0
    for dataset in val_datasets:
        if total_collected >= n_samples:
            break

        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for batch in loader:
            if total_collected >= n_samples:
                break

            m30 = batch["m30_input"].to(device)
            h1 = batch["h1_context"].to(device)
            h4 = batch["h4_context"].to(device)
            target = batch["target"]  # Actual future (this is "chosen")

            h1_mask = (h1.abs().sum(dim=-1) > 0).float()
            h4_mask = (h4.abs().sum(dim=-1) > 0).float()

            # Generate model's prediction (this is "rejected")
            with torch.no_grad():
                predicted, _ = model.finetune_forward(
                    m30, h1, h4, h1_mask, h4_mask, predict_len
                )

            batch_size = m30.shape[0]
            remaining = n_samples - total_collected
            take = min(batch_size, remaining)

            all_contexts.append(m30[:take].cpu().numpy())
            all_h1.append(h1[:take].cpu().numpy())
            all_h4.append(h4[:take].cpu().numpy())
            all_chosen.append(target[:take].numpy())
            all_rejected.append(predicted[:take].cpu().numpy())
            total_collected += take

    return {
        "contexts": np.concatenate(all_contexts, axis=0),
        "h1_contexts": np.concatenate(all_h1, axis=0),
        "h4_contexts": np.concatenate(all_h4, axis=0),
        "chosen": np.concatenate(all_chosen, axis=0),
        "rejected": np.concatenate(all_rejected, axis=0),
    }


def run_dpo_alignment(
    model: DeepTrader,
    val_datasets: List[FinetuneDataset],
    training_config: dict,
    output_dir: str = "models",
    log_dir: str = "runs",
    n_preference_pairs: int = 5000,
) -> DeepTrader:
    """
    Full Phase 3 DPO alignment pipeline.

    1. Generate preference pairs from fine-tuned model
    2. Create reference model (frozen copy)
    3. Train with DPO loss
    """
    cfg = training_config["align"]
    set_seed(training_config.get("seed", 42))
    device = get_device(training_config.get("device", "auto"))

    model = model.to(device)
    model.prepare_for_dpo()

    # Create reference model (frozen copy)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"\n{'='*60}")
    print(f"  Phase 3: DPO Outcome Alignment")
    print(f"{'='*60}\n")

    # Generate preference pairs
    pairs = generate_preference_pairs(
        model, val_datasets, device, n_preference_pairs
    )

    # Create DPO dataset
    dpo_dataset = AlignDataset(
        contexts=pairs["contexts"],
        h1_contexts=pairs["h1_contexts"],
        h4_contexts=pairs["h4_contexts"],
        chosen=pairs["chosen"],
        rejected=pairs["rejected"],
    )

    print(f"  DPO samples: {len(dpo_dataset):,}")

    dpo_loader = DataLoader(
        dpo_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    total_steps = len(dpo_loader) * cfg["epochs"]
    warmup_steps = len(dpo_loader) * cfg.get("warmup_epochs", 2)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    logger = TrainingLogger(log_dir, "dpo")
    os.makedirs(output_dir, exist_ok=True)

    best_accuracy = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        pbar = tqdm(dpo_loader, desc=f"  DPO Epoch {epoch}", leave=False)
        for batch in pbar:
            m30 = batch["m30_input"].to(device)
            h1 = batch["h1_context"].to(device)
            h4 = batch["h4_context"].to(device)
            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)

            h1_mask = (h1.abs().sum(dim=-1) > 0).float()
            h4_mask = (h4.abs().sum(dim=-1) > 0).float()

            losses = model.dpo_loss(
                m30, h1, h4, chosen, rejected,
                h1_mask, h4_mask,
                beta=cfg.get("beta", 0.1),
                ref_model=ref_model,
            )

            optimizer.zero_grad()
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += losses["loss"].item()
            total_acc += losses["accuracy"].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['loss'].item():.4f}",
                "acc": f"{losses['accuracy'].item():.2%}",
            })

        avg_loss = total_loss / max(1, n_batches)
        avg_acc = total_acc / max(1, n_batches)
        elapsed = time.time() - t0

        print(f"  Epoch {epoch}: loss={avg_loss:.4f} | "
              f"preference_acc={avg_acc:.2%} | time={elapsed:.1f}s")

        logger.log_epoch(epoch, {"loss": avg_loss, "accuracy": avg_acc}, {})

        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            model.save_checkpoint(
                os.path.join(output_dir, "dpo_best.pt"),
                optimizer=optimizer, epoch=epoch,
                extra={"accuracy": best_accuracy},
            )

    logger.close()
    print(f"\n{'='*60}")
    print(f"  DPO alignment complete! Best accuracy: {best_accuracy:.2%}")
    print(f"{'='*60}\n")

    return model
