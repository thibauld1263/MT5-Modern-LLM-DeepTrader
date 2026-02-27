"""
DeepTrader — Training Utilities
Learning rate scheduling, logging, and common training helpers.
"""

import os
import math
import torch
import torch.nn as nn
from typing import Optional


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  [OK] Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            print(f"  [WARN] No GPU found, using CPU (training will be slow)")
    else:
        device = torch.device(preference)
    return device


class CosineWarmupScheduler:
    """
    Cosine annealing with linear warmup.
    Same schedule used by GPT-2, GPT-3, etc.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = self._compute_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _compute_lr(self) -> float:
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.step_count / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class TrainingLogger:
    """Logging to console (Tensorboard disabled due to compatibility)."""

    def __init__(self, log_dir: str, phase: str = "pretrain"):
        self.log_dir = os.path.join(log_dir, phase)
        os.makedirs(self.log_dir, exist_ok=True)
        self.phase = phase

    def log_step(self, step: int, metrics: dict):
        pass

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        pass

    def close(self):
        pass


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
