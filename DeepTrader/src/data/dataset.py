"""
DeepTrader — PyTorch Datasets
Provides datasets for all 3 training phases:
  1. PretrainDataset: autoregressive next-candle prediction (single-TF)
  2. FinetuneDataset: multi-TF context + trajectory generation
  3. AlignDataset: DPO preference pairs for outcome alignment
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict


class PretrainDataset(Dataset):
    """
    Phase 1: Self-supervised next-candle prediction.

    Given a window of `seq_len` candles, the target is the next `predict_len` candles.
    Like GPT: given tokens [1..t], predict [2..t+1].

    The dataset applies per-window z-score normalization on OHLC features
    so the model sees relative patterns, not absolute price levels.
    """

    def __init__(
        self,
        features: np.ndarray,       # (N, 17) feature matrix from encode_candles
        seq_len: int = 256,          # Context window
        predict_len: int = 1,        # Number of candles to predict
        stride: int = 1,             # Step between windows
    ):
        super().__init__()
        self.features = features.astype(np.float32)
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.stride = stride
        self.price_cols = [0, 1, 2, 3]  # OHLC columns

        # Valid window start positions
        total_need = seq_len + predict_len
        self.n_samples = max(0, (len(features) - total_need) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        target_end = end + self.predict_len

        # Extract window + target
        window = self.features[start:end].copy()
        target = self.features[end:target_end].copy()

        # Per-window z-score normalization on OHLC
        prices = window[:, self.price_cols]
        mu = prices.mean()
        sigma = prices.std()
        if sigma < 1e-10:
            sigma = 1e-10

        window[:, self.price_cols] = (window[:, self.price_cols] - mu) / sigma
        target[:, self.price_cols] = (target[:, self.price_cols] - mu) / sigma

        # Direction label: was the next candle bullish?
        # (close > open in the target candle)
        direction = (target[:, 3] > target[:, 0]).astype(np.float32)

        return {
            "input": torch.from_numpy(window),           # (seq_len, 17)
            "target": torch.from_numpy(target),           # (predict_len, 17)
            "direction": torch.from_numpy(direction),     # (predict_len,)
            "norm_mu": torch.tensor(mu, dtype=torch.float32),
            "norm_sigma": torch.tensor(sigma, dtype=torch.float32),
        }


class FinetuneDataset(Dataset):
    """
    Phase 2: Contextual fine-tuning with multi-TF.

    The model receives:
    - M30 sequence (primary, seq_len bars)
    - H1 context (h1_context_window bars aligned to current M30 time)
    - H4 context (h4_context_window bars aligned to current M30 time)

    Target: next `predict_len` M30 candles (trajectory).
    """

    def __init__(
        self,
        m30_features: np.ndarray,    # (N_m30, 17)
        h1_features: np.ndarray,     # (N_h1, 17)
        h4_features: np.ndarray,     # (N_h4, 17)
        h1_indices: np.ndarray,      # (N_m30, h1_window) from alignment
        h4_indices: np.ndarray,      # (N_m30, h4_window) from alignment
        seq_len: int = 256,
        predict_len: int = 10,
        stride: int = 1,
    ):
        super().__init__()
        self.m30 = m30_features.astype(np.float32)
        self.h1 = h1_features.astype(np.float32)
        self.h4 = h4_features.astype(np.float32)
        self.h1_indices = h1_indices
        self.h4_indices = h4_indices
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.stride = stride
        self.price_cols = [0, 1, 2, 3]

        total_need = seq_len + predict_len
        self.n_samples = max(0, (len(m30_features) - total_need) // stride + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        target_end = end + self.predict_len

        # --- M30 window + target ---
        m30_window = self.m30[start:end].copy()
        m30_target = self.m30[end:target_end].copy()

        # Z-score normalize M30 prices
        prices = m30_window[:, self.price_cols]
        mu = prices.mean()
        sigma = max(prices.std(), 1e-10)

        # --- Calculate ATR-based Outcomes ---
        # Calculate ATR on the last 14 bars of m30_window (raw prices)
        lookback = min(14, len(m30_window) - 1)
        if lookback < 1:
            atr = 1e-4
        else:
            h14 = m30_window[-lookback:, 1]
            l14 = m30_window[-lookback:, 2]
            c14_prev = m30_window[-lookback-1:-1, 3]
            tr = np.maximum(h14 - l14, np.maximum(np.abs(h14 - c14_prev), np.abs(l14 - c14_prev)))
            atr = tr.mean()
            if atr < 1e-5: atr = 1e-5

        entry_price = m30_window[-1, 3]
        target_highs = m30_target[:, 1]
        target_lows = m30_target[:, 2]

        long_win = 0.0
        short_win = 0.0
        abort = 1.0

        for i in range(len(target_highs)):
            h = target_highs[i]
            l = target_lows[i]
            if h >= entry_price + 2.5 * atr:
                long_win = 1.0
                abort = 0.0
                break
            if l <= entry_price - 2.0 * atr:
                break
                
        for i in range(len(target_highs)):
            h = target_highs[i]
            l = target_lows[i]
            if l <= entry_price - 2.5 * atr:
                short_win = 1.0
                abort = 0.0
                break
            if h >= entry_price + 2.0 * atr:
                break
        
        target_probs = np.array([long_win, short_win, abort], dtype=np.float32)

        # --- Context Windows ---
        h1_idx = self.h1_indices[end - 1]
        h4_idx = self.h4_indices[end - 1]

        h1_context = self._gather_context(self.h1, h1_idx, mu, sigma)
        h4_context = self._gather_context(self.h4, h4_idx, mu, sigma)

        # Normalize M30 prices per window
        m30_window[:, self.price_cols] = (m30_window[:, self.price_cols] - mu) / sigma

        return {
            "m30_input": torch.from_numpy(m30_window),    # (seq_len, 17)
            "h1_context": torch.from_numpy(h1_context),   # (h1_window, 17)
            "h4_context": torch.from_numpy(h4_context),   # (h4_window, 17)
            "target_probs": torch.from_numpy(target_probs), # (3,)
            "norm_mu": torch.tensor(mu, dtype=torch.float32),
            "norm_sigma": torch.tensor(sigma, dtype=torch.float32),
        }

    def _gather_context(self, features: np.ndarray, indices: np.ndarray,
                        mu: float, sigma: float) -> np.ndarray:
        """Gather context features using pre-computed indices, with padding."""
        n_features = features.shape[1]
        context = np.zeros((len(indices), n_features), dtype=np.float32)

        valid = indices >= 0
        if valid.any():
            context[valid] = features[indices[valid]].copy()
            # Normalize prices using same M30 window stats
            context[valid, 0] = (context[valid, 0] - mu) / sigma
            context[valid, 1] = (context[valid, 1] - mu) / sigma
            context[valid, 2] = (context[valid, 2] - mu) / sigma
            context[valid, 3] = (context[valid, 3] - mu) / sigma

        return context


class AlignDataset(Dataset):
    """
    Phase 3: DPO alignment dataset.

    Each sample contains:
    - context (M30 + H1 + H4)
    - chosen trajectory (model prediction that was closer to actual)
    - rejected trajectory (model prediction that diverged from actual)

    Built offline after Phase 2 training by running the model on validation data
    and comparing generated trajectories against actual outcomes.
    """

    def __init__(
        self,
        contexts: np.ndarray,       # (N, seq_len, 17) — M30 windows
        h1_contexts: np.ndarray,    # (N, h1_window, 17)
        h4_contexts: np.ndarray,    # (N, h4_window, 17)
        chosen: np.ndarray,         # (N, predict_len, 17) — better trajectories
        rejected: np.ndarray,       # (N, predict_len, 17) — worse trajectories
    ):
        super().__init__()
        self.contexts = torch.from_numpy(contexts.astype(np.float32))
        self.h1_contexts = torch.from_numpy(h1_contexts.astype(np.float32))
        self.h4_contexts = torch.from_numpy(h4_contexts.astype(np.float32))
        self.chosen = torch.from_numpy(chosen.astype(np.float32))
        self.rejected = torch.from_numpy(rejected.astype(np.float32))

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {
            "m30_input": self.contexts[idx],
            "h1_context": self.h1_contexts[idx],
            "h4_context": self.h4_contexts[idx],
            "chosen": self.chosen[idx],
            "rejected": self.rejected[idx],
        }


def create_pretrain_splits(
    features: np.ndarray,
    train_end_idx: int,
    val_end_idx: int,
    seq_len: int = 256,
    predict_len: int = 1,
    stride: int = 1,
) -> tuple:
    """
    Create chronological train/val/test splits for pre-training.
    No data leakage — strict time-based splits.
    """
    train_data = PretrainDataset(features[:train_end_idx], seq_len, predict_len, stride)
    val_data = PretrainDataset(features[train_end_idx:val_end_idx], seq_len, predict_len, stride)
    test_data = PretrainDataset(features[val_end_idx:], seq_len, predict_len, stride)

    return train_data, val_data, test_data
