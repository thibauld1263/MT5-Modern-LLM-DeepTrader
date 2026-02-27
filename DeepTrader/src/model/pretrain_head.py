"""
DeepTrader — Pre-training Head
Output head for Phase 1: next-candle prediction.

Predicts all 17 features of the next candle + binary direction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainHead(nn.Module):
    """
    Autoregressive next-candle prediction head.

    Takes the Transformer's output at each position and predicts
    the next candle's 17 features + direction (up/down).

    Loss: MSE on features + BCE on direction.
    """

    def __init__(self, d_model: int = 256, n_features: int = 17):
        super().__init__()
        self.n_features = n_features

        # Feature regression head
        self.feature_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_features),
        )

        # Direction classification head (binary: bullish or bearish)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (batch, seq_len, d_model) — Transformer output
        Returns:
            features_pred: (batch, seq_len, 17) — predicted candle features
            direction_logits: (batch, seq_len, 1) — direction logit
        """
        features_pred = self.feature_head(hidden_states)
        direction_logits = self.direction_head(hidden_states)
        return features_pred, direction_logits

    def compute_loss(self, hidden_states: torch.Tensor,
                     target_features: torch.Tensor,
                     target_direction: torch.Tensor,
                     feature_weight: float = 1.0,
                     direction_weight: float = 0.3):
        """
        Compute combined loss for pre-training.

        Args:
            hidden_states: (batch, seq_len, d_model) — use LAST position for next prediction
            target_features: (batch, predict_len, 17)
            target_direction: (batch, predict_len)
            feature_weight: weight for MSE loss
            direction_weight: weight for BCE loss
        Returns:
            total_loss, feature_loss, direction_loss
        """
        # Use the last hidden state for prediction
        last_hidden = hidden_states[:, -1:, :]  # (batch, 1, d_model)

        features_pred, direction_logits = self.forward(last_hidden)

        # Feature regression loss (MSE)
        feature_loss = F.mse_loss(features_pred.squeeze(1), target_features.squeeze(1))

        # Direction loss (BCE)
        direction_loss = F.binary_cross_entropy_with_logits(
            direction_logits.squeeze(-1).squeeze(1),
            target_direction.squeeze(1) if target_direction.dim() > 1 else target_direction
        )

        total_loss = feature_weight * feature_loss + direction_weight * direction_loss

        return total_loss, feature_loss, direction_loss
