"""
DeepTrader — Fine-tune Head
Output head for Phase 2: ATR-based classification.

Predicts 3 independent probabilities: P(Long Win), P(Short Win), P(Early Abort).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinetuneHead(nn.Module):
    """
    Classification head for Phase 2.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Prediction head: hidden state -> 3 logits
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 3), # 3 classes: Long, Short, Abort
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor, predict_len: int = None):
        """
        Args:
            hidden_states: (batch, seq_len, d_model) — Transformer output
        Returns:
            logits: (batch, 3) 
        """
        # Only care about the final hidden state to predict the future
        final_hidden = hidden_states[:, -1, :]  # (batch, d_model)
        logits = self.classifier(final_hidden)  # (batch, 3)
        return logits

    def compute_loss(self, hidden_states: torch.Tensor, target_probs: torch.Tensor):
        """
        Compute Multi-label BCE loss.
        """
        logits = self.forward(hidden_states)
        
        # Binary Cross Entropy for independent probabilities
        loss = F.binary_cross_entropy_with_logits(logits, target_probs)
        
        return loss
