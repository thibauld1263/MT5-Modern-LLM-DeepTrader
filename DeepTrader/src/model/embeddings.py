"""
DeepTrader — Embeddings
Input projection: candle vector (17-dim) → d_model embedding
Plus learned positional encoding and timeframe embedding.
"""

import math
import torch
import torch.nn as nn

from .transformer import RMSNorm


class CandleEmbedding(nn.Module):
    """
    Projects a 17-dimensional candle feature vector into d_model space.
    Adds learned positional encoding.

    This is the equivalent of token_embedding + position_embedding in GPT.
    """

    def __init__(self, n_features: int = 17, d_model: int = 256, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Linear projection: 17-dim candle -> d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # RMSNorm + dropout (pre-norm architecture)
        self.layer_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) — raw candle features
        Returns:
            (batch, seq_len, d_model) — embedded candle sequence
        """
        # Project candles into d_model space
        embedded = self.input_projection(x)  # (batch, seq_len, d_model)

        # Apply RMSNorm and dropout (RoPE handles positional embedding later)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)

        return embedded


class TimeframeEmbedding(nn.Module):
    """
    Adds a learned timeframe-specific embedding.
    Allows the model to distinguish M30/H1/H4 context.

    Like a "segment embedding" in BERT, but for timeframes.
    """

    def __init__(self, n_timeframes: int = 3, d_model: int = 256):
        super().__init__()
        # 0 = M30, 1 = H1, 2 = H4
        self.tf_embedding = nn.Embedding(n_timeframes, d_model)
        nn.init.normal_(self.tf_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, tf_id: int) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — already embedded sequence
            tf_id: 0=M30, 1=H1, 2=H4
        Returns:
            (batch, seq_len, d_model) — with timeframe embedding added
        """
        tf_embed = self.tf_embedding(
            torch.tensor(tf_id, device=x.device)
        )  # (d_model,)
        return x + tf_embed.unsqueeze(0).unsqueeze(0)  # broadcast over batch & seq
