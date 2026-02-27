"""
DeepTrader — Context Encoder
Encodes higher-timeframe (H1, H4) candle sequences into context vectors
for cross-attention in Phase 2.

The context encoder uses the SAME pre-trained weights as the main Transformer
(shared architecture), but processes HTF candles independently.
"""

import torch
import torch.nn as nn
from typing import Optional

from .embeddings import CandleEmbedding, TimeframeEmbedding
from .transformer import TransformerBlock, RotaryEmbedding, RMSNorm


class ContextEncoder(nn.Module):
    """
    Encodes a higher-timeframe candle sequence into context vectors.

    Uses a frozen subset of the pre-trained Transformer layers.
    Adds timeframe-specific embedding to distinguish H1 from H4.
    Output is used as keys/values in cross-attention.
    """

    def __init__(
        self,
        n_features: int = 17,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,       # Use fewer layers than main decoder
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embedding = CandleEmbedding(n_features, d_model, max_seq_len, dropout)
        self.tf_embedding = TimeframeEmbedding(n_timeframes=3, d_model=d_model)

        # Self-attention only (no cross-attention in context encoder)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len, use_cross_attention=False)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, tf_id: int,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, ctx_len, n_features) — HTF candle features
            tf_id: 1 for H1, 2 for H4
            padding_mask: (batch, ctx_len) — 1=valid, 0=padding
        Returns:
            (batch, ctx_len, d_model) — encoded context for cross-attention
        """
        # Embed candles + add timeframe embedding
        h = self.embedding(x)
        h = self.tf_embedding(h, tf_id)

        # Compute RoPE
        cos, sin = self.rope(h)

        # Process through encoder layers
        for layer in self.layers:
            h = layer(h, cos, sin)

        return self.norm(h)

    def load_pretrained_layers(self, pretrained_layers: nn.ModuleList):
        """
        Initialize context encoder layers from pre-trained main decoder layers.
        This is the transfer learning step — the context encoder starts from
        the same weights the model learned during Phase 1.
        """
        n_to_copy = min(len(self.layers), len(pretrained_layers))
        for i in range(n_to_copy):
            # Copy self-attention, FFN, and RMSNorm weights (skip cross-attention)
            self.layers[i].norm1.load_state_dict(pretrained_layers[i].norm1.state_dict())
            self.layers[i].self_attn.load_state_dict(pretrained_layers[i].self_attn.state_dict())
            self.layers[i].norm2.load_state_dict(pretrained_layers[i].norm2.state_dict())
            self.layers[i].ff.load_state_dict(pretrained_layers[i].ff.state_dict())
        print(f"  [OK] Loaded {n_to_copy} pre-trained layers into context encoder")
