"""
DeepTrader — Transformer Architecture
"True LLM" architecture utilizing:
  - RMSNorm (replacing LayerNorm)
  - Rotary Position Embeddings (RoPE)
  - SwiGLU FeedForward Networks
  - Causal self-attention
  - Cross-attention for multi-TF context
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (used in LLaMA, Mistral)"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)


def apply_rotary_emb(x, cos, sin):
    # x shape: (B, T, heads, head_dim)
    d = x.shape[-1]
    x_half1, x_half2 = x[..., :d//2], x[..., d//2:]
    x_rot = torch.cat((-x_half2, x_half1), dim=-1)
    
    # cos, sin shape: (T, head_dim) -> (1, T, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return x * cos + x_rot * sin


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if cos is not None and sin is not None:
            # apply_rotary_emb expects (B, T, heads, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            q = apply_rotary_emb(q, cos[:T, :], sin[:T, :])
            k = apply_rotary_emb(k, cos[:T, :], sin[:T, :])
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, T, T)

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for multi-TF context.
    RoPE is not applied here as the temporal alignment between main TF and HTF is disjointed.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.out_proj, self.kv_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        _, S, _ = context.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, S, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, T, S)

        # Mask padding positions in context
        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    LLaMA style: replaces standard 2-layer MLP with Swish-Gated Linear Units.
    """

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        for linear in [self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(xW1) * (xW3) -> W2
        gate = F.silu(self.w1(x)) # SiLU is Swish with beta=1
        x = gate * self.w3(x)
        x = self.dropout(x)
        return self.w2(x)


class TransformerBlock(nn.Module):
    """
    Single True LLM Transformer block with:
    1. Causal self-attention + RoPE
    2. Cross-attention (Phase 2+)
    3. SwiGLU FFN
    4. RMSNorm (Pre-norm)
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024,
                 dropout: float = 0.1, max_seq_len: int = 512, use_cross_attention: bool = False):
        super().__init__()

        # Self-attention
        self.norm1 = RMSNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)

        # Cross-attention (optional, for Phase 2+)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.norm_cross = RMSNorm(d_model)
            self.cross_attn = CrossAttention(d_model, n_heads, dropout)

        # Feed-forward
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None, 
                context: Optional[torch.Tensor] = None, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x), cos, sin)

        # Cross-attention with residual
        if self.use_cross_attention and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context, context_mask)

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x


class TransformerDecoder(nn.Module):
    """
    The full True LLM decoder stack.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        n_cross_attn_layers: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # RoPE applies up to max_seq_len
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

        # Build layers: bottom layers = self-attn only, top layers = self + cross
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            use_cross = (i >= n_layers - n_cross_attn_layers)
            self.layers.append(
                TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len, use_cross)
            )

        self.final_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Precompute RoPE
        cos, sin = self.rope(x)

        for layer in self.layers:
            x = layer(x, cos, sin, context, context_mask)

        return self.final_norm(x)
