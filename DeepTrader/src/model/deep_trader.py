"""
DeepTrader — Full Model Assembly
Combines all components into a single model that supports all 3 training phases.

Phase 1: Pre-training (autoregressive next-candle prediction)
Phase 2: Fine-tuning (multi-TF context + trajectory generation)
Phase 3: DPO alignment (preference optimization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .embeddings import CandleEmbedding, TimeframeEmbedding
from .transformer import TransformerDecoder
from .pretrain_head import PretrainHead
from .finetune_head import FinetuneHead
from .context_encoder import ContextEncoder


class DeepTrader(nn.Module):
    """
    The complete DeepTrader model.

    Architecture:
    - CandleEmbedding: 17-dim → d_model
    - TimeframeEmbedding: adds TF-specific signal
    - TransformerDecoder: causal self-attn + cross-attn
    - PretrainHead: next-candle prediction (Phase 1)
    - ContextEncoder: encodes H1/H4 for cross-attention (Phase 2+)
    - FinetuneHead: K-candle trajectory generation (Phase 2+)
    """

    def __init__(
        self,
        n_features: int = 17,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        predict_len: int = 10,
        n_cross_attn_layers: int = 4,
        context_encoder_layers: int = 4,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.predict_len = predict_len

        # --- Shared components ---
        self.embedding = CandleEmbedding(n_features, d_model, max_seq_len, dropout)
        self.tf_embedding = TimeframeEmbedding(n_timeframes=3, d_model=d_model)
        self.decoder = TransformerDecoder(
            d_model, n_heads, n_layers, d_ff, dropout, max_seq_len, n_cross_attn_layers
        )

        # --- Phase 1 head ---
        self.pretrain_head = PretrainHead(d_model, n_features)

        # --- Phase 2+ components ---
        self.context_encoder = ContextEncoder(
            n_features, d_model, n_heads, context_encoder_layers, d_ff, dropout, max_seq_len
        )
        self.finetune_head = FinetuneHead(d_model)

        # Track parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  DeepTrader initialized:")
        print(f"    Total parameters:     {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        print(f"    d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Pre-training (next-candle prediction, single TF)
    # ══════════════════════════════════════════════════════════════════

    def pretrain_forward(self, candles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 1 forward pass.

        Args:
            candles: (batch, seq_len, 17) — input candle sequence
        Returns:
            features_pred: (batch, seq_len, 17) — predicted next-candle features
            direction_logits: (batch, seq_len, 1) — direction logits
        """
        # Embed + add M30 timeframe embedding
        h = self.embedding(candles)
        h = self.tf_embedding(h, tf_id=0)  # 0 = M30

        # Run through decoder (no cross-attention context)
        h = self.decoder(h, context=None)

        # Predict next candle
        return self.pretrain_head(h)

    def pretrain_loss(self, candles: torch.Tensor, target_features: torch.Tensor,
                      target_direction: torch.Tensor,
                      feature_weight: float = 1.0,
                      direction_weight: float = 0.3) -> Dict[str, torch.Tensor]:
        """Compute pre-training loss."""
        h = self.embedding(candles)
        h = self.tf_embedding(h, tf_id=0)
        h = self.decoder(h, context=None)

        total, feat_loss, dir_loss = self.pretrain_head.compute_loss(
            h, target_features, target_direction, feature_weight, direction_weight
        )
        return {"loss": total, "feature_loss": feat_loss, "direction_loss": dir_loss}

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Fine-tuning (multi-TF context + trajectory generation)
    # ══════════════════════════════════════════════════════════════════

    def finetune_forward(
        self,
        m30_candles: torch.Tensor,
        h1_context: torch.Tensor,
        h4_context: torch.Tensor,
        h1_mask: Optional[torch.Tensor] = None,
        h4_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Phase 2 forward pass.

        Args:
            m30_candles: (batch, seq_len, 17) — main M30 sequence
            h1_context: (batch, h1_window, 17) — H1 context candles
            h4_context: (batch, h4_window, 17) — H4 context candles
            h1_mask: (batch, h1_window) — 1=valid, 0=padding
            h4_mask: (batch, h4_window) — 1=valid, 0=padding
        Returns:
            logits: (batch, 3) — ATR probability logits (Long, Short, Abort)
        """
        # Encode higher-TF contexts
        h1_encoded = self.context_encoder(h1_context, tf_id=1, padding_mask=h1_mask)
        h4_encoded = self.context_encoder(h4_context, tf_id=2, padding_mask=h4_mask)

        # Concatenate H1 + H4 context for cross-attention
        htf_context = torch.cat([h1_encoded, h4_encoded], dim=1)  # (batch, h1+h4, d_model)

        # Build context mask
        if h1_mask is not None and h4_mask is not None:
            htf_mask = torch.cat([h1_mask, h4_mask], dim=1)
        else:
            htf_mask = None

        # Embed M30 + run through decoder with cross-attention
        h = self.embedding(m30_candles)
        h = self.tf_embedding(h, tf_id=0)
        h = self.decoder(h, context=htf_context, context_mask=htf_mask)

        # Generate ATR logits
        return self.finetune_head(h)

    def finetune_loss(
        self,
        m30_candles: torch.Tensor,
        h1_context: torch.Tensor,
        h4_context: torch.Tensor,
        target_probs: torch.Tensor,
        h1_mask: Optional[torch.Tensor] = None,
        h4_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute fine-tuning ATR classification loss."""
        h1_encoded = self.context_encoder(h1_context, tf_id=1, padding_mask=h1_mask)
        h4_encoded = self.context_encoder(h4_context, tf_id=2, padding_mask=h4_mask)
        htf_context = torch.cat([h1_encoded, h4_encoded], dim=1)
        htf_mask = torch.cat([h1_mask, h4_mask], dim=1) if h1_mask is not None else None

        h = self.embedding(m30_candles)
        h = self.tf_embedding(h, tf_id=0)
        h = self.decoder(h, context=htf_context, context_mask=htf_mask)

        loss = self.finetune_head.compute_loss(h, target_probs)
        return {"loss": loss}

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: DPO (Direct Preference Optimization)
    # ══════════════════════════════════════════════════════════════════

    def dpo_loss(
        self,
        m30_candles: torch.Tensor,
        h1_context: torch.Tensor,
        h4_context: torch.Tensor,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
        h1_mask: Optional[torch.Tensor] = None,
        h4_mask: Optional[torch.Tensor] = None,
        beta: float = 0.1,
        ref_model: Optional['DeepTrader'] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        DPO loss for outcome alignment.

        Computes preference loss between chosen (better) and rejected (worse)
        trajectories relative to a frozen reference model.

        Args:
            chosen: (batch, K, 17) — trajectory closer to actual outcome
            rejected: (batch, K, 17) — trajectory diverging from actual
            ref_model: frozen copy of the model before DPO training
            beta: temperature parameter
        """
        # Get log-probabilities for chosen and rejected under current model
        chosen_logprob = self._trajectory_log_prob(
            m30_candles, h1_context, h4_context, chosen, h1_mask, h4_mask
        )
        rejected_logprob = self._trajectory_log_prob(
            m30_candles, h1_context, h4_context, rejected, h1_mask, h4_mask
        )

        # Get log-probabilities under reference model (frozen)
        if ref_model is not None:
            with torch.no_grad():
                ref_chosen = ref_model._trajectory_log_prob(
                    m30_candles, h1_context, h4_context, chosen, h1_mask, h4_mask
                )
                ref_rejected = ref_model._trajectory_log_prob(
                    m30_candles, h1_context, h4_context, rejected, h1_mask, h4_mask
                )
        else:
            ref_chosen = torch.zeros_like(chosen_logprob)
            ref_rejected = torch.zeros_like(rejected_logprob)

        # DPO loss
        logits = beta * (
            (chosen_logprob - ref_chosen) - (rejected_logprob - ref_rejected)
        )
        loss = -F.logsigmoid(logits).mean()

        # Accuracy: how often does the model prefer chosen over rejected?
        accuracy = (chosen_logprob > rejected_logprob).float().mean()

        return {"loss": loss, "accuracy": accuracy}

    def _trajectory_log_prob(
        self,
        m30_candles: torch.Tensor,
        h1_context: torch.Tensor,
        h4_context: torch.Tensor,
        trajectory: torch.Tensor,
        h1_mask: Optional[torch.Tensor],
        h4_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute log-probability of a trajectory under the model.

        Uses MSE-based likelihood: log P(trajectory) ∝ -MSE(predicted, trajectory)
        """
        # Forward pass
        h1_encoded = self.context_encoder(h1_context, tf_id=1, padding_mask=h1_mask)
        h4_encoded = self.context_encoder(h4_context, tf_id=2, padding_mask=h4_mask)
        htf_context = torch.cat([h1_encoded, h4_encoded], dim=1)
        htf_mask = torch.cat([h1_mask, h4_mask], dim=1) if h1_mask is not None else None

        h = self.embedding(m30_candles)
        h = self.tf_embedding(h, tf_id=0)
        h = self.decoder(h, context=htf_context, context_mask=htf_mask)

        # Generate trajectory
        K = trajectory.shape[1]
        pred_trajectory, _ = self.finetune_head(h, predict_len=K)

        # Log-probability ∝ -MSE (Gaussian likelihood)
        mse = F.mse_loss(pred_trajectory, trajectory, reduction="none")
        log_prob = -mse.mean(dim=(1, 2))  # (batch,)

        return log_prob

    # ══════════════════════════════════════════════════════════════════
    # Inference / Generation
    # ══════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        m30_candles: torch.Tensor,
        h1_context: Optional[torch.Tensor] = None,
        h4_context: Optional[torch.Tensor] = None,
        h1_mask: Optional[torch.Tensor] = None,
        h4_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate ATR outcome probabilities (Phase 2).

        Returns:
            probs: (batch, 3) — Sigmoid probabilities for [Long Win, Short Win, Early Abort]
        """
        self.eval()

        if h1_context is not None and h4_context is not None:
            # Phase 2+ mode (ATR Classification)
            logits = self.finetune_forward(
                m30_candles, h1_context, h4_context, h1_mask, h4_mask
            )
            return torch.sigmoid(logits)
        else:
            # Phase 1 mode (no context, single candle prediction)
            h = self.embedding(m30_candles)
            h = self.tf_embedding(h, tf_id=0)
            h = self.decoder(h)
            features_pred, dir_logits = self.pretrain_head(h)
            trajectory = features_pred[:, -1:, :]  # Last position's prediction
            return trajectory

    # ══════════════════════════════════════════════════════════════════
    # Phase transition utilities
    # ══════════════════════════════════════════════════════════════════

    def prepare_for_finetune(self, freeze_bottom_layers: int = 4):
        """
        Prepare the model for Phase 2 fine-tuning:
        1. Transfer pre-trained weights to context encoder
        2. Freeze bottom N decoder layers
        3. Initialize cross-attention layers
        """
        # Transfer pre-trained layers to context encoder
        self.context_encoder.load_pretrained_layers(self.decoder.layers)

        # Freeze bottom layers of the main decoder
        for i in range(min(freeze_bottom_layers, len(self.decoder.layers))):
            for param in self.decoder.layers[i].parameters():
                param.requires_grad = False

        # Freeze embedding (already well-trained)
        for param in self.embedding.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [OK] Prepared for fine-tuning:")
        print(f"    Frozen bottom {freeze_bottom_layers} decoder layers + embeddings")
        print(f"    Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def prepare_for_dpo(self):
        """
        Prepare for Phase 3 DPO:
        Unfreeze all layers but with very small learning rate.
        """
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [OK] Prepared for DPO: all {trainable:,} parameters trainable")

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = 0, extra: dict = None):
        """Save model checkpoint."""
        state = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)
        print(f"  [OK] Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(cls, path: str, **model_kwargs) -> Tuple['DeepTrader', dict]:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**model_kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  [OK] Loaded checkpoint: {path} (epoch {checkpoint.get('epoch', '?')})")
        return model, checkpoint
