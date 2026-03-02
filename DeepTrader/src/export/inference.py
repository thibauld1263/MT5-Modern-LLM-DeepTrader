"""
DeepTrader — ONNX Inference Wrapper
Lightweight inference using onnxruntime (no PyTorch needed).
"""

import numpy as np
from typing import Tuple, Optional

class DeepTraderInference:
    """
    ONNX-based inference engine for DeepTrader.
    No PyTorch dependency — just numpy and onnxruntime.
    """

    def __init__(self, onnx_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for inference: pip install onnxruntime")

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print(f"  [OK] ONNX model loaded: {onnx_path}")
        print(f"    Provider: {self.session.get_providers()}")

    def predict(
        self,
        m30_candles: np.ndarray,
        h1_context: np.ndarray,
        h4_context: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference.

        Args:
            m30_candles: (batch, seq_len, 17) — normalized M30 candles
            h1_context: (batch, h1_window, 17) — normalized H1 context
            h4_context: (batch, h4_window, 17) — normalized H4 context
        Returns:
            probs: (batch, 3) — predicted ATR probabilities [P(Long), P(Short), P(Abort)]
        """
        # Ensure float32
        m30 = m30_candles.astype(np.float32)
        h1 = h1_context.astype(np.float32)
        h4 = h4_context.astype(np.float32)

        # Add batch dim if needed
        if m30.ndim == 2:
            m30 = m30[np.newaxis]
            h1 = h1[np.newaxis]
            h4 = h4[np.newaxis]

        result = self.session.run(None, {
            "m30_candles": m30,
            "h1_context": h1,
            "h4_context": h4,
        })

        return result[0]  # probs

    def analyze_probs(
        self,
        probs: np.ndarray,
    ) -> dict:
        """
        Analyze predicted ATR probabilities and generate trading signal.

        Args:
            probs: (batch, 3) direction probabilities
        Returns:
            Signal dict with direction, confidence, and raw probabilities
        """
        p_long, p_short, p_abort = probs[0]  # taking first item in batch

        direction = "NEUTRAL"
        confidence = 0.0

        if p_abort > 0.5:
            direction = "ABORT"
            confidence = float(p_abort)
        elif p_long > p_short:
            direction = "LONG"
            confidence = float(p_long)
        elif p_short > p_long:
            direction = "SHORT"
            confidence = float(p_short)

        signal = {
            "direction": direction,
            "confidence": confidence,
            "p_long": float(p_long),
            "p_short": float(p_short),
            "p_abort": float(p_abort),
        }

        return signal
