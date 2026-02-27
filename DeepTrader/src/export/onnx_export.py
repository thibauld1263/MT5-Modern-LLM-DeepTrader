"""
DeepTrader — ONNX Export
Export the trained PyTorch model to ONNX format for deployment.
"""

import os
import torch
import numpy as np
from typing import Tuple

def export_to_onnx(
    model,
    output_path: str = "models/deep_trader.onnx",
    seq_len: int = 256,
    h1_window: int = 100,
    h4_window: int = 50,
    n_features: int = 17,
    predict_len: int = 10,
    opset_version: int = 17,
) -> str:
    """
    Export the full DeepTrader model to ONNX.

    The exported model takes:
    - m30_candles: (1, seq_len, 17)
    - h1_context: (1, h1_window, 17)
    - h4_context: (1, h4_window, 17)

    - target_probs: (1, 3)
    """
    model.eval()
    model.cpu()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Create dummy inputs
    dummy_m30 = torch.randn(1, seq_len, n_features)
    dummy_h1 = torch.randn(1, h1_window, n_features)
    dummy_h4 = torch.randn(1, h4_window, n_features)

    # Wrap in an export-friendly module
    export_model = DeepTraderONNX(model, predict_len)

    # Export
    torch.onnx.export(
        export_model,
        (dummy_m30, dummy_h1, dummy_h4),
        output_path,
        output_names=["target_probs"],
        dynamic_axes={
            "m30_candles": {0: "batch_size"},
            "h1_context": {0: "batch_size"},
            "h4_context": {0: "batch_size"},
            "target_probs": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] ONNX exported: {output_path} ({size_mb:.1f} MB)")

    return output_path

class DeepTraderONNX(torch.nn.Module):
    """Wrapper for ONNX-compatible forward pass."""

    def __init__(self, model, predict_len: int = 10):
        super().__init__()
        self.model = model
        self.predict_len = predict_len

    def forward(self, m30_candles, h1_context, h4_context):
        h1_mask = (h1_context.abs().sum(dim=-1) > 0).float()
        h4_mask = (h4_context.abs().sum(dim=-1) > 0).float()

        probs = self.model.generate(
            m30_candles, h1_context, h4_context,
            h1_mask, h4_mask
        )
        return probs

def verify_onnx(
    onnx_path: str,
    pytorch_model,
    seq_len: int = 256,
    h1_window: int = 100,
    h4_window: int = 50,
    n_features: int = 17,
    predict_len: int = 10,
) -> bool:
    """
    Verify ONNX model output matches PyTorch model output.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] onnxruntime not installed, skipping verification")
        return False

    pytorch_model.eval()
    pytorch_model.cpu()

    # Create test input
    m30 = torch.randn(1, seq_len, n_features)
    h1 = torch.randn(1, h1_window, n_features)
    h4 = torch.randn(1, h4_window, n_features)

    # PyTorch inference
    with torch.no_grad():
        h1_mask = (h1.abs().sum(dim=-1) > 0).float()
        h4_mask = (h4.abs().sum(dim=-1) > 0).float()
        pt_probs = pytorch_model.generate(
            m30, h1, h4, h1_mask, h4_mask
        )

    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    onnx_result = session.run(None, {
        "m30_candles": m30.numpy(),
        "h1_context": h1.numpy(),
        "h4_context": h4.numpy(),
    })
    onnx_probs = onnx_result[0]

    # Compare
    probs_diff = np.abs(pt_probs.numpy() - onnx_probs).max()

    print(f"  ONNX Verification:")
    print(f"    Target Probs max diff: {probs_diff:.2e}")

    ok = probs_diff < 1e-4
    print(f"    Status: {'[OK] PASS' if ok else '[FAIL]'}")
    return ok
