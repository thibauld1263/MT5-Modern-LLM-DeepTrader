"""Quick validation script for DeepTrader imports and model instantiation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

# Data pipeline
from src.data.features import N_FEATURES, FEATURE_NAMES
print(f"  [OK] Features: {N_FEATURES} features")

from src.data.preprocessor import load_config
cfg = load_config("config")
print(f"  [OK] Config loaded: {cfg['data']['symbols']}")

from src.data.dataset import PretrainDataset, FinetuneDataset, AlignDataset
print(f"  [OK] Datasets: PretrainDataset, FinetuneDataset, AlignDataset")

from src.data.alignment import align_timeframes, build_aligned_dataset
print(f"  [OK] Alignment module")

# Model
import torch
from src.model.deep_trader import DeepTrader

arch = cfg["model"]["architecture"]
model = DeepTrader(
    n_features=arch["n_features"],
    d_model=arch["d_model"],
    n_heads=arch["n_heads"],
    n_layers=arch["n_layers"],
    d_ff=arch["d_ff"],
    dropout=arch["dropout"],
    max_seq_len=arch["max_seq_len"],
)

# Test forward pass
dummy = torch.randn(2, 128, 17)
feat_pred, dir_pred = model.pretrain_forward(dummy)
print(f"  [OK] Pretrain forward: input={dummy.shape} -> features={feat_pred.shape}, direction={dir_pred.shape}")

# Test fine-tune forward
dummy_h1 = torch.randn(2, 100, 17)
dummy_h4 = torch.randn(2, 50, 17)
traj, dirs = model.finetune_forward(dummy, dummy_h1, dummy_h4, predict_len=10)
print(f"  [OK] Finetune forward: trajectory={traj.shape}, direction={dirs.shape}")

# Test generation
with torch.no_grad():
    gen_traj, gen_dirs = model.generate(dummy, dummy_h1, dummy_h4, predict_len=5)
print(f"  [OK] Generate: trajectory={gen_traj.shape}, direction_probs={gen_dirs.shape}")

# Training utils
from src.training.utils import CosineWarmupScheduler, EarlyStopping
print(f"  [OK] Training utilities")

# Evaluation
from src.evaluation.metrics import evaluate_model
print(f"  [OK] Evaluation metrics")

# Export
from src.export.onnx_export import export_to_onnx
from src.export.inference import DeepTraderInference
print(f"  [OK] Export modules")

total = sum(p.numel() for p in model.parameters())
print(f"\n{'='*50}")
print(f"  ALL TESTS PASSED")
print(f"  Model: {total:,} parameters")
print(f"{'='*50}")
