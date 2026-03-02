# DeepTrader

## What is it?
DeepTrader is a modern Large Language Model (LLM) architecture (Transformer with RMSNorm, RoPE, and SwiGLU) to read raw, multi-timeframe price action (M30, H1, H4) and make trading decisions. 

As any modern LLM, the training happens in 3 phases:
1. Pre-training
2. Fine-tuning
3. DPO alignment

How to use it?

1- Export you datas with the DeepTrader_Exporter

### 1. Training the Model
The repository contains a full Python pipeline to train your own model from MT5 exported data.
```bash
pip install -r requirements.txt

# Phase 1: Pre-train the Transformer to understand market structures
python -m scripts.run_pretrain

# Phase 2: Fine-tune the model to predict the ATR-based probabilities
python -m scripts.run_finetune

# Export the trained model to ONNX format
python -m scripts.run_export
```

### 2. Running in MT5
1. Copy the generated `models/deep_trader.onnx` file into your MetaTrader 5 `Common/Files/` directory.
2. Open `mt5/DeepTrader_EA.mq5` in MetaEditor and compile it.
3. Attach the compiled Expert Advisor to any M30 chart. The EA will load the ONNX model and start trading autonomously.
