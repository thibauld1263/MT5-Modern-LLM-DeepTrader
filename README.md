**DeepTrader**

What is it?


DeepTrader is a modern Large Language Model (LLM) architecture (Transformer with RMSNorm, RoPE, and SwiGLU, similar structure as Llama 3, recent Mistral updates, etc) to read raw, multi-timeframe price action (M30, H1, H4) and make trading decisions. 

As any modern LLM, the training happens in 3 phases:
1. Pre-training
2. Fine-tuning
3. DPO alignment

Instead of relying on standard technical indicators, it evaluates the market state and directly outputs three probabilities:
- **Long Win:** Probability of hitting a Take Profit.
- **Short Win:** Probability of hitting a Take Profit.
- **Abort:** Probability that the trade will stall, allowing for an early exit.

That type of big model doesn't need indicators as they are able to find them on their own. DeepTrader is entirely built to be exported as a `.onnx` file. The heavy neural network runs inside MT5's engine for speed and zero dependencies.

## How to use it?

### 1. Export Data from MT5
1. Copy `mt5/DeepTrader_Exporter.mq5` to your MT5 `Scripts` folder.
2. Run it on your chart. It exports EURUSD, GBPUSD, and XAUUSD on M30/H1/H4 since 2020.
3. Move the generated CSV files to the `data/raw/` folder in this project.

### 2. Training the Model
Run the Python pipeline to train the model on your data. 

pip install -r requirements.txt

# Phase 1: Pre-train the Transformer to understand market structures
python -m scripts.run_pretrain


# Phase 2: Fine-tune the model to predict the ATR-based probabilities
python -m scripts.run_finetune


# Phase 3: Align the model with DPO
python -m scripts.run_align

# Export the trained model to ONNX format
python -m scripts.run_export


### 3. Running in MT5
1. Copy the generated `models/deep_trader.onnx` file into your MetaTrader 5 `/Files/` directory.
2. Open `mt5/DeepTrader_EA.mq5` in MetaEditor and compile it.
3. Attach the compiled Expert Advisor to any M30 chart. The EA will load the ONNX model and start trading autonomously.
