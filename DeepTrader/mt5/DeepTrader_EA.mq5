//+------------------------------------------------------------------+
//|                                          DeepTrader_EA.mq5       |
//|           Expert Advisor — ONNX-based trajectory trading         |
//+------------------------------------------------------------------+
#property copyright "DeepTrader"
#property version   "1.00"
#property strict
#property tester_file "deep_trader.onnx"

//--- (ONNX model loaded from file array dynamically in OnInit)

//--- Input parameters
input group "═══ Model Settings ═══"
input int      InpSeqLen       = 256;          // M30 context window
input int      InpH1Window     = 100;          // H1 context bars
input int      InpH4Window     = 50;           // H4 context bars
input int      InpClasses      = 3;            // Output probabilities (Long, Short, Abort)
input int      InpFeatures     = 17;           // Number of input features

input group "═══ Trading Settings ═══"
input double   InpMinConfidence = 0.55;        // Minimum probability for entry
input double   InpRiskPercent   = 1.0;         // Risk per trade (%)
input double   InpSLMultiplier  = 1.5;         // SL = ATR × multiplier
input double   InpTPMultiplier  = 2.0;         // TP = ATR × multiplier
input int      InpMagicNumber   = 202601;      // Magic number
input int      InpMaxTrades     = 3;           // Maximum concurrent trades

input group "═══ Time Filter ═══"
input int      InpStartHour    = 4;            // Trading start hour (UTC)
input int      InpEndHour      = 20;           // Trading end hour (UTC)

//--- Global variables
long           g_onnx_handle = INVALID_HANDLE;
int            g_atr_handle  = INVALID_HANDLE;
datetime       g_last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   // Load ONNX model from Common/Files folder to bypass Strategy Tester sandbox
   g_onnx_handle = OnnxCreate("deep_trader.onnx", FILE_COMMON);
   if(g_onnx_handle == INVALID_HANDLE)
   {
      Print("[ERROR] Failed to load ONNX model (Ensure it is in MQL5/Files/): ", GetLastError());
      return INIT_FAILED;
   }

   // Define input shapes
   long m30_shape[]  = {1, InpSeqLen, InpFeatures};
   long h1_shape[]   = {1, InpH1Window, InpFeatures};
   long h4_shape[]   = {1, InpH4Window, InpFeatures};

   if(!OnnxSetInputShape(g_onnx_handle, 0, m30_shape) ||
      !OnnxSetInputShape(g_onnx_handle, 1, h1_shape) ||
      !OnnxSetInputShape(g_onnx_handle, 2, h4_shape))
   {
      Print("[ERROR] Failed to set ONNX input shapes: ", GetLastError());
      return INIT_FAILED;
   }

   // Define output shapes
   long probs_shape[] = {1, InpClasses};

   if(!OnnxSetOutputShape(g_onnx_handle, 0, probs_shape))
   {
      Print("[ERROR] Failed to set ONNX output shapes: ", GetLastError());
      return INIT_FAILED;
   }

   // ATR indicator
   g_atr_handle = iATR(_Symbol, PERIOD_M30, 14);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("[ERROR] Failed to create ATR indicator");
      return INIT_FAILED;
   }

   Print("═══════════════════════════════════════════════════");
   Print("  DeepTrader EA Initialized");
   Print("  Symbol: ", _Symbol);
   Print("  Model: deep_trader.onnx");
   Print("  Confidence threshold: ", InpMinConfidence);
   Print("═══════════════════════════════════════════════════");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_onnx_handle != INVALID_HANDLE)
      OnnxRelease(g_onnx_handle);
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);

   Print("DeepTrader EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only process on new M30 bar
   datetime bar_time = iTime(_Symbol, PERIOD_M30, 0);
   if(bar_time == g_last_bar_time)
      return;
   g_last_bar_time = bar_time;

   // Time filter
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.hour < InpStartHour || dt.hour >= InpEndHour)
      return;

   // Check max trades
   if(CountOpenTrades() >= InpMaxTrades)
      return;

   // Get ATR
   double atr[];
   if(CopyBuffer(g_atr_handle, 0, 1, 1, atr) <= 0)
      return;
   double current_atr = atr[0];

   // Prepare input data
   float m30_input[];
   float h1_input[];
   float h4_input[];

   if(!PrepareInputs(m30_input, h1_input, h4_input))
      return;

   // Run ONNX inference
   float target_probs[];
   ArrayResize(target_probs, InpClasses);

   if(!OnnxRun(g_onnx_handle, ONNX_DEFAULT,
               m30_input, h1_input, h4_input,
               target_probs))
   {
      Print("[ERROR] ONNX inference failed: ", GetLastError());
      return;
   }

   // Analyze probabilities
   double p_long = target_probs[0];
   double p_short = target_probs[1];
   double p_abort = target_probs[2];

   // Check signal
   if(p_abort > 0.5) return; // Model predicts stall
   
   int signal_direction = 0; // 1 = Long, -1 = Short
   double confidence = 0;

   if(p_long > p_short && p_long >= InpMinConfidence)
   {
      signal_direction = 1;
      confidence = p_long;
   }
   else if(p_short > p_long && p_short >= InpMinConfidence)
   {
      signal_direction = -1;
      confidence = p_short;
   }

   if(signal_direction == 0) return;

   // Calculate position sizing
   double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl_distance = current_atr * InpSLMultiplier;
   double tp_distance = current_atr * InpTPMultiplier;
   double lots = CalculateLots(sl_distance);

   if(lots <= 0)
      return;

   // Execute trade
   if(signal_direction == 1)
   {
      // LONG signal
      double sl = price - sl_distance;
      double tp = price + tp_distance;
      ExecuteTrade(ORDER_TYPE_BUY, lots, price, sl, tp,
                   StringFormat("DT|L|P=%.2f", confidence));
   }
   else
   {
      // SHORT signal
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + sl_distance;
      double tp = price - tp_distance;
      ExecuteTrade(ORDER_TYPE_SELL, lots, price, sl, tp,
                   StringFormat("DT|S|P=%.2f", confidence));
   }
}

//+------------------------------------------------------------------+
//| Prepare normalized input tensors                                  |
//+------------------------------------------------------------------+
bool PrepareInputs(float &m30_out[], float &h1_out[], float &h4_out[])
{
   ArrayResize(m30_out, InpSeqLen * InpFeatures);
   ArrayResize(h1_out, InpH1Window * InpFeatures);
   ArrayResize(h4_out, InpH4Window * InpFeatures);

   // Fetch M30 candles
   MqlRates m30_rates[];
   if(CopyRates(_Symbol, PERIOD_M30, 1, InpSeqLen, m30_rates) < InpSeqLen)
      return false;

   // Fetch H1 candles
   MqlRates h1_rates[];
   if(CopyRates(_Symbol, PERIOD_H1, 1, InpH1Window, h1_rates) < InpH1Window)
      return false;

   // Fetch H4 candles
   MqlRates h4_rates[];
   if(CopyRates(_Symbol, PERIOD_H4, 1, InpH4Window, h4_rates) < InpH4Window)
      return false;

   // Compute normalization stats from M30 window (z-score)
   double price_sum = 0;
   double price_sq_sum = 0;
   int n_prices = 0;
   for(int i = 0; i < InpSeqLen; i++)
   {
      price_sum += m30_rates[i].open + m30_rates[i].high +
                   m30_rates[i].low + m30_rates[i].close;
      price_sq_sum += MathPow(m30_rates[i].open, 2) + MathPow(m30_rates[i].high, 2) +
                      MathPow(m30_rates[i].low, 2) + MathPow(m30_rates[i].close, 2);
      n_prices += 4;
   }
   double mu = price_sum / n_prices;
   double sigma = MathSqrt(price_sq_sum / n_prices - mu * mu);
   if(sigma < 1e-10) sigma = 1e-10;

   // Encode M30
   EncodeCandles(m30_rates, InpSeqLen, m30_out, mu, sigma, PERIOD_M30);

   // Encode H1 (using same normalization as M30)
   EncodeCandles(h1_rates, InpH1Window, h1_out, mu, sigma, PERIOD_H1);

   // Encode H4
   EncodeCandles(h4_rates, InpH4Window, h4_out, mu, sigma, PERIOD_H4);

   return true;
}

//+------------------------------------------------------------------+
//| Encode candle array into 17-feature vectors                       |
//+------------------------------------------------------------------+
void EncodeCandles(MqlRates &rates[], int count, float &output[],
                   double mu, double sigma, ENUM_TIMEFRAMES tf)
{
   double atr_buf[];
   int atr_h = iATR(_Symbol, tf, 14);
   CopyBuffer(atr_h, 0, 1, count, atr_buf);

   double vol_sma[];
   ArrayResize(vol_sma, count);

   // Compute volume SMA(20)
   for(int i = 0; i < count; i++)
   {
      double vsum = 0;
      int vcount = 0;
      for(int j = MathMax(0, i - 19); j <= i; j++)
      {
         vsum += (double)rates[j].tick_volume;
         vcount++;
      }
      vol_sma[i] = (vcount > 0) ? vsum / vcount : 1.0;
   }

   for(int i = 0; i < count; i++)
   {
      int base = i * InpFeatures;
      double o = rates[i].open;
      double h = rates[i].high;
      double l = rates[i].low;
      double c = rates[i].close;
      double v = (double)rates[i].tick_volume;
      double range = h - l + 1e-10;
      double atr_val = (i < ArraySize(atr_buf) && atr_buf[i] > 1e-10) ? atr_buf[i] : range;

      // Feature 0-3: Normalized OHLC
      output[base + 0]  = (float)((o - mu) / sigma);
      output[base + 1]  = (float)((h - mu) / sigma);
      output[base + 2]  = (float)((l - mu) / sigma);
      output[base + 3]  = (float)((c - mu) / sigma);

      // Feature 4: body_ratio
      output[base + 4]  = (float)((c - o) / range);

      // Feature 5: upper_wick
      output[base + 5]  = (float)((h - MathMax(o, c)) / range);

      // Feature 6: lower_wick
      output[base + 6]  = (float)((MathMin(o, c) - l) / range);

      // Feature 7: close_position
      output[base + 7]  = (float)((c - l) / range);

      // Feature 8: log_return
      double prev_c = (i > 0) ? rates[i-1].close : c;
      output[base + 8]  = (float)(MathLog(c / (prev_c + 1e-10)));

      // Feature 9: atr_ratio
      output[base + 9]  = (float)(range / atr_val);

      // Feature 10: rel_volume
      output[base + 10] = (float)(v / MathMax(vol_sma[i], 1.0));

      // Feature 11: vol_price_corr (simplified — set to 0 for inference)
      output[base + 11] = 0.0f;

      // Feature 12: gap
      double prev_close = (i > 0) ? rates[i-1].close : o;
      output[base + 12] = (float)((o - prev_close) / atr_val);

      // Feature 13-14: hour sin/cos
      MqlDateTime dt;
      TimeToStruct(rates[i].time, dt);
      double hour_frac = dt.hour + dt.min / 60.0;
      output[base + 13] = (float)(MathSin(2.0 * M_PI * hour_frac / 24.0));
      output[base + 14] = (float)(MathCos(2.0 * M_PI * hour_frac / 24.0));

      // Feature 15-16: dow sin/cos
      output[base + 15] = (float)(MathSin(2.0 * M_PI * dt.day_of_week / 5.0));
      output[base + 16] = (float)(MathCos(2.0 * M_PI * dt.day_of_week / 5.0));
   }

   if(atr_h != INVALID_HANDLE)
      IndicatorRelease(atr_h);
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk                                  |
//+------------------------------------------------------------------+
double CalculateLots(double sl_distance)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = balance * InpRiskPercent / 100.0;

   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tick_value <= 0 || tick_size <= 0 || sl_distance <= 0)
      return 0;

   double ticks_in_sl = sl_distance / tick_size;
   double lots = risk_amount / (ticks_in_sl * tick_value);

   // Round to lot step
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double lot_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lot_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   lots = MathFloor(lots / lot_step) * lot_step;
   lots = MathMax(lot_min, MathMin(lot_max, lots));

   return lots;
}

//+------------------------------------------------------------------+
//| Execute trade                                                     |
//+------------------------------------------------------------------+
bool ExecuteTrade(ENUM_ORDER_TYPE type, double lots, double price,
                  double sl, double tp, string comment)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = _Symbol;
   request.volume    = lots;
   request.type      = type;
   request.price     = price;
   request.sl        = NormalizeDouble(sl, _Digits);
   request.tp        = NormalizeDouble(tp, _Digits);
   request.deviation = 20;
   request.magic     = InpMagicNumber;
   request.comment   = comment;
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result))
   {
      Print("[ERROR] Order failed: ", result.retcode, " - ", result.comment);
      return false;
   }

   Print("  ✓ ", (type == ORDER_TYPE_BUY ? "BUY" : "SELL"), " ",
         DoubleToString(lots, 2), " @ ", DoubleToString(price, _Digits),
         " SL=", DoubleToString(sl, _Digits),
         " TP=", DoubleToString(tp, _Digits));
   return true;
}

//+------------------------------------------------------------------+
//| Count current open trades with our magic number                   |
//+------------------------------------------------------------------+
int CountOpenTrades()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         count++;
   }
   return count;
}
//+------------------------------------------------------------------+
