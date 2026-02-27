//+------------------------------------------------------------------+
//|                                      DeepTrader_Exporter.mq5     |
//|                        Export OHLCV data to CSV for model training|
//+------------------------------------------------------------------+
#property copyright "DeepTrader"
#property version   "1.00"
#property script_show_inputs

//--- Input parameters
input string   InpSymbols    = "EURUSD,GBPUSD,XAUUSD";     // Symbols (comma-separated)
input datetime InpStartDate  = D'2020.01.01 00:00';         // Start date
input datetime InpEndDate    = D'2026.02.26 00:00';         // End date
input string   InpOutputDir  = "DeepTrader";                // Output folder in MQL5/Files/

//+------------------------------------------------------------------+
//| Timeframe info structure                                          |
//+------------------------------------------------------------------+
struct TimeframeInfo
{
   ENUM_TIMEFRAMES tf;
   string          name;
};

//+------------------------------------------------------------------+
//| Script program start function                                     |
//+------------------------------------------------------------------+
void OnStart()
{
   // Define timeframes
   TimeframeInfo timeframes[];
   ArrayResize(timeframes, 3);
   timeframes[0].tf = PERIOD_M30;  timeframes[0].name = "M30";
   timeframes[1].tf = PERIOD_H1;   timeframes[1].name = "H1";
   timeframes[2].tf = PERIOD_H4;   timeframes[2].name = "H4";

   // Parse symbols
   string symbols[];
   int n_symbols = StringSplit(InpSymbols, ',', symbols);

   Print("═══════════════════════════════════════════════════");
   Print("  DeepTrader Data Export");
   Print("  Symbols: ", InpSymbols);
   Print("  Range: ", TimeToString(InpStartDate), " → ", TimeToString(InpEndDate));
   Print("═══════════════════════════════════════════════════");

   int total_exported = 0;

   for(int s = 0; s < n_symbols; s++)
   {
      string symbol = symbols[s];
      StringTrimLeft(symbol);
      StringTrimRight(symbol);

      // Ensure symbol is available
      if(!SymbolSelect(symbol, true))
      {
         Print("[ERROR] Cannot select symbol: ", symbol);
         continue;
      }

      for(int t = 0; t < ArraySize(timeframes); t++)
      {
         string tf_name = timeframes[t].name;
         ENUM_TIMEFRAMES tf = timeframes[t].tf;

         Print("──────────────────────────────────────────────");
         Print("  Exporting: ", symbol, " ", tf_name);

         // Fetch rates
         MqlRates rates[];
         int copied = CopyRates(symbol, tf, InpStartDate, InpEndDate, rates);

         if(copied <= 0)
         {
            Print("  [WARN] No data for ", symbol, " ", tf_name, " Error: ", GetLastError());
            continue;
         }

         Print("  Bars fetched: ", copied);

         // Build filename
         string filename = InpOutputDir + "\\" + symbol + "_" + tf_name + ".csv";

         // Open file
         int handle = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
         if(handle == INVALID_HANDLE)
         {
            Print("  [ERROR] Cannot create file: ", filename, " Error: ", GetLastError());
            continue;
         }

         // Write header
         FileWrite(handle, "datetime", "open", "high", "low", "close",
                   "tick_volume", "real_volume", "spread");

         // Write data
         for(int i = 0; i < copied; i++)
         {
            FileWrite(handle,
                      TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES | TIME_SECONDS),
                      DoubleToString(rates[i].open, 6),
                      DoubleToString(rates[i].high, 6),
                      DoubleToString(rates[i].low, 6),
                      DoubleToString(rates[i].close, 6),
                      IntegerToString(rates[i].tick_volume),
                      IntegerToString(rates[i].real_volume),
                      IntegerToString(rates[i].spread));
         }

         FileClose(handle);
         Print("  ✓ Saved: ", filename, " (", copied, " bars)");
         total_exported++;
      }
   }

   Print("═══════════════════════════════════════════════════");
   Print("  Export complete: ", total_exported, " files");
   Print("  Location: MQL5/Files/", InpOutputDir, "/");
   Print("═══════════════════════════════════════════════════");
}
//+------------------------------------------------------------------+
