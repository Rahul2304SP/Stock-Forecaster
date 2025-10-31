//+------------------------------------------------------------------+
//|                                            XOiosV7_Multi.mq5     |
//|    Multi-instrument compatible version of XOiosV7_MQL5           |
//|    Refactored by ChatGPT for dynamic symbol handling             |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, Rahul Shaji Parmeshwar."
#property link      "https://www.mql5.com"
#property version   "1.10"
#include <Trade\Trade.mqh>
CTrade trade;

#define  BUTTON_NAME "LiquidatePosition"

//--- Inputs
input double XPips = 0;
input double XPipsStop = 4;
input bool OGTradeSL = false;
input double Lotsize = 0.1;
input double LotsizeIncrement = 0.1;
input double XPValue = 1;
input double BreakevenDetectBuy = 5;
input double BreakevenBuy = 2;
input double BreakevenDetectSell = 5;
input double BreakevenSell = 2;
input bool AllStopsEnable = true;
input bool StopTrail = false;
input int Magic = 343353;

bool firstTradeExecuted = false;

struct SymbolInfoStruct {
   string symbol;
   double minLot;
   double lotStep;
   int digits;
   double point;
   double ask;
   double bid;
   double spread;
   bool tradeable;
};

bool GetSymbolInfo(SymbolInfoStruct &info, string symbol)
{
   info.symbol = symbol;
   info.tradeable = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_FULL;
   info.digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   info.point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   info.minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   info.lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   SymbolInfoDouble(symbol, SYMBOL_ASK, info.ask);
   SymbolInfoDouble(symbol, SYMBOL_BID, info.bid);
   info.spread = info.ask - info.bid;
   return info.tradeable;
}

// Normalize volume to broker's lot step
double GetValidVolume(SymbolInfoStruct &info, double desired)
{
   double normalized = MathMax(desired, info.minLot);
   normalized = NormalizeDouble(normalized, (int)MathLog10(1.0 / info.lotStep));
   return normalized;
}

int OnInit()
{
   if(!ObjectCreate(0, BUTTON_NAME, OBJ_BUTTON, 0, 0, 0))
      return INIT_FAILED;

   ObjectSetInteger(0, BUTTON_NAME, OBJPROP_XSIZE, 25);
   ObjectSetInteger(0, BUTTON_NAME, OBJPROP_YSIZE, 25);
   ObjectSetInteger(0, BUTTON_NAME, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(0, BUTTON_NAME, OBJPROP_YDISTANCE, 75);
   ObjectSetInteger(0, BUTTON_NAME, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, BUTTON_NAME, OBJPROP_BGCOLOR, clrRed);
   ObjectSetString(0, BUTTON_NAME, OBJPROP_TEXT, "X");

   return INIT_SUCCEEDED;
}

void OnTick()
{
   SymbolInfoStruct info;
   if(!GetSymbolInfo(info, _Symbol))
      return;

   if(!firstTradeExecuted)
      return;

   double buyPriceLast = 0, sellPriceLast = 0;
   datetime lastTimeBuy = 0, lastTimeSell = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetTicket(i) <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != info.symbol) continue;

      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
      {
         datetime t = (datetime)PositionGetInteger(POSITION_TIME);
         if(t > lastTimeBuy)
         {
            buyPriceLast = PositionGetDouble(POSITION_PRICE_OPEN);
            lastTimeBuy = t;
         }
      }
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
      {
         datetime t = (datetime)PositionGetInteger(POSITION_TIME);
         if(t > lastTimeSell)
         {
            sellPriceLast = PositionGetDouble(POSITION_PRICE_OPEN);
            lastTimeSell = t;
         }
      }
   }

   double volume = GetValidVolume(info, Lotsize);

   if(buyPriceLast > 0 && info.ask > buyPriceLast + info.spread + XPips * info.point)
   {
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      request.action = TRADE_ACTION_DEAL;
      request.symbol = info.symbol;
      request.volume = volume;
      request.type = ORDER_TYPE_BUY;
      request.price = info.ask;
      request.deviation = 5;
      request.magic = Magic;
      OrderSend(request, result);
   }

   if(sellPriceLast > 0 && info.bid < sellPriceLast - info.spread - XPips * info.point)
   {
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      request.action = TRADE_ACTION_DEAL;
      request.symbol = info.symbol;
      request.volume = volume;
      request.type = ORDER_TYPE_SELL;
      request.price = info.bid;
      request.deviation = 10;
      request.magic = Magic;
      OrderSend(request, result);
   }

   if(AllStopsEnable)
      ApplyTrailingStop(info);
}

void ApplyTrailingStop(SymbolInfoStruct &info)
{
   double buySL = NormalizeDouble(info.bid - XPipsStop * info.point, info.digits);
   double sellSL = NormalizeDouble(info.ask + XPipsStop * info.point, info.digits);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetTicket(i) <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != info.symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != Magic) continue;

      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      ulong ticket = PositionGetTicket(i);
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      if(!StopTrail)
      {
         if(type == POSITION_TYPE_BUY && buySL > openPrice + XPValue * info.point && (currentSL == 0 || buySL > currentSL))
            trade.PositionModify(ticket, buySL, PositionGetDouble(POSITION_TP));

         if(type == POSITION_TYPE_SELL && sellSL < openPrice - XPValue * info.point && (currentSL == 0 || sellSL < currentSL))
            trade.PositionModify(ticket, sellSL, PositionGetDouble(POSITION_TP));
      }

      if(type == POSITION_TYPE_BUY && info.ask > openPrice + BreakevenDetectBuy * info.point && currentSL == 0)
         trade.PositionModify(ticket, openPrice + BreakevenBuy * info.point, 0);

      if(type == POSITION_TYPE_SELL && info.bid < openPrice - BreakevenDetectSell * info.point && currentSL == 0)
         trade.PositionModify(ticket, openPrice - BreakevenSell * info.point, 0);
   }
}

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   // When the Liquidate Position button is clicked
   if(sparam == BUTTON_NAME)
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(PositionGetSymbol(i) == _Symbol)
         {
            ulong ticket = PositionGetTicket(i);
            trade.PositionClose(ticket);
         }
      }

      // Reset the button state
      ObjectSetInteger(0, BUTTON_NAME, OBJPROP_STATE, false);

      // Reset first trade flag for this symbol
      firstTradeExecuted = false;
   }

   // Set firstTradeExecuted to true when a trade is detected
   if(!firstTradeExecuted)
   {
      for(int i = 0; i < PositionsTotal(); i++)
      {
         if(PositionGetSymbol(i) == _Symbol)
         {
            firstTradeExecuted = true;
            break;
         }
      }
   }
}
