import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------
# 1) Inputs (make these consistent!)
# ---------------------------
# Your current tickers list (clean or replace with the *actual Dow 30* you want to use)
tickers = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","JPM","JNJ","V","PG",
    "UNH","HD","MA","DIS","PYPL","BAC","VZ","ADBE","CMCSA","NFLX","XOM","KO","PFE",
    "T","PEP","CSCO","INTC","WMT","CVX"
]

# Hypothetical manual levels (example). Ensure you include ONLY tickers you intend to use.
support_levels = {
    "AAPL": 260.0, "AMGN": 290.0, "AMZN": 220.0, "AXP": 355.0, "BA": 220.0, "CAT": 520.0,
    "CRM": 250.0, "CSCO": 70.0, "CVX": 155.0, "DIS": 110.0, "GS": 780.0, "HD": 385.0,
    "HON": 215.0, "IBM": 305.0, "JNJ": 190.0, "JPM": 300.0, "KO": 65.0, "MCD": 305.0,
    "MMM": 165.0, "MRK": 85.0, "MSFT": 520.0, "NKE": 65.0, "NVDA": 185.0, "PG": 150.0,
    "SHW": 330.0, "TRV": 265.0, "UNH": 360.0, "V": 345.0, "VZ": 35.0, "WMT": 105.0
}
resistance_levels = {
    "AAPL": 265.0, "AMGN": 295.0, "AMZN": 225.0, "AXP": 360.0, "BA": 225.0, "CAT": 525.0,
    "CRM": 255.0, "CSCO": 75.0, "CVX": 160.0, "DIS": 115.0, "GS": 785.0, "HD": 390.0,
    "HON": 220.0, "IBM": 310.0, "JNJ": 195.0, "JPM": 305.0, "KO": 70.0, "MCD": 310.0,
    "MMM": 170.0, "MRK": 90.0, "MSFT": 525.0, "NKE": 70.0, "NVDA": 190.0, "PG": 155.0,
    "SHW": 335.0, "TRV": 270.0, "UNH": 365.0, "V": 350.0, "VZ": 40.0, "WMT": 110.0
}

# ---------------------------
# 2) Helpers
# ---------------------------
def fetch_last_prices(ticker_list):
    """Return dict of last close prices for each ticker."""
    df = yf.download(ticker_list, period="1d", group_by="ticker", progress=False)
    # yfinance returns multi-columns when group_by='ticker'; handle both shapes robustly
    prices = {}
    for t in ticker_list:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                prices[t] = float(df[t]['Close'].iloc[-1])
            else:
                # single-index fallback (unlikely for multiple tickers)
                prices[t] = float(df['Close'][t].iloc[-1])
        except Exception:
            prices[t] = None
    return {t: p for t, p in prices.items() if p is not None} 

def compute_price_weights(price_dict):
    """Price weights for a price-weighted index (sum to 1)."""
    total = sum(price_dict.values())
    return {t: price_dict[t] / total for t in price_dict}

def pct_move(current, target):
    """Signed percent move from current to target, e.g., (target-current)/current."""
    return (target - current) / current

def sum_point_contributions(
    prices: dict,
    levels: dict,   # support or resistance dict
    weights: dict,
    dow_current: float,
    mode: str = "support"
):
    """
    For each ticker:
      1) compute % distance to level (support/resistance)
      2) convert to Dow points: |%| * Dow_current
      3) multiply by price weight
    Sum across all -> total Dow points to move (always positive).
    """
    total_points = 0.0
    missing_levels = []
    for t, px in prices.items():
        if t not in levels:
            missing_levels.append(t)
            continue
        # percent move to the level (signed)
        pct = pct_move(px, levels[t])
        # take magnitude: we only care about *distance* in points
        pts = abs(pct) * dow_current
        w = weights.get(t, 0.0)
        total_points += pts * w

    return total_points, missing_levels

# ---------------------------
# 3) Main calc
# ---------------------------
# Fetch current component prices
prices = fetch_last_prices(tickers)
if not prices:
    raise RuntimeError("No prices fetched. Check ticker symbols or network.")

# Current Dow value
dow_current = float(yf.download("^DJI", period="1d", progress=False)["Close"].iloc[-1])

# Compute price weights
weights = compute_price_weights(prices)

# Support & resistance total point contributions (always positive point distances)
support_points, missing_sup = sum_point_contributions(prices, support_levels, weights, dow_current, mode="support")
resist_points,  missing_res = sum_point_contributions(prices, resistance_levels, weights, dow_current, mode="resistance")

# Projected Dow levels (apply your rule: subtract support points, add resistance points)
dow_support_level = dow_current - support_points
dow_resistance_level = dow_current + resist_points

print(f"Dow current:        {dow_current:,.2f}")
print(f"Support distance:   {support_points:,.2f} pts  -> projected support:   {dow_support_level:,.2f}")
print(f"Resistance distance:{resist_points:,.2f} pts  -> projected resistance:{dow_resistance_level:,.2f}")

# Warn about any tickers missing levels or price
tick_set = set(tickers)
miss_price = tick_set.difference(prices.keys())
if miss_price:
    print("\n[WARN] Missing prices for:", sorted(miss_price))
if missing_sup:
    print("[WARN] No SUPPORT level provided for:", sorted(missing_sup))
if missing_res:
    print("[WARN] No RESISTANCE level provided for:", sorted(missing_res))

# ---------------------------
# 4) Simple visualization (last 3 months)
# ---------------------------
djia_hist = yf.download("^DJI", period="3mo", interval="1d", progress=False)["Close"]

plt.figure(figsize=(11,6))
plt.plot(djia_hist.index, djia_hist.values, label="Dow Jones (^DJI)")
plt.axhline(dow_support_level, linestyle="--", label=f"Projected Support ~ {dow_support_level:,.0f}")
plt.axhline(dow_resistance_level, linestyle="--", label=f"Projected Resistance ~ {dow_resistance_level:,.0f}")
plt.title("Dow Jones with Projected Support/Resistance from Component Levels")
plt.xlabel("Date")
plt.ylabel("Index Level (points)")
plt.legend()
plt.tight_layout()
plt.show()
