import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Optional: knee for elbow selection
try:
    from kneed import KneeLocator
    _HAS_KNEED = True
except Exception:
    _HAS_KNEED = False

# Optional: fix SSL certs on macOS (harmless elsewhere)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

# Ensure Matplotlib cache dir is writable on macOS
mpl_dir = Path.home() / ".config" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
mpl_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1) Universe & Config
# ---------------------------
# Full Dow 30 universe (Yahoo Finance symbols)
DOW30_TICKERS = [
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","PG","TRV","UNH","V","VZ","WMT","INTC","DOW"
]

# Use hourly bars instead of daily
INTERVAL = "1h"  # valid: "1h"/"60m"; sub-hour intervals have shorter max lookbacks
# Note: Yahoo allows ~730 days for 1h, ~60 days for <=30m, and ~7–60 days for minutes; adjust lookbacks if needed.

# Plot lookbacks (period -> color). Only these are plotted.
LOOKBACKS = [
    ("1mo", "tab:purple"),
    ("3mo", "tab:green"),
    ("6mo", "tab:orange"),
    ("1y",  "tab:red"),
]

# Additional lookbacks computed ONLY for CSV output (won't be plotted)
CSV_EXTRA_LOOKBACKS = [
    # Note: very short windows like '3d' can have < 3 samples around weekends/holidays.
    "3d",
    "7d",
    "14d",
    "20d",
    "25d",
    # "max",
]

CSV_OUTPUT_PATH = "dj30_levels_by_lookback.csv"

# ---------------------------
# 2) Helpers
# ---------------------------

def normalize_ticker(t: str) -> str:
    """Sanitize symbols for Yahoo Finance (strip $, fix Berkshire classes)."""
    s = (t or "").strip().upper().lstrip("$")
    if s in {"BRK.B", "BRK B"}:  # Berkshire Hathaway Class B
        return "BRK-B"
    if s in {"BRK.A", "BRK A"}:  # Berkshire Hathaway Class A
        return "BRK-A"
    return s


def fetch_last_prices(ticker_list):
    """Return dict of last close prices per ticker using per‑ticker calls with retries."""
    prices = {}
    for t in ticker_list:
        sym = normalize_ticker(t)
        last_err = None
        for attempt in range(3):
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(period="10d", interval=INTERVAL, auto_adjust=False)
                if hist is None or hist.empty or "Close" not in hist:
                    hist = tk.history(period="1mo", interval=INTERVAL, auto_adjust=False)
                if hist is not None and not hist.empty and "Close" in hist:
                    prices[sym] = float(hist["Close"].dropna().iloc[-1])
                    last_err = None
                    break
                else:
                    raise RuntimeError("empty history")
            except Exception as e:
                last_err = e
                time.sleep(0.8 * (attempt + 1))
        if last_err is not None:
            print(f"[WARN] Failed to fetch {t} (normalized {sym}): {last_err}")
    return prices


def choose_k_by_elbow(values: np.ndarray, max_k: int = 10) -> int:
    """Use elbow (knee) method if available; otherwise default to a safe K.
    Ensures n_clusters never exceeds the number of available (unique) samples.
    """
    # Flatten to 1D float array and drop NaNs
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[~np.isnan(vals)]
    # Effective sample size is the number of unique points
    uniq = np.unique(vals)
    eff_n = int(len(uniq))
    if eff_n <= 1:
        # Not enough variation to cluster
        return 1
    # Cap K by both max_k and the number of unique samples
    max_k_eff = max(1, min(max_k, eff_n))
    X = vals.reshape(-1, 1)
    if not _HAS_KNEED:
        # Choose a small, safe K but never exceeding max_k_eff
        return min(5, max_k_eff)
    xs, inertias = [], []
    for k in range(1, max_k_eff + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
        xs.append(k)
        inertias.append(km.inertia_)
    kn = KneeLocator(xs, inertias, S=1.0, curve="convex", direction="decreasing")
    if getattr(kn, "knee", None):
        return int(kn.knee)
    # Fallback if no knee detected
    return min(5, max_k_eff)


def kmeans_levels_for_ticker(ticker: str, period: str = "3mo", k: int | None = None):
    """Return (last_close, support, resistance) using K‑means on daily closes."""
    sym = normalize_ticker(ticker)
    hist = yf.Ticker(sym).history(period=period, interval=INTERVAL, auto_adjust=False)
    closes = hist["Close"].dropna()
    if closes.empty:
        return (None, None, None)
    last_close = float(closes.iloc[-1])
    X = closes.values.astype(float)
    uniq = np.unique(X[~np.isnan(X)])
    # If not enough data or variation, return trivial levels at last_close
    if len(X) < 2 or len(uniq) <= 1:
        return (last_close, last_close, last_close)
    if k is None:
        # Cap max_k by unique sample count
        k = choose_k_by_elbow(X, max_k=min(10, len(uniq)))
    # Ensure n_clusters does not exceed sample count or unique values
    n_clusters = int(max(2, min(k, len(X), len(uniq))))
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(X.reshape(-1, 1))
    cents = sorted(float(c) for c in kmeans.cluster_centers_.ravel())
    below = [c for c in cents if c <= last_close]
    above = [c for c in cents if c >= last_close]
    support = max(below) if below else (min(cents) if cents else None)
    resistance = min(above) if above else (max(cents) if cents else None)
    return (last_close, support, resistance)


def build_kmeans_levels(ticker_list: list[str], period: str = "3mo", k: int | None = None):
    """Compute dicts: prices, support_levels, resistance_levels for all tickers in list."""
    prices, sup, res = {}, {}, {}
    for t in ticker_list:
        last_close, s, r = kmeans_levels_for_ticker(t, period=period, k=k)
        if last_close is not None:
            prices[normalize_ticker(t)] = last_close
        if s is not None:
            sup[normalize_ticker(t)] = float(s)
        if r is not None:
            res[normalize_ticker(t)] = float(r)
    return prices, sup, res


def select_top_n_by_price(price_dict: dict, n: int = 5) -> list[str]:
    """Return tickers of the top‑N highest prices from a dict {symbol: price}."""
    return [t for t, _ in sorted(price_dict.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def sum_point_contributions(
    prices: dict,
    levels: dict,
    weights: dict,
    dow_current: float,
    mode: str = "support",
):
    """Return (total_points, missing_tickers) for the specified mode.

    points_i = |pct_gap| * dow_current * weight_i
    pct_gap = (level - price) / price
    """
    total = 0.0
    missing = []
    for t, p in prices.items():
        if t not in levels or t not in weights:
            missing.append(t)
            continue
        level = levels[t]
        if level is None or p is None or p == 0:
            missing.append(t)
            continue
        pct = (level - p) / p
        total += abs(pct) * dow_current * float(weights[t])
    return total, missing

# ---------------------------
# 3) Main calc
# ---------------------------
if __name__ == "__main__":
    levels_by_lb = {}

    # Fetch *all* component prices to form true Dow weights
    all_prices = fetch_last_prices(DOW30_TICKERS)
    if not all_prices or len(all_prices) < 5:
        raise RuntimeError("Failed to fetch component prices for top-5 selection.")

    all_price_sum = sum(all_prices.values())
    top5 = select_top_n_by_price(all_prices, n=5)
    print("Top-5 by price:", ", ".join(top5))

    # Dow current (defensive intraday interval fetch)
    try:
        dji_hist = yf.download("^DJI", period="10d", interval=INTERVAL, progress=False, threads=False)
        dow_current = float(dji_hist["Close"].dropna().iloc[-1])
    except Exception as e:
        raise RuntimeError(f"Failed to fetch ^DJI close: {e}")

    # Compute projected levels for plotted lookbacks
    for lb, _color in LOOKBACKS:
        prices, support_levels, resistance_levels = build_kmeans_levels(top5, period=lb, k=None)
        if not prices:
            print(f"[WARN] No prices for lookback {lb}; skipping.")
            continue
        # True Dow weights: price_i / sum(all 30 prices). Do NOT renormalize to top-5.
        weights = {t: all_prices.get(t, 0.0) / all_price_sum for t in prices.keys()}
        support_points, _ = sum_point_contributions(prices, support_levels, weights, dow_current, mode="support")
        resist_points,  _ = sum_point_contributions(prices, resistance_levels, weights, dow_current, mode="resistance")
        levels_by_lb[lb] = {
            "support_points": support_points,
            "resist_points": resist_points,
            "dow_support_level": dow_current - support_points,
            "dow_resistance_level": dow_current + resist_points,
        }

    # Console summary
    print("\nProjected Dow levels by lookback (TOP-5 contributions; weights = price_i / sum(all 30 prices)):")
    for lb, _ in LOOKBACKS:
        vals = levels_by_lb.get(lb)
        if vals:
            print(f"  {lb:>3}: support −{vals['support_points']:.2f} → {vals['dow_support_level']:.2f};  "
                  f"resistance +{vals['resist_points']:.2f} → {vals['dow_resistance_level']:.2f}")

    # ---------------------------
    # 4) CSV export (plotted + CSV-only extras)
    # ---------------------------
    csv_lookbacks = [lb for lb, _ in LOOKBACKS] + [lb for lb in CSV_EXTRA_LOOKBACKS if lb not in {x for x, _ in LOOKBACKS}]
    rows = []
    run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    for lb in csv_lookbacks:
        vals = levels_by_lb.get(lb)
        if vals is None:
            # Compute on demand for CSV-only lookbacks
            prices, support_levels, resistance_levels = build_kmeans_levels(top5, period=lb, k=None)
            if not prices:
                print(f"[WARN] No prices for CSV lookback {lb}; skipping.")
                continue
            weights = {t: all_prices.get(t, 0.0) / all_price_sum for t in prices.keys()}
            support_points, _ = sum_point_contributions(prices, support_levels, weights, dow_current, mode="support")
            resist_points,  _ = sum_point_contributions(prices, resistance_levels, weights, dow_current, mode="resistance")
            vals = {
                "support_points": support_points,
                "resist_points": resist_points,
                "dow_support_level": dow_current - support_points,
                "dow_resistance_level": dow_current + resist_points,
            }
            # Save into levels_by_lb so we can plot CSV lookbacks later
            levels_by_lb[lb] = vals
        rows.append({
            "lookback": lb,
            "dow_current": round(dow_current, 2),
            "support_points": round(vals["support_points"], 2),
            "resist_points": round(vals["resist_points"], 2),
            "dow_support_level": round(vals["dow_support_level"], 2),
            "dow_resistance_level": round(vals["dow_resistance_level"], 2),
            "top5_components": ",".join(top5),
            "weights_basis": "price_i / sum(all 30 prices)",
            "run_utc": run_ts,
        })

    try:
        pd.DataFrame(rows).to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"\nCSV saved -> {CSV_OUTPUT_PATH}  (lookbacks included: {', '.join(csv_lookbacks)})")
    except Exception as e:
        print(f"[WARN] Failed to write CSV {CSV_OUTPUT_PATH}: {e}")

    # ---------------------------
    # 5) Plot (LOOKBACKS only)
    # ---------------------------
    # Determine longest requested lookback for chart range
    longest = "3mo"
    order = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
    for lb, _ in LOOKBACKS:
        try:
            if order.index(lb) > order.index(longest):
                longest = lb
        except ValueError:
            pass

    series = yf.download("^DJI", period=longest, interval=INTERVAL, progress=False)["Close"]

    plt.figure(figsize=(11, 6))
    plt.plot(series.index, series.values, label="Dow Jones (^DJI)")

    for lb, color in LOOKBACKS:
        vals = levels_by_lb.get(lb)
        if not vals:
            continue
        plt.axhline(vals["dow_support_level"], linestyle="--", linewidth=2, color=color,
                    label=f"{lb} Support ~ {vals['dow_support_level']:.0f}")
        plt.axhline(vals["dow_resistance_level"], linestyle="--", linewidth=2, color=color, alpha=0.6,
                    label=f"{lb} Resistance ~ {vals['dow_resistance_level']:.0f}")

    plt.title("Dow with K-means Projected Support/Resistance by Lookback (Top-5 by Price)")
    plt.xlabel("Date")
    plt.ylabel("Index Level (points)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 5b) Separate plot for CSV-only lookbacks
    # ---------------------------
    csv_only = [lb for lb in CSV_EXTRA_LOOKBACKS if lb not in {x for x, _ in LOOKBACKS}]
    if csv_only:
        # Determine longest requested CSV-only lookback for chart range
        longest_csv = csv_only[0]
        order = ["1d","3d","5d","7d","14d","20d","25d","1mo","3mo","6mo","1y","2y","5y","max"]
        for lb in csv_only:
            try:
                if order.index(lb) > order.index(longest_csv):
                    longest_csv = lb
            except ValueError:
                # If not in order list, fall back to the first
                pass

        series_csv = yf.download("^DJI", period=longest_csv, interval=INTERVAL, progress=False)["Close"]

        plt.figure(figsize=(11, 6))
        plt.plot(series_csv.index, series_csv.values, label="Dow Jones (^DJI)")

        palette = [
            "tab:blue","tab:orange","tab:green","tab:red","tab:purple",
            "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan",
        ]
        for idx, lb in enumerate(csv_only):
            vals = levels_by_lb.get(lb)
            if not vals:
                continue
            color = palette[idx % len(palette)]
            plt.axhline(vals["dow_support_level"], linestyle="--", linewidth=2, color=color,
                        label=f"{lb} Support ~ {vals['dow_support_level']:.0f}")
            plt.axhline(vals["dow_resistance_level"], linestyle="--", linewidth=2, color=color, alpha=0.6,
                        label=f"{lb} Resistance ~ {vals['dow_resistance_level']:.0f}")

        plt.title("Dow with K-means Projected S/R — CSV-Only Lookbacks")
        plt.xlabel("Date")
        plt.ylabel("Index Level (points)")
        plt.legend()
        plt.tight_layout()
        plt.show()