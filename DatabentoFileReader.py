import databento as db
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

MSFT = "E:/Data/1 Min Data/XNAS-20251101-RDYVM6QWH5/MSFT.dbn"
GS = "E:/Data/1 Min Data/XNAS-20251101-RDYVM6QWH5/GS.dbn"
AXP = "E:/Data/1 Min Data/XNAS-20251101-RDYVM6QWH5/AXP.dbn"
HD = "E:/Data/1 Min Data/XNAS-20251101-RDYVM6QWH5/HD.dbn"
CAT = "E:/Data/1 Min Data/XNAS-20251101-RDYVM6QWH5/CAT.dbn"
US30 = "E:/Data/1 Min Data/XNAS-20251101-PESNAKLB7V/US30.dbn"

import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("NUMEXPR_MAX_THREADS", str(os.cpu_count() or 6))

# =============================
# Configuration: file paths
# =============================
# Update these paths as needed. Keys must be the tickers you want in the output.
TICKER_PATHS: Dict[str, str] = {
    "MSFT": MSFT,
    "GS":   GS,
    "AXP":  AXP,
    "HD":   HD,
    "CAT":  CAT,
}

# Optional index proxy (Dow). If present, it will be joined into the wide frame as well.
US30_PATH = US30
US30_TICKER = "US30"

# Output locations
OUT_DIR = Path("C:/Users/Rahul Parmeshwar/Documents/GitHub/Stock-Forecaster/Outputs/Dow30Levels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# Helpers
# =============================

def _ensure_ts_col(df: pd.DataFrame) -> str:
    """Return the timestamp column name (prefer explicit ts_*),
    or '__index__' if the index is already datetime-like.
    Looks case-insensitively and falls back to any datetime-typed column.
    """
    # If index is already datetime-like, use it
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"

    # Common timestamp column names (case-insensitive)
    preferred = (
        "ts_event", "ts_recv", "ts_ref", "ts", "timestamp", "time", "date", "ts_exchange"
    )
    lower_map = {c.lower(): c for c in df.columns}

    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    # Fallback: any datetime-like column
    for c in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        except Exception:
            pass

    raise KeyError(
        f"No timestamp column found. Available columns: {list(df.columns)}"
    )


def load_dbn_ohlcv_1s(path: str, ticker: str) -> pd.DataFrame:
    """Load a .dbn ohlcv-1s file -> pandas DataFrame indexed by UTC timestamp.

    Columns after rename:
      {ticker}_open, {ticker}_high, {ticker}_low, {ticker}_close, {ticker}_volume
    """
    store = db.DBNStore.from_file(path)
    df = store.to_df()
    # Debug: uncomment to inspect schema
    # print(f"[DEBUG] {ticker} columns: {list(df.columns)}")

    ts_col = _ensure_ts_col(df)
    # Normalize timestamp to tz-aware UTC
    if ts_col == "__index__":
        # Ensure index is tz-aware UTC
        df.index = pd.to_datetime(df.index, utc=True)
        ts_series = df.index
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        ts_series = df[ts_col]

    # Some schemas may use different column case; standardize safely
    colmap = {c.lower(): c for c in df.columns}

    def get(col: str) -> str:
        # return the actual column name matching lowercase key
        if col in colmap:
            return colmap[col]
        raise KeyError(f"Column '{col}' not found in DBN file for {ticker}")

    # Map/rename
    rename_map = {
        get("open"):   f"{ticker}_open",
        get("high"):   f"{ticker}_high",
        get("low"):    f"{ticker}_low",
        get("close"):  f"{ticker}_close",
        get("volume"): f"{ticker}_volume",
    }

    if ts_col == "__index__":
        out = df[list(rename_map.keys())].rename(columns=rename_map)
        out.index.name = "timestamp"
    else:
        out = df[[ts_col, *rename_map.keys()]].rename(columns=rename_map)
        out = out.set_index(ts_col)
    out = out.sort_index()

    # Drop duplicate timestamps keeping the last tick within the second
    out = out[~out.index.duplicated(keep="last")]

    # Downcast floats to float32 for memory efficiency; volume to float32/int64
    for c in out.columns:
        if c.endswith(("_open", "_high", "_low", "_close")):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        elif c.endswith("_volume"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if np.issubdtype(out[c].dtype, np.floating):
                out[c] = out[c].astype("float32")
            else:
                out[c] = out[c].astype("int64", errors="ignore")

    return out


def _load_one(ticker: str, path: str) -> Tuple[str, pd.DataFrame]:
    """Helper to load a single ticker file (for multiprocess pools)."""
    frame = load_dbn_ohlcv_1s(path, ticker)
    return ticker, frame


def combine_wide(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join a list of per-ticker frames on timestamp index."""
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1, join="outer").sort_index()


def align_to_one_second(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete 1-second UTC timeline spanning the data range."""
    if df.empty:
        return df
    start = df.index.min().floor("S")
    end = df.index.max().ceil("S")
    full_index = pd.date_range(start=start, end=end, freq="S", tz=df.index.tz)
    return df.reindex(full_index)


def resample_to_1min_ohlcv(wide_1s: pd.DataFrame) -> pd.DataFrame:
    """Resample a ticker-prefixed wide 1-second OHLCV frame to 1-minute OHLCV, per ticker.
    open=first, high=max, low=min, close=last, volume=sum
    """
    if wide_1s.empty:
        return wide_1s

    agg: Dict[str, str] = {}
    for col in wide_1s.columns:
        if col.endswith("_open"):
            agg[col] = "first"
        elif col.endswith("_high"):
            agg[col] = "max"
        elif col.endswith("_low"):
            agg[col] = "min"
        elif col.endswith("_close"):
            agg[col] = "last"
        elif col.endswith("_volume"):
            agg[col] = "sum"

    return wide_1s.resample("1min").agg(agg)


def add_returns(wide_1s: pd.DataFrame, periods: Tuple[int, ...] = (1,)) -> pd.DataFrame:
    """Add per-ticker close-to-close returns columns for given periods (in seconds)."""
    out = wide_1s.copy()
    for col in [c for c in out.columns if c.endswith("_close")]:
        for p in periods:
            out[f"{col}_ret{p}s"] = out[col].pct_change(p)
    return out


# =============================
# Main
# =============================
if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 50)

    # Load components using all available CPU cores
    frames: List[pd.DataFrame] = []
    load_queue: List[Tuple[str, str]] = []
    for ticker, path in TICKER_PATHS.items():
        if not Path(path).exists():
            print(f"[WARN] File not found for {ticker}: {path}")
            continue
        load_queue.append((ticker, path))

    if not load_queue:
        raise SystemExit("No component data loaded; aborting.")

    loaded_frames: Dict[str, pd.DataFrame] = {}
    max_workers = min(len(load_queue), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_load_one, ticker, path): ticker for ticker, path in load_queue
        }
        for fut in as_completed(future_map):
            ticker = future_map[fut]
            try:
                ticker_key, frame = fut.result()
            except Exception as e:
                print(f"[ERROR] Failed loading {ticker}: {e}")
                continue
            loaded_frames[ticker_key] = frame
            print(f"Loaded {ticker_key}: {len(frame):,} rows from {frame.index.min()} to {frame.index.max()}")

    if not loaded_frames:
        raise SystemExit("No component data loaded; aborting.")

    frames = [loaded_frames[t] for t in TICKER_PATHS if t in loaded_frames]
    wide_1s = combine_wide(frames)

    # Optionally load US30 proxy and join
    if US30_PATH and Path(US30_PATH).exists():
        try:
            us30 = load_dbn_ohlcv_1s(US30_PATH, US30_TICKER)
            wide_1s = wide_1s.join(us30, how="outer")
            print(f"Joined {US30_TICKER}: {len(us30):,} rows")
        except Exception as e:
            print(f"[WARN] Could not load {US30_TICKER}: {e}")

    # Align to complete 1-second grid (no forward fills on OHLC to avoid artificial candles)
    wide_1s = align_to_one_second(wide_1s)

    # Add 1-second returns on closes (optional)
    wide_1s = add_returns(wide_1s, periods=(1,))

    # Persist outputs
    out_parquet = OUT_DIR / "dow_top5_ohlcv_1s.parquet"
    out_csv = OUT_DIR / "dow_top5_ohlcv_1s.csv"

    # Use Parquet for efficient storage
    wide_1s.to_parquet(out_parquet)
    # Also dump a CSV for quick inspection (can be large!)
    wide_1s.to_csv(out_csv)

    print("\n=== 1-second wide frame ===")
    print(wide_1s.iloc[:5])
    print("\nSaved:")
    print(f"  {out_parquet}")
    print(f"  {out_csv}")

    # Optional: 1-minute resample for faster plotting/analysis
    wide_1m = resample_to_1min_ohlcv(wide_1s)
    out_parquet_1m = OUT_DIR / "dow_top5_ohlcv_1m.parquet"
    wide_1m.to_parquet(out_parquet_1m)
    print(f"  {out_parquet_1m}")


# =============================
# Plot section (auto-runs after saving Parquet)
# =============================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_instruments_vs_time(parquet_path: Path, resample_rule: str = "1min", normalize: bool = True):
    """
    Load the saved parquet and plot each instrument's close price vs time.
    - resample_rule: e.g. '1min', '5min', '' for raw 1s
    - normalize: if True, rebases each series to 100 at its first point
    """
    if not parquet_path.exists():
        print(f"[WARN] Parquet not found: {parquet_path}")
        return

    print(f"\n=== Plotting instruments vs time from {parquet_path.name} ===")
    df = pd.read_parquet(parquet_path)

    # Pick close columns
    close_cols = [c for c in df.columns if c.endswith("_close")]
    if not close_cols:
        print("[WARN] No *_close columns found, skipping plot.")
        return

    closes = df[close_cols].copy()
    closes.columns = [c.replace("_close", "") for c in closes.columns]

    # Ensure datetime index and resample if needed
    if not isinstance(closes.index, pd.DatetimeIndex):
        closes.index = pd.to_datetime(closes.index, utc=True)
    elif closes.index.tz is None:
        closes.index = closes.index.tz_localize("UTC")

    if resample_rule:
        closes = closes.resample(resample_rule).last()

    # Normalize (index all to 100)
    if normalize:
        first_vals = closes.apply(lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA)
        closes = (closes / first_vals) * 100

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in closes.columns:
        ax.plot(closes.index, closes[col], label=col)

    ax.set_title(
        f"Dow Top 5 Instruments vs Time  •  {resample_rule or '1s'}"
        + ("  •  Normalized to 100" if normalize else "")
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Indexed to 100" if normalize else "Price")
    ax.legend(ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()

    out_png = parquet_path.with_suffix("").with_name(parquet_path.stem + "_plot.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✅ Plot saved to {out_png}")


# Auto-run plot after saving Parquet
try:
    plot_instruments_vs_time(out_parquet, resample_rule="1min", normalize=True)
except Exception as e:
    print(f"[WARN] Plotting failed: {e}")
