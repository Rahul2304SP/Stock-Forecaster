import databento as db

MSFT = "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.MSFT.dbn"
GS = "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.GS.dbn"
AXP = "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.AXP.dbn"
HD = "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.HD.dbn"
CAT = "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.CAT.dbn"
US30 = "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/DJ30/xnas-itch-20241031-20251031.ohlcv-1s.dbn"

MSFTDBN = db.DBNStore.from_file(MSFT)
GSDBN = db.DBNStore.from_file(GS)
AXPDBN = db.DBNStore.from_file(AXP)
HDDBN = db.DBNStore.from_file(HD)
CATDBN = db.DBNStore.from_file(CAT)
US30DBN = db.DBNStore.from_file(US30)


df = MSFTDBN.to_df()

print(df.head)
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import databento as db

# =============================
# Configuration: file paths
# =============================
# Update these paths as needed. Keys must be the tickers you want in the output.
TICKER_PATHS: Dict[str, str] = {
    "MSFT": "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.MSFT.dbn",
    "GS":   "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.GS.dbn",
    "AXP":  "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.AXP.dbn",
    "HD":   "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.HD.dbn",
    "CAT":  "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/XNAS-20251101-RDYVM6QWH5/xnas-itch-20241031-20251030.ohlcv-1s.CAT.dbn",
}

# Optional index proxy (Dow). If present, it will be joined into the wide frame as well.
US30_PATH = "/Users/rahulparmeshwar/Documents/Algo Bots/Data/Dow Top 5/DBN File/DJ30/xnas-itch-20241031-20251031.ohlcv-1s.dbn"
US30_TICKER = "US30"

# Output locations
OUT_DIR = Path("/Users/rahulparmeshwar/Documents/DJ30Levels/outputs")
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


def combine_wide(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join a list of per-ticker frames on timestamp index."""
    if not frames:
        return pd.DataFrame()
    base = frames[0]
    for f in frames[1:]:
        base = base.join(f, how="outer")
    return base.sort_index()


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

    # Load components
    frames: List[pd.DataFrame] = []
    for ticker, path in TICKER_PATHS.items():
        if not Path(path).exists():
            print(f"[WARN] File not found for {ticker}: {path}")
            continue
        try:
            f = load_dbn_ohlcv_1s(path, ticker)
            frames.append(f)
            print(f"Loaded {ticker}: {len(f):,} rows from {f.index.min()} to {f.index.max()}")
        except Exception as e:
            print(f"[ERROR] Failed loading {ticker}: {e}")

    if not frames:
        raise SystemExit("No component data loaded; aborting.")

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



import matplotlib.pyplot as plt

# =============================================
# Multi-series plotting from saved Parquet files
# =============================================
import matplotlib.dates as mdates

def _pick_present_close_cols(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    cols = []
    for t in tickers:
        c = f"{t}_close"
        if c in df.columns:
            cols.append(c)
        else:
            print(f"[WARN] {c} not found in parquet; skipping")
    if not cols:
        raise KeyError("No requested _close columns found in parquet file")
    return cols


def plot_multi_series(parquet_path: Path, tickers: list[str], normalize: bool = True,
                      ffill_limit: int | None = 2, title_suffix: str = "(1m)") -> None:
    """Load resampled (1m) parquet and plot selected tickers' close series aligned by time.
    - normalize: rebases each series to 100 at the first common timestamp for visual comparability
    - ffill_limit: small forward-fill to bridge tiny gaps due to sparse seconds within each minute
    """
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    close_cols = _pick_present_close_cols(df, tickers)

    # Align on common timeline; drop rows where all requested are NaN
    sub = df[close_cols].copy()
    sub = sub.dropna(how="all")

    # Small forward-fill to make lines continuous without inventing large gaps
    if ffill_limit is not None and ffill_limit > 0:
        sub = sub.ffill(limit=ffill_limit)

    # Normalize
    if normalize:
        first_valid = sub.apply(lambda s: s.dropna().iloc[0])
        sub = (sub / first_valid) * 100.0
        ylabel = "Rebased to 100"
    else:
        ylabel = "Price"

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in close_cols:
        label = col.replace("_close", "")
        ax.plot(sub.index, sub[col], label=label)

    ax.set_title(f"Multi-series Close {title_suffix}: " + ", ".join([c.replace("_close","") for c in close_cols]))
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3, frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()

    # Save a copy alongside parquet for quick viewing
    png_path = parquet_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot -> {png_path}")


# Auto-plot from the 1-minute parquet if available
if __name__ == "__main__":
    try:
        default_parquet_1m = OUT_DIR / "dow_top5_ohlcv_1m.parquet"
        # Edit the list below to choose which instruments to show.
        PLOT_TICKERS = ["GS", "MSFT", "AXP", "HD", "CAT", "US30"]
        plot_multi_series(default_parquet_1m, PLOT_TICKERS, normalize=True, ffill_limit=2, title_suffix="(1m)")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")
