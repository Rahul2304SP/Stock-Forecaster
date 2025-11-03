from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARQUET_PATH = Path(
    "C:/Users/Rahul Parmeshwar/Documents/GitHub/Stock-Forecaster/Outputs/Dow30Levels/dow_top5_ohlcv_1s.parquet"
)
OUT_DIR = PARQUET_PATH.parent
PREVIEW_CSV = OUT_DIR / "dow_top5_ohlcv_1s_preview100k.csv"
RETURNS_CSV = OUT_DIR / "dow_top5_ohlcv_1s_returns.csv"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FFT_DIR = OUT_DIR / "fft"
FFT_DIR.mkdir(parents=True, exist_ok=True)
RETURNS_5MIN_CSV = OUT_DIR / "dow_top5_ohlcv_5min_returns.csv"


def compute_fft(series: pd.Series, sample_spacing: float = 1.0) -> pd.DataFrame:
    """Return FFT frequency/amplitude pairs for the provided series."""
    clean = series.dropna()
    if clean.size < 2:
        return pd.DataFrame(columns=["frequency_hz", "amplitude"])

    values = clean.to_numpy(dtype=float)
    values = values - values.mean()
    fft_vals = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(len(values), d=sample_spacing)
    amplitude = np.abs(fft_vals)
    return pd.DataFrame({"frequency_hz": freqs, "amplitude": amplitude})

if __name__ == "__main__":
    df = pd.read_parquet(PARQUET_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()

    df.head(100_000).to_csv(PREVIEW_CSV)
    print(f"Saved preview CSV to {PREVIEW_CSV}")

    close_cols = [c for c in df.columns if c.endswith("_close")]
    volume_cols = [c for c in df.columns if c.endswith("_volume")]
    if not close_cols:
        raise SystemExit("No *_close columns found in the parquet file.")

    # Drop rows that are entirely missing and forward-fill partial gaps.
    relevant_cols = close_cols + volume_cols
    filtered = df[relevant_cols].copy()
    filtered = filtered.dropna(how="all")
    filtered = filtered.ffill()
    filtered = filtered.dropna()
    if filtered.empty:
        raise SystemExit("No rows remain after cleaning close/volume columns.")

    closes = filtered[close_cols]
    volumes = filtered[volume_cols]

    returns = closes.pct_change().dropna()
    returns.columns = [f"{col}_ret1s" for col in close_cols]

    volume_pct = volumes.pct_change()
    volume_pct = volume_pct.loc[returns.index].fillna(0.0)
    volume_pct = volume_pct.rename(
        columns={col: f"{col.replace('_volume', '')}_volume_ret1s" for col in volume_pct.columns}
    )

    output = pd.concat([returns, volume_pct], axis=1)
    output.to_csv(RETURNS_CSV, index=True)
    print(f"Saved returns CSV to {RETURNS_CSV}")

    # Resample to 5-minute frequency using the mean of 1-second changes.
    output_5min = output.resample("5min").mean()
    output_5min.to_csv(RETURNS_5MIN_CSV, index=True)
    print(f"Saved 5-minute resampled returns to {RETURNS_5MIN_CSV}")

    # Infer sample spacing (seconds) for FFT calculations.
    sample_spacing = 1.0
    if isinstance(output.index, pd.DatetimeIndex) and len(output.index) > 1:
        diffs = output.index.to_series().diff().dt.total_seconds().dropna()
        if not diffs.empty and diffs.median() > 0:
            sample_spacing = float(diffs.median())

    # Save FFT results for price and volume changes.
    for col in returns.columns:
        ticker = col.replace("_close_ret1s", "")
        fft_df = compute_fft(returns[col], sample_spacing=sample_spacing)
        if fft_df.empty:
            continue
        fft_path = FFT_DIR / f"{ticker}_price_fft.csv"
        fft_df.to_csv(fft_path, index=False)
        print(f"Saved FFT for {ticker} price returns to {fft_path}")

    for col in volume_pct.columns:
        ticker = col.replace("_volume_ret1s", "")
        fft_df = compute_fft(volume_pct[col], sample_spacing=sample_spacing)
        if fft_df.empty:
            continue
        fft_path = FFT_DIR / f"{ticker}_volume_fft.csv"
        fft_df.to_csv(fft_path, index=False)
        print(f"Saved FFT for {ticker} volume changes to {fft_path}")

    for col in returns.columns:
        ticker = col.replace("_close_ret1s", "")
        fig, ax = plt.subplots(figsize=(12, 5))
        returns[col].plot(ax=ax, lw=0.8)
        ax.set_title(f"{ticker} 1-second Returns")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Return")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = PLOTS_DIR / f"{ticker}_returns.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {ticker} plot to {out_path}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in returns.columns:
        ticker = col.replace("_close_ret1s", "")
        returns[col].plot(ax=ax, lw=0.8, label=ticker)
    ax.set_title("1-second Returns (All Instruments)")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Return")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    combined_path = PLOTS_DIR / "all_returns.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined plot to {combined_path}")
