from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.express as px
except ImportError:
    px = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

PARQUET_PATH = Path(
    "C:/Users/Rahul Parmeshwar/Documents/GitHub/Stock-Forecaster/Outputs/Dow30Levels/dow_top5_ohlcv_1s.parquet"
)

OUT_DIR = PARQUET_PATH.parent / "insights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SPREAD_DIR = OUT_DIR / "spreads"
SPREAD_DIR.mkdir(parents=True, exist_ok=True)

VOLUME_PRICE_SUMMARY = OUT_DIR / "volume_price_summary.csv"
VOLUME_PRICE_SPIKES_DIR = OUT_DIR / "volume_price_spikes"
VOLUME_PRICE_SPIKES_DIR.mkdir(parents=True, exist_ok=True)

INTRADAY_STATS_FILE = OUT_DIR / "intraday_move_statistics.csv"
INTRADAY_TOP_INTERVALS_FILE = OUT_DIR / "intraday_top_intervals.csv"

DOW_IMPACT_COEFFS = OUT_DIR / "dow_regression_coefficients.csv"
DOW_IMPACT_METRICS = OUT_DIR / "dow_regression_diagnostics.csv"
DOW_IMPACT_CORRELATIONS = OUT_DIR / "dow_correlations.csv"
DOW_SPREAD_SERIES = OUT_DIR / "dow_spread_series.csv"
DOW_SPREAD_STATS = OUT_DIR / "dow_spread_statistics.csv"
DOW_SPREAD_PLOT = PLOTS_DIR / "US30_spread_zscore.png"

INTERACTIVE_RETURNS_3D_PLOT = OUT_DIR / "returns_3d.html"
INTERACTIVE_ZSCORE_3D_PLOT = OUT_DIR / "zscore_3d.html"
SPEED_DURATION_DIR = OUT_DIR / "speed_duration"
SPEED_DURATION_DIR.mkdir(parents=True, exist_ok=True)
SPEED_DURATION_SUMMARY = SPEED_DURATION_DIR / "speed_duration_summary.csv"
CYCLICALITY_DIR = OUT_DIR / "cyclicality"
CYCLICALITY_DIR.mkdir(parents=True, exist_ok=True)
CYCLICALITY_SUMMARY = CYCLICALITY_DIR / "cyclicality_summary.csv"

TICKERS: Tuple[str, ...] = ("MSFT", "HD", "GS", "AXP", "CAT")
US30_TICKER = "US30"
RTH_START = time(9, 30)
RTH_END = time(16, 0)


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a DataFrame to CSV, tolerating permission errors."""
    try:
        df.to_csv(path, **kwargs)
    except PermissionError:
        print(f"[WARN] Permission denied writing {path}; file may be open elsewhere. Skipping.")


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load the parquet file and ensure a UTC DatetimeIndex."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def compute_returns_and_volume(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return per-ticker frames containing closes, volumes, and pct changes."""
    results: Dict[str, pd.DataFrame] = {}
    for ticker in TICKERS + (US30_TICKER,):
        close_col = f"{ticker}_close"
        vol_col = f"{ticker}_volume"
        if close_col not in df.columns or vol_col not in df.columns:
            continue

        frame = df[[close_col, vol_col]].copy()
        frame = frame.dropna(how="all").ffill().dropna()
        if frame.empty:
            continue

        closes = frame[close_col]
        volumes = frame[vol_col]
        returns = closes.pct_change()
        volume_ret = volumes.pct_change()

        tick_df = pd.DataFrame(
            {
                "close": closes,
                "volume": volumes,
                "return": returns,
                "volume_ret": volume_ret,
            }
        ).dropna()
        if tick_df.empty:
            continue

        results[ticker] = tick_df
    return results


def analyze_volume_price(returns_map: Dict[str, pd.DataFrame]) -> None:
    """Quantify relationship between price moves and volume changes, per ticker."""
    summary_rows: List[Dict[str, float]] = []

    for ticker, frame in returns_map.items():
        price_ret = frame["return"]
        vol_ret = frame["volume_ret"]
        if price_ret.empty or vol_ret.empty:
            continue

        corr = price_ret.corr(vol_ret)
        corr_abs = price_ret.abs().corr(vol_ret.abs())
        joint = pd.concat([price_ret, vol_ret], axis=1, keys=["return", "volume_ret"]).dropna()

        ret_threshold = joint["return"].abs().quantile(0.99)
        vol_threshold = joint["volume_ret"].abs().quantile(0.99)
        spikes = joint[
            (joint["return"].abs() >= ret_threshold) & (joint["volume_ret"].abs() >= vol_threshold)
        ]

        summary_rows.append(
            {
                "ticker": ticker,
                "corr_return_volume": corr,
                "corr_abs_return_volume": corr_abs,
                "ret_threshold_99pct": ret_threshold,
                "vol_threshold_99pct": vol_threshold,
                "joint_spike_count": len(spikes),
            }
        )

        spike_path = VOLUME_PRICE_SPIKES_DIR / f"{ticker}_spikes.csv"
        safe_to_csv(spikes, spike_path, index=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            joint["volume_ret"],
            joint["return"],
            s=6,
            alpha=0.2,
            edgecolor="none",
        )
        if not spikes.empty:
            ax.scatter(
                spikes["volume_ret"],
                spikes["return"],
                s=12,
                alpha=0.6,
                color="crimson",
                label="Joint spikes",
            )
            ax.legend()
        ax.set_title(f"{ticker} Price vs Volume Change (1s)")
        ax.set_xlabel("Volume % change")
        ax.set_ylabel("Price % change")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{ticker}_volume_vs_price.png", dpi=150)
        plt.close(fig)

    if summary_rows:
        safe_to_csv(pd.DataFrame(summary_rows), VOLUME_PRICE_SUMMARY, index=False)


def analyze_intraday_patterns(
    returns_map: Dict[str, pd.DataFrame],
    tz: str = "US/Eastern",
    rth_start: time = RTH_START,
    rth_end: time = RTH_END,
) -> None:
    """Aggregate returns by time-of-day to highlight common move windows."""
    intraday_frames: List[pd.DataFrame] = []
    top_rows: List[pd.Series] = []

    for ticker, frame in returns_map.items():
        if frame.empty:
            continue
        series = frame["return"]
        if not isinstance(series.index, pd.DatetimeIndex):
            continue
        localized = series.copy()
        localized.index = localized.index.tz_convert(tz)
        per_minute = localized.resample("1min").mean().dropna()
        if per_minute.empty:
            continue

        per_minute = per_minute.between_time(
            rth_start.strftime("%H:%M"),
            rth_end.strftime("%H:%M"),
            inclusive="both",
        )
        if per_minute.empty:
            continue

        grouped = per_minute.groupby(per_minute.index.time)
        stats = pd.DataFrame(
            {
                "ticker": ticker,
                "mean_return": grouped.mean(),
                "median_return": grouped.median(),
                "std_return": grouped.std(),
                "prob_up": grouped.apply(lambda x: (x > 0).mean()),
                "avg_abs_return": grouped.apply(lambda x: x.abs().mean()),
                "count": grouped.size(),
            }
        )
        stats.index = [t.strftime("%H:%M") for t in stats.index]
        intraday_frames.append(stats)

        top = stats.nlargest(5, "avg_abs_return").assign(ticker=ticker)
        top_rows.extend(top.itertuples())

        fig, ax = plt.subplots(figsize=(10, 4))
        stats["avg_abs_return"].plot(ax=ax)
        ax.set_title(f"{ticker} Intraday Average Absolute Return ({tz})")
        ax.set_xlabel("Time of day")
        ax.set_ylabel("Average |return| (per min)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{ticker}_intraday_abs_return.png", dpi=150)
        plt.close(fig)

    if intraday_frames:
        combined = pd.concat(intraday_frames)
        safe_to_csv(combined, INTRADAY_STATS_FILE, index=True)

    if top_rows:
        top_df = pd.DataFrame(top_rows)
        safe_to_csv(top_df, INTRADAY_TOP_INTERVALS_FILE, index=False)


def analyze_spreads(returns_map: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """Model each instrument's returns as a linear combo of peers; return z-score frames."""

    def run_regression(target: str, features: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, float]] | None:
        if target not in returns_map:
            print(f"[WARN] Missing data for {target}; skipping spread analysis.")
            return None

        feature_list = [f for f in features if f in returns_map and f != target]
        if not feature_list:
            print(f"[WARN] No usable features for {target}; skipping.")
            return None

        frames = [returns_map[f]["return"].rename(f) for f in feature_list]
        frames.append(returns_map[target]["return"].rename(target))
        design = pd.concat(frames, axis=1).dropna()
        if design.empty:
            print(f"[WARN] No overlapping data for {target} regression.")
            return None

        X = design[feature_list].values
        y = design[target].values

        coef, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
        predictions = X @ coef
        residuals_array = y - predictions
        ss_res = np.sum(residuals_array**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        coeff_df = pd.DataFrame({"feature": feature_list, "coefficient": coef})
        metrics_df = pd.DataFrame(
            {
                "metric": ["r_squared", "rmse", "samples", "rank"],
                "value": [
                    r_squared,
                    np.sqrt(ss_res / len(y)),
                    len(y),
                    rank,
                ],
            }
        )
        corr_matrix = design.corr()

        spread = pd.Series(residuals_array, index=design.index, name="spread")
        spread_mean = spread.mean()
        spread_std = spread.std()
        if spread_std > 0:
            spread_z = (spread - spread_mean) / spread_std
        else:
            spread_z = pd.Series(data=np.nan, index=spread.index, name="zscore")
        spread_out = pd.concat([spread, spread_z.rename("zscore")], axis=1)

        stats: Dict[str, float] = {
            "mean": float(spread_mean),
            "std": float(spread_std),
            "median": float(spread.median()),
            "mad": float(np.mean(np.abs(spread - spread_mean))),
            "p95": float(spread.quantile(0.95)),
            "p05": float(spread.quantile(0.05)),
            "pct_outside_2sigma": float((spread_z.abs() >= 2).mean()),
        }
        stats["two_sigma_return"] = float(2.0 * spread_std)
        stats["two_sigma_percent"] = float(stats["two_sigma_return"] * 100.0)

        sample_spacing = 1.0
        if isinstance(spread.index, pd.DatetimeIndex) and len(spread.index) > 1:
            diffs = spread.index.to_series().diff().dt.total_seconds().dropna()
            if not diffs.empty:
                sample_spacing = float(diffs.median())

        spread_diff = spread.diff().dropna()
        spread_lag = spread.shift(1).dropna()
        stats["half_life_seconds"] = np.nan
        stats["half_life_minutes"] = np.nan
        if not spread_diff.empty and not spread_lag.empty:
            aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
            if not aligned.empty:
                y_delta = aligned.iloc[:, 0].values
                x_lag = aligned.iloc[:, 1].values
                A = np.column_stack([x_lag, np.ones_like(x_lag)])
                beta, intercept = np.linalg.lstsq(A, y_delta, rcond=None)[0]
                if beta < 0:
                    half_life_steps = -np.log(2) / beta
                    stats["half_life_seconds"] = float(half_life_steps * sample_spacing)
                    stats["half_life_minutes"] = float(stats["half_life_seconds"] / 60.0)

        series_path = SPREAD_DIR / f"{target}_spread_series.csv"
        stats_path = SPREAD_DIR / f"{target}_spread_stats.csv"
        coeff_path = SPREAD_DIR / f"{target}_spread_coefficients.csv"
        diag_path = SPREAD_DIR / f"{target}_spread_diagnostics.csv"
        corr_path = SPREAD_DIR / f"{target}_spread_correlations.csv"
        fit_plot_path = PLOTS_DIR / f"{target}_actual_vs_predicted.png"
        spread_plot_path = PLOTS_DIR / f"{target}_spread_zscore.png"

        safe_to_csv(spread_out, series_path, index=True)
        safe_to_csv(pd.DataFrame([stats]), stats_path, index=False)
        safe_to_csv(coeff_df, coeff_path, index=False)
        safe_to_csv(metrics_df, diag_path, index=False)
        safe_to_csv(corr_matrix, corr_path)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(design.index, y, label=f"{target} actual", linewidth=1)
        ax.plot(design.index, predictions, label=f"{target} predicted", linewidth=1, alpha=0.7)
        ax.set_title(f"{target} Returns vs Predicted Basket")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fit_plot_path, dpi=150)
        if target == US30_TICKER:
            fig.savefig(PLOTS_DIR / "US30_actual_vs_predicted.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(spread_out.index, spread_out["zscore"], label="Spread z-score", linewidth=1)
        ax.axhline(0, color="black", linewidth=1.0)
        band_label_pos = f"+2σ (~{stats['two_sigma_return']:.3%})"
        band_label_neg = f"-2σ (~{-stats['two_sigma_return']:.3%})"
        ax.axhline(2, color="crimson", linestyle="--", linewidth=2.5, label=band_label_pos)
        ax.axhline(-2, color="royalblue", linestyle="--", linewidth=2.5, label=band_label_neg)
        ax.set_title(f"{target} Spread Z-score")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Z-score")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(spread_plot_path, dpi=150)
        if target == US30_TICKER:
                fig.savefig(DOW_SPREAD_PLOT, dpi=150)
        plt.close(fig)

        if target == US30_TICKER:
            safe_to_csv(coeff_df, DOW_IMPACT_COEFFS, index=False)
            safe_to_csv(metrics_df, DOW_IMPACT_METRICS, index=False)
            safe_to_csv(corr_matrix, DOW_IMPACT_CORRELATIONS)
            safe_to_csv(spread_out, DOW_SPREAD_SERIES, index=True)
            safe_to_csv(pd.DataFrame([stats]), DOW_SPREAD_STATS, index=False)

        return spread_out, stats

    spread_map: Dict[str, pd.DataFrame] = {}
    stats_map: Dict[str, Dict[str, float]] = {}

    us30_result = run_regression(US30_TICKER, TICKERS)
    if us30_result is not None:
        spread_map[US30_TICKER], stats_map[US30_TICKER] = us30_result

    for ticker in TICKERS:
        feature_candidates = [t for t in TICKERS if t != ticker] + [US30_TICKER]
        result = run_regression(ticker, feature_candidates)
        if result is not None:
            spread_map[ticker], stats_map[ticker] = result

    return spread_map, stats_map


def analyze_speed_vs_duration(
    returns_map: Dict[str, pd.DataFrame],
    resample: str = "1min",
    min_abs_cum_return: float = 0.0005,
    max_scatter_points: int = 5000,
) -> None:
    """Measure how move speed relates to swing duration for each instrument."""
    summary_rows: List[Dict[str, float]] = []

    for ticker, frame in returns_map.items():
        series = frame["return"].dropna()
        if series.empty:
            continue

        sample_spacing = 1.0
        if resample:
            series = series.resample(resample).sum()
            series = series.dropna()
            if series.empty:
                continue
            sample_spacing = pd.to_timedelta(resample).total_seconds()
        elif isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 1:
            diffs = series.index.to_series().diff().dt.total_seconds().dropna()
            if not diffs.empty:
                sample_spacing = float(diffs.median())

        non_zero = series[series != 0]
        if non_zero.empty:
            continue

        sign_series = pd.Series(np.where(non_zero > 0, 1, -1), index=non_zero.index)
        groups = sign_series.ne(sign_series.shift()).cumsum()

        run_data = []
        for group_id, run_returns in non_zero.groupby(groups):
            run_sign = np.sign(run_returns.iloc[0])
            run_direction = "up" if run_sign > 0 else "down"

            duration_steps = len(run_returns)
            duration_sec = duration_steps * sample_spacing
            duration_min = duration_sec / 60.0
            cumulative_return = float(run_returns.sum())
            cumulative_abs = float(run_returns.abs().sum())
            avg_abs_return = float(run_returns.abs().mean())
            max_abs_return = float(run_returns.abs().max())

            if abs(cumulative_return) < min_abs_cum_return:
                continue

            run_data.append(
                {
                    "ticker": ticker,
                    "direction": run_direction,
                    "start": run_returns.index[0],
                    "end": run_returns.index[-1],
                    "duration_steps": duration_steps,
                    "duration_sec": duration_sec,
                    "duration_min": duration_min,
                    "cum_return": cumulative_return,
                    "cum_abs_return": cumulative_abs,
                    "avg_abs_return_per_step": avg_abs_return,
                    "avg_abs_return_per_min": avg_abs_return * (60.0 / sample_spacing),
                    "max_abs_return_per_step": max_abs_return,
                }
            )

        if not run_data:
            continue

        run_df = pd.DataFrame(run_data)
        corr_speed = run_df["avg_abs_return_per_step"].corr(run_df["duration_sec"])
        corr_total = run_df["cum_abs_return"].corr(run_df["duration_sec"])

        summary_rows.append(
            {
                "ticker": ticker,
                "sample_spacing_seconds": sample_spacing,
                "median_duration_sec": float(run_df["duration_sec"].median()),
                "median_duration_min": float(run_df["duration_min"].median()),
                "median_speed_per_step": float(run_df["avg_abs_return_per_step"].median()),
                "median_speed_per_min": float(run_df["avg_abs_return_per_min"].median()),
                "corr_speed_vs_duration": float(corr_speed) if not np.isnan(corr_speed) else np.nan,
                "corr_total_vs_duration": float(corr_total) if not np.isnan(corr_total) else np.nan,
                "max_duration_sec": float(run_df["duration_sec"].max()),
                "max_duration_min": float(run_df["duration_min"].max()),
                "runs_count": len(run_df),
            }
        )

        if len(run_df) > max_scatter_points:
            scatter_df = run_df.sample(max_scatter_points, random_state=42)
        else:
            scatter_df = run_df

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            scatter_df["avg_abs_return_per_step"],
            scatter_df["duration_sec"],
            s=12,
            alpha=0.4,
            edgecolor="none",
        )
        ax.set_title(f"{ticker} Speed vs Duration ({resample or 'raw'})")
        ax.set_xlabel(f"Avg |return| per {resample or 'step'}")
        ax.set_ylabel("Duration (seconds)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{ticker}_speed_vs_duration.png", dpi=150)
        plt.close(fig)

    if summary_rows:
        safe_to_csv(pd.DataFrame(summary_rows), SPEED_DURATION_SUMMARY, index=False)


def analyze_cyclicality(
    returns_map: Dict[str, pd.DataFrame],
    resample: str = "1min",
    top_n: int = 5,
) -> None:
    """Use FFT to highlight cyclical patterns in returns."""
    summary_rows: List[Dict[str, float]] = []

    for ticker, frame in returns_map.items():
        series = frame["return"]
        if series.empty:
            continue

        if resample:
            series = series.resample(resample).mean()
        series = series.dropna()
        if len(series) < 16:
            continue

        sample_spacing = 1.0
        if resample:
            try:
                sample_spacing = pd.to_timedelta(resample).total_seconds()
            except (ValueError, TypeError):
                pass
        elif isinstance(series.index, pd.DatetimeIndex) and len(series.index) > 1:
            diffs = series.index.to_series().diff().dt.total_seconds().dropna()
            if not diffs.empty:
                sample_spacing = float(diffs.median())

        values = series.to_numpy(dtype=float)
        values = values - values.mean()
        fft_vals = np.fft.rfft(values)
        freqs = np.fft.rfftfreq(len(values), d=sample_spacing)
        amplitudes = np.abs(fft_vals)

        mask = freqs > 0
        freqs = freqs[mask]
        amplitudes = amplitudes[mask]
        if amplitudes.size == 0:
            continue

        order = np.argsort(amplitudes)[::-1][:top_n]
        top_freqs = freqs[order]
        top_amps = amplitudes[order]

        period_seconds = 1.0 / top_freqs
        period_minutes = period_seconds / 60.0

        for rank, (freq, amp, per_sec, per_min) in enumerate(
            zip(top_freqs, top_amps, period_seconds, period_minutes), start=1
        ):
            summary_rows.append(
                {
                    "ticker": ticker,
                    "rank": rank,
                    "frequency_hz": float(freq),
                    "period_seconds": float(per_sec),
                    "period_minutes": float(per_min),
                    "amplitude": float(amp),
                    "sample_spacing_seconds": float(sample_spacing),
                    "resample": resample or "raw",
                }
            )

        spectrum_df = pd.DataFrame(
            {
                "frequency_hz": freqs,
                "period_minutes": 1.0 / freqs / 60.0,
                "amplitude": amplitudes,
            }
        )
        safe_to_csv(spectrum_df, CYCLICALITY_DIR / f"{ticker}_spectrum.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(spectrum_df["period_minutes"], spectrum_df["amplitude"])
        ax.set_xscale("log")
        ax.set_title(f"{ticker} Return Spectrum ({resample or 'raw'})")
        ax.set_xlabel("Period (minutes, log scale)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(CYCLICALITY_DIR / f"{ticker}_spectrum.png", dpi=150)
        plt.close(fig)

    if summary_rows:
        safe_to_csv(pd.DataFrame(summary_rows), CYCLICALITY_SUMMARY, index=False)


def create_interactive_returns_3d(
    returns_map: Dict[str, pd.DataFrame],
    resample: str | None = "1min",
) -> None:
    """Generate an interactive 3D scatter of returns vs time vs ticker."""
    if px is None:
        print("[WARN] Plotly is not installed; skipping 3D returns plot.")
        return

    frames: List[pd.DataFrame] = []
    for ticker, frame in returns_map.items():
        series = frame["return"]
        if series.empty:
            continue
        if resample:
            series = series.resample(resample).mean()
        series = series.dropna()
        if series.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": series.index,
                    "value": series.values,
                    "ticker": ticker,
                }
            )
        )

    if not frames:
        print("[WARN] No returns data available for 3D plot.")
        return

    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.scatter_3d(
        plot_df,
        x="timestamp",
        y="ticker",
        z="value",
        color="ticker",
        title=f"Instrument Returns ({resample or 'raw'} samples)",
        hover_data={
            "timestamp": "|%Y-%m-%d %H:%M:%S",
            "value": ":.4%",
            "ticker": True,
        },
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            zaxis_title="Return",
            xaxis_title="Timestamp",
            yaxis_title="Ticker",
        )
    )
    fig.write_html(str(INTERACTIVE_RETURNS_3D_PLOT))
    print(f"Saved interactive 3D plot to {INTERACTIVE_RETURNS_3D_PLOT}")


def create_interactive_zscore_3d(
    spread_map: Dict[str, pd.DataFrame],
    stats_map: Dict[str, Dict[str, float]],
    resample: str | None = "1min",
) -> None:
    """Generate an interactive 3D scatter of spread z-scores vs time vs ticker."""
    if px is None:
        print("[WARN] Plotly is not installed; skipping 3D z-score plot.")
        return

    frames: List[pd.DataFrame] = []
    for ticker, df in spread_map.items():
        series = df["zscore"]
        if series.empty:
            continue
        if resample:
            series = series.resample(resample).mean()
        series = series.dropna()
        if series.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": series.index,
                    "value": series.values,
                    "ticker": ticker,
                }
            )
        )

    if not frames:
        print("[WARN] No z-score data available for 3D plot.")
        return

    plot_df = pd.concat(frames, ignore_index=True)
    fig = px.scatter_3d(
        plot_df,
        x="timestamp",
        y="ticker",
        z="value",
        color="ticker",
        title=f"Spread Z-scores ({resample or 'raw'} samples)",
        hover_data={
            "timestamp": "|%Y-%m-%d %H:%M:%S",
            "value": ":.2f",
            "ticker": True,
        },
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            zaxis_title="Z-score",
            xaxis_title="Timestamp",
            yaxis_title="Ticker",
        )
    )
    if go is not None:
        plus_shown = False
        minus_shown = False
        for ticker in plot_df["ticker"].unique():
            subset = plot_df[plot_df["ticker"] == ticker]
            if subset.empty:
                continue
            t_min = subset["timestamp"].min()
            t_max = subset["timestamp"].max()
            stats = stats_map.get(ticker, {})
            two_sigma = stats.get("two_sigma_return")
            band_label = f"+2σ (~{two_sigma:.3%})" if two_sigma is not None else "+2σ band"
            fig.add_trace(
                go.Scatter3d(
                    x=[t_min, t_max],
                    y=[ticker, ticker],
                    z=[2.0, 2.0],
                    mode="lines",
                    line=dict(color="crimson", width=4),
                    name=band_label,
                    legendgroup="bands",
                    showlegend=not plus_shown,
                )
            )
            plus_shown = True
            neg_label = f"-2σ (~{-two_sigma:.3%})" if two_sigma is not None else "-2σ band"
            fig.add_trace(
                go.Scatter3d(
                    x=[t_min, t_max],
                    y=[ticker, ticker],
                    z=[-2.0, -2.0],
                    mode="lines",
                    line=dict(color="royalblue", width=4),
                    name=neg_label,
                    legendgroup="bands",
                    showlegend=not minus_shown,
                )
            )
            minus_shown = True
    fig.write_html(str(INTERACTIVE_ZSCORE_3D_PLOT))
    print(f"Saved interactive 3D z-score plot to {INTERACTIVE_ZSCORE_3D_PLOT}")


if __name__ == "__main__":
    df = load_and_prepare(PARQUET_PATH)
    returns_map = compute_returns_and_volume(df)

    analyze_volume_price(returns_map)
    analyze_intraday_patterns(returns_map)
    spread_map, stats_map = analyze_spreads(returns_map)
    analyze_speed_vs_duration(returns_map)
    analyze_cyclicality(returns_map)
    create_interactive_returns_3d(returns_map)
    create_interactive_zscore_3d(spread_map, stats_map)
