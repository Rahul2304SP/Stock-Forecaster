from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

INSIGHTS_DIR = Path("Outputs/Dow30Levels/insights")
PDF_PATH = INSIGHTS_DIR / "insights_file_guide.pdf"


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def fmt_pct(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def fmt_num(value: float, decimals: int = 4) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt_minutes(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f} minutes"


def add_image(story: List, styles, relative_path: str, caption: str, max_width: float = 6.0 * inch, max_height: float = 4.0 * inch) -> None:
    path = INSIGHTS_DIR / relative_path
    if not path.exists():
        return
    with PILImage.open(path) as pil:
        width_px, height_px = pil.size
    scale = min(max_width / width_px, max_height / height_px)
    img = Image(str(path), width=width_px * scale, height=height_px * scale)
    story.append(img)
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(caption, styles["Caption"]))
    story.append(Spacer(1, 0.12 * inch))


def build_document() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1Tight", parent=styles["Heading1"], spaceAfter=12))
    styles.add(ParagraphStyle(name="Heading2Tight", parent=styles["Heading2"], spaceAfter=6))
    styles.add(ParagraphStyle(name="Heading3Tight", parent=styles["Heading3"], spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyLead", parent=styles["BodyText"], leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=9, italic=True, spaceAfter=10))

    story: List = []

    # Preload datasets and per-ticker slices for later narrative sections.
    volume_summary = read_csv(INSIGHTS_DIR / "volume_price_summary.csv")
    volume_summary_dict: Dict[str, Dict[str, float]] = {}
    if volume_summary is not None:
        for row in volume_summary.itertuples():
            row_dict = row._asdict()
            volume_summary_dict[row.ticker] = row_dict

    spike_tables: Dict[str, pd.DataFrame] = {}
    spike_stats: Dict[str, Dict[str, float]] = {}
    spikes_dir = INSIGHTS_DIR / "volume_price_spikes"
    if spikes_dir.exists():
        for csv_path in sorted(spikes_dir.glob("*.csv")):
            ticker = csv_path.stem.replace("_spikes", "").upper()
            df = read_csv(csv_path)
            if df is None or df.empty:
                continue
            df = df.rename(columns={"Unnamed: 0": "timestamp"}).copy()
            df["abs_return"] = df["return"].abs()
            spike_tables[ticker] = df
            pos = df[df["return"] > 0]
            neg = df[df["return"] < 0]
            spike_stats[ticker] = {
                "total": len(df),
                "positive": len(pos),
                "negative": len(neg),
                "avg_pos_return": pos["return"].mean() if not pos.empty else float("nan"),
                "avg_neg_return": neg["return"].mean() if not neg.empty else float("nan"),
                "avg_pos_volume": pos["volume_ret"].mean() if not pos.empty else float("nan"),
                "avg_neg_volume": neg["volume_ret"].mean() if not neg.empty else float("nan"),
                "largest_abs_return": df.loc[df["abs_return"].idxmax()].to_dict(),
            }

    intraday_stats = read_csv(INSIGHTS_DIR / "intraday_move_statistics.csv")
    if intraday_stats is not None:
        intraday_stats = intraday_stats.rename(columns={"Unnamed: 0": "minute"})
    intraday_by_ticker: Dict[str, pd.DataFrame] = {}
    if intraday_stats is not None:
        for ticker in sorted(intraday_stats["ticker"].unique()):
            intraday_by_ticker[ticker] = intraday_stats[intraday_stats["ticker"] == ticker].copy()

    intraday_top = read_csv(INSIGHTS_DIR / "intraday_top_intervals.csv")
    intraday_top_map: Dict[str, pd.DataFrame] = {}
    if intraday_top is not None:
        for ticker in sorted(intraday_top["ticker"].unique()):
            intraday_top_map[ticker] = intraday_top[intraday_top["ticker"] == ticker].copy()

    speed_summary = read_csv(INSIGHTS_DIR / "speed_duration" / "speed_duration_summary.csv")
    speed_summary_map: Dict[str, Dict[str, float]] = {}
    if speed_summary is not None:
        for row in speed_summary.itertuples():
            speed_summary_map[row.ticker] = row._asdict()

    cyc_summary = read_csv(INSIGHTS_DIR / "cyclicality" / "cyclicality_summary.csv")
    cyc_summary_map: Dict[str, pd.DataFrame] = {}
    if cyc_summary is not None:
        for ticker in sorted(cyc_summary["ticker"].unique()):
            cyc_summary_map[ticker] = cyc_summary[cyc_summary["ticker"] == ticker].copy()

    spread_stats_map: Dict[str, Dict[str, float]] = {}
    spread_coeff_map: Dict[str, pd.DataFrame] = {}
    spread_diag_map: Dict[str, pd.DataFrame] = {}
    spreads_dir = INSIGHTS_DIR / "spreads"
    if spreads_dir.exists():
        for stats_path in sorted(spreads_dir.glob("*_spread_stats.csv")):
            ticker = stats_path.stem.replace("_spread_stats", "").upper()
            df = read_csv(stats_path)
            if df is None or df.empty:
                continue
            spread_stats_map[ticker] = df.iloc[0].to_dict()
        for coeff_path in sorted(spreads_dir.glob("*_spread_coefficients.csv")):
            ticker = coeff_path.stem.replace("_spread_coefficients", "").upper()
            df = read_csv(coeff_path)
            if df is None:
                continue
            spread_coeff_map[ticker] = df
        for diag_path in sorted(spreads_dir.glob("*_spread_diagnostics.csv")):
            ticker = diag_path.stem.replace("_spread_diagnostics", "").upper()
            df = read_csv(diag_path)
            if df is None:
                continue
            spread_diag_map[ticker] = df

    dow_stats = read_csv(INSIGHTS_DIR / "dow_spread_statistics.csv")

    def add_heading(text: str, level: int = 1) -> None:
        style = styles["Heading1Tight"] if level == 1 else styles["Heading2Tight"] if level == 2 else styles["Heading3Tight"]
        story.append(Paragraph(text, style))

    def add_para(text: str) -> None:
        story.append(Paragraph(text, styles["BodyLead"]))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

    add_heading("Dow Top 5 Insights Output Guide", level=1)
    add_para(f"Generated on {now}.")
    add_para(
        "This guide explains every artifact created by MarketInsights.py. It introduces the market concepts used, shows how to read each file, "
        "and summarises the empirical results captured for Microsoft (MSFT), Home Depot (HD), Goldman Sachs (GS), American Express (AXP), "
        "Caterpillar (CAT), and the Dow Jones index (US30)."
    )
    story.append(PageBreak())

    # Concepts
    add_heading("Foundational Concepts", level=1)
    add_para("Simple return r_t = (P_t - P_{t-1}) / P_{t-1}. Positive values indicate a gain between two timestamps; negative values indicate a loss.")
    add_para("Volume percent change dV_t = (V_t - V_{t-1}) / V_{t-1}. Large spikes reveal abnormal participation which often validates breakouts.")
    add_para("Z-score z_t = (x_t - mu) / sigma. Observations with |z_t| greater than 2 are more than two standard deviations from normal and often signal mean-reversion opportunities.")
    add_para("Linear regression models an instrument as a weighted combination of explanatory instruments. The residual (actual minus predicted) isolates the portion of movement unexplained by the basket.")
    add_para("Fast Fourier Transform (FFT) decomposes a series into constituent frequencies. Peaks in the spectrum identify cycles such as recurring intraday rhythms.")
    story.append(PageBreak())

    # Volume-price diagnostics
    add_heading("Volume and Price Diagnostics", level=1)
    add_heading("Files: volume_price_summary.csv and volume_price_spikes/*.csv", level=2)
    add_para(
        "The summary table holds one row per instrument with price-volume correlations, 99th percentile thresholds for absolute moves, "
        "and counts of simultaneous spikes. Each spikes file lists the exact timestamps where both price and volume exceeded these thresholds."
    )
    add_para(
        "Use these outputs to pinpoint levels where price surges were backed by extraordinary participation. Those locations often form support or resistance zones."
    )
    add_image(
        story,
        styles,
        "plots/MSFT_volume_vs_price.png",
        "MSFT price change versus volume change scatter. Diagonal structure shows alignment between momentum and flows; highlighted outliers match rows in MSFT_spikes.csv.",
    )

    # Intraday behaviour
    add_heading("Intraday Behaviour Profiles", level=1)
    add_heading("Files: intraday_move_statistics.csv and intraday_top_intervals.csv", level=2)
    add_para(
        "The statistics file records minute-by-minute metrics over the 09:30 to 16:00 US/Eastern cash session: mean return, median, standard deviation, probability of an uptick, average absolute move, and sample count. "
        "The top intervals file isolates the five most volatile minutes per instrument."
    )
    add_image(
        story,
        styles,
        "plots/US30_intraday_abs_return.png",
        "US30 average absolute return by minute. Volatility clusters around the opening rush, lunchtime lull, and closing ramp.",
    )
    add_image(
        story,
        styles,
        "plots/MSFT_intraday_abs_return.png",
        "MSFT minute-level volatility pattern. Note the pronounced surge immediately after the open and persistence into the close.",
        max_height=3.0 * inch,
    )

    # Regression and spreads
    add_heading("Regression and Spread Analytics", level=1)
    add_heading("Files: dow_regression_coefficients.csv, dow_regression_diagnostics.csv, dow_correlations.csv", level=2)
    add_para(
        "These tables describe the US30 regression on its top five components. Coefficients quantify sensitivity to each stock, diagnostics report fit quality, and the correlation matrix shows how tightly the instruments co-move."
    )
    add_heading("Files: dow_spread_series.csv, dow_spread_statistics.csv, spreads/*", level=2)
    add_para(
        "Each instrument has a residual series, descriptive statistics, regression coefficients, diagnostic metrics, and an instrument-specific correlation matrix. "
        "The residual z-scores highlight dislocations between each instrument and its explanatory basket."
    )
    add_image(
        story,
        styles,
        "plots/US30_actual_vs_predicted.png",
        "US30 actual versus predicted 1-second returns. The close alignment indicates that the component basket explains most index variation.",
    )
    add_image(
        story,
        styles,
        "plots/US30_spread_zscore.png",
        "US30 spread z-score with bold +/- 2 sigma bands, labelled by the equivalent return (~0.033%). Breaches flag statistically rare divergences.",
    )

    # Speed vs duration
    add_heading("Trend Speed versus Duration", level=1)
    add_heading("File: speed_duration/speed_duration_summary.csv", level=2)
    add_para(
        "Runs are built from 1-minute returns aggregated by consecutive sign. The summary table includes median run length, median speed per minute, correlation between speed and duration, maximum run size, and sample counts. "
        "These statistics help set expectations for how long quick bursts versus slower trends usually persist."
    )
    add_image(
        story,
        styles,
        "plots/MSFT_speed_vs_duration.png",
        "MSFT run speed versus duration. Faster moves tend to terminate earlier than slower builds.",
    )

    # Cyclicality
    add_heading("Cyclical Signatures", level=1)
    add_heading("Files: cyclicality/cyclicality_summary.csv and cyclicality/*", level=2)
    add_para(
        "FFT spectra highlight dominant return cycles after resampling to 1-minute bars. Summary rows list the five strongest frequencies per instrument, while CSV and PNG spectrum files show the full distribution of amplitudes across periods."
    )
    add_image(
        story,
        styles,
        "cyclicality/MSFT_spectrum.png",
        "MSFT return spectrum on a log-scaled period axis. Peaks around the 2 to 3 minute band reveal consistent intraday oscillations.",
    )

    # Interactive grids
    add_heading("Interactive Dashboards", level=1)
    add_heading("returns_3d.html", level=2)
    add_para("3D Plotly chart of 1-minute returns by timestamp and ticker. Rotate and hover to inspect synchronous rallies or divergences.")
    add_heading("zscore_3d.html", level=2)
    add_para("Interactive z-score grid. Crimson and blue rails mark +/- 2 sigma boundaries for each instrument, annotated with the return equivalent.")

    # Detailed findings
    add_heading("Detailed Conclusions from Each Dataset", level=1)

    # Volume-price detailed narrative
    if volume_summary is not None:
        add_heading("Volume and Price Findings", level=2)
        for row in volume_summary.itertuples():
            ticker = row.ticker
            corr = fmt_num(row.corr_return_volume, 4)
            corr_abs = fmt_num(row.corr_abs_return_volume, 4)
            ret_thr = fmt_pct(row.ret_threshold_99pct, 2)
            vol_thr = fmt_num(row.vol_threshold_99pct, 3)
            spike_count = int(row.joint_spike_count)
            add_para(
                f"{ticker}: price-volume correlation = {corr}, absolute correlation = {corr_abs}. "
                f"The 99th percentile absolute return was {ret_thr} and the volume change threshold was {vol_thr}. "
                f"There were {spike_count} joint spikes where both measures exceeded these cutoffs, signalling significant liquidity-backed moves."
            )
        for ticker, df in spike_tables.items():
            largest = df.nlargest(3, "abs_return")
            add_para(
                f"{ticker} spike highlights: "
                + "; ".join(
                    f"{row.timestamp} UTC with return {fmt_pct(row['return'])} and volume change {fmt_num(row['volume_ret'], 3)}"
                    for _, row in largest.iterrows()
                )
                + ". These timestamps are prime candidates for reviewing order book snapshots or news catalysts."
            )

    # Intraday detailed narrative
    if intraday_stats is not None:
        add_heading("Intraday Patterns", level=2)
        for ticker, subset in intraday_by_ticker.items():
            if subset.empty:
                continue
            peak_row = subset.loc[subset["avg_abs_return"].idxmax()]
            calm_row = subset.loc[subset["avg_abs_return"].idxmin()]
            add_para(
                f"{ticker}: most volatile minute = {peak_row['minute']} (avg abs return {fmt_pct(peak_row['avg_abs_return'])}, "
                f"probability of uptick {fmt_pct(peak_row['prob_up'])}). Calmest minute = {calm_row['minute']} "
                f"(avg abs return {fmt_pct(calm_row['avg_abs_return'])}, probability of uptick {fmt_pct(calm_row['prob_up'])}). "
                "These values highlight when to expect activity versus consolidation."
            )
            top_direction = subset.loc[subset["mean_return"].idxmax()]
            bottom_direction = subset.loc[subset["mean_return"].idxmin()]
            add_para(
                f"The strongest positive drift occurred around {top_direction['minute']} with mean return {fmt_pct(top_direction['mean_return'], 4)}. "
                f"The most negative drift appeared at {bottom_direction['minute']} with mean return {fmt_pct(bottom_direction['mean_return'], 4)}."
            )
        if intraday_top_map:
            add_para("Highest-volatility minute snapshots per instrument:")
            for ticker, subset in intraday_top_map.items():
                subset = subset.sort_values("avg_abs_return", ascending=False)
                descriptions = [
                    f"{row.Index} (avg abs return {fmt_pct(row.avg_abs_return)}, prob up {fmt_pct(row.prob_up)}, samples {int(row.count)})"
                    for row in subset.itertuples()
                ]
                add_para(f"{ticker}: " + "; ".join(descriptions) + ".")

    # Regression/spread detailed narrative
    if spread_stats_map:
        add_heading("Regression Residual Insights", level=2)
        for ticker, stats in spread_stats_map.items():
            add_para(
                f"{ticker}: residual mean = {fmt_num(stats.get('mean', float('nan')), 6)}, standard deviation = {fmt_num(stats.get('std', float('nan')), 6)}. "
                f"+/-2 sigma corresponds to +/-{fmt_pct(stats.get('two_sigma_return', float('nan')), 3)}. "
                f"{fmt_pct(stats.get('pct_outside_2sigma', float('nan')), 2)} of samples breached that band. "
                f"Estimated half-life of mean reversion = {fmt_minutes(stats.get('half_life_minutes', float('nan')))}."
            )
        for ticker, df in spread_coeff_map.items():
            if df is None or df.empty:
                continue
            largest = df.iloc[df["coefficient"].abs().idxmax()]
            add_para(
                f"{ticker} regression weights: strongest influence from {largest['feature']} with beta {fmt_num(largest['coefficient'], 4)}. "
                "Monitor this relationship, as shocks in the leading contributor are most likely to spill over."
            )
        for ticker, df in spread_diag_map.items():
            if df is None or df.empty:
                continue
            diag = df.set_index("metric")["value"]
            r2 = diag.get("r_squared", float("nan"))
            rmse = diag.get("rmse", float("nan"))
            add_para(
                f"{ticker} regression diagnostics: R-squared = {fmt_num(r2, 4)}, RMSE = {fmt_num(rmse, 6)}, sample count = {fmt_num(diag.get('samples', float('nan')), 0)}. "
                "R-squared near one implies the basket explains most behaviour; lower values point to idiosyncratic drivers."
            )

    # Speed-duration detailed narrative
    if speed_summary_map:
        add_heading("Trend Persistence Observations", level=2)
        for ticker, row_dict in speed_summary_map.items():
            add_para(
                f"{ticker}: median run length {fmt_num(row_dict.get('median_duration_min', float('nan')), 2)} minutes, median speed per minute {fmt_pct(row_dict.get('median_speed_per_min', float('nan')), 3)}. "
                f"Maximum run lasted {fmt_num(row_dict.get('max_duration_min', float('nan')), 2)} minutes with {int(row_dict.get('runs_count', 0))} qualifying runs. "
                f"Correlation between speed and duration = {fmt_num(row_dict.get('corr_speed_vs_duration', float('nan')), 3)}; correlation between cumulative absolute move and duration = {fmt_num(row_dict.get('corr_total_vs_duration', float('nan')), 3)}. "
                "Negative speed-duration correlation suggests quick bursts mean-revert faster, while positive cumulative correlation indicates large swings require time to unfold."
            )

    # Cyclicality detailed narrative
    if cyc_summary_map:
        add_heading("Cyclical Behaviour Findings", level=2)
        for ticker, subset in cyc_summary_map.items():
            subset = subset.sort_values("rank")
            peaks = []
            for row in subset.itertuples():
                period_min = fmt_num(row.period_minutes, 2)
                amp = fmt_num(row.amplitude, 4)
                peaks.append(f"rank {row.rank} period {period_min} min amplitude {amp}")
            add_para(
                f"{ticker}: dominant FFT peaks -> " + "; ".join(peaks) + ". "
                "Consistent peaks indicate stable intraday oscillations that can be exploited with cycle-aware strategies."
            )

    # Dow summary for quick reference
    if dow_stats is not None and not dow_stats.empty:
        stats = dow_stats.iloc[0]
        add_heading("Dow Jones Residual Summary", level=2)
        add_para(
            f"US30 residual mean {fmt_num(stats['mean'], 6)}, standard deviation {fmt_num(stats['std'], 6)}, +/-2 sigma return {fmt_pct(stats['two_sigma_return'], 3)} "
            f"({fmt_pct(stats['two_sigma_percent'], 3)}). Only {fmt_pct(stats['pct_outside_2sigma'], 2)} of observations breached the +/-2 sigma band, "
            f"and estimated half-life of mean reversion is {fmt_minutes(stats['half_life_minutes'])}. These values calibrate statistical arbitrage triggers on the index."
        )

    # Ticker specific synthesis
    add_heading("Ticker Specific Syntheses", level=1)
    ordered_tickers = ["MSFT", "HD", "GS", "AXP", "CAT", "US30"]
    for ticker in ordered_tickers:
        add_heading(f"{ticker} Integrated Summary", level=2)
        vs = volume_summary_dict.get(ticker)
        if vs:
            add_para(
                f"Volume-price alignment: correlation {fmt_num(vs.get('corr_return_volume', float('nan')), 4)} "
                f"(absolute correlation {fmt_num(vs.get('corr_abs_return_volume', float('nan')), 4)}). "
                f"The 99th percentile thresholds were {fmt_pct(vs.get('ret_threshold_99pct', float('nan')), 2)} for absolute return and "
                f"{fmt_num(vs.get('vol_threshold_99pct', float('nan')), 3)} for volume change, yielding {int(vs.get('joint_spike_count', 0))} joint spikes."
            )
        spikes = spike_stats.get(ticker)
        if spikes:
            hi = spikes["largest_abs_return"]
            add_para(
                f"Spike profile: {spikes['total']} qualifying events with {spikes['positive']} positive and {spikes['negative']} negative moves. "
                f"Average positive return {fmt_pct(spikes.get('avg_pos_return', float('nan')), 3)} with mean volume change {fmt_num(spikes.get('avg_pos_volume', float('nan')), 3)}. "
                f"Average negative return {fmt_pct(spikes.get('avg_neg_return', float('nan')), 3)} with mean volume change {fmt_num(spikes.get('avg_neg_volume', float('nan')), 3)}. "
                f"The largest absolute spike occurred {hi.get('timestamp', 'N/A')} UTC with return {fmt_pct(hi.get('return', float('nan')), 3)} and volume change {fmt_num(hi.get('volume_ret', float('nan')), 3)}."
            )
        intraday_profile = intraday_by_ticker.get(ticker)
        if intraday_profile is not None and not intraday_profile.empty:
            mean_abs = fmt_pct(intraday_profile["avg_abs_return"].mean(), 3)
            mean_prob = fmt_pct(intraday_profile["prob_up"].mean(), 2)
            peak_row = intraday_profile.loc[intraday_profile["avg_abs_return"].idxmax()]
            calm_row = intraday_profile.loc[intraday_profile["avg_abs_return"].idxmin()]
            add_para(
                f"Intraday behaviour: average minute volatility {mean_abs} with mean probability of an uptick {mean_prob}. "
                f"Most turbulent minute {peak_row['minute']} (avg abs return {fmt_pct(peak_row['avg_abs_return'], 3)}), calmest minute {calm_row['minute']} "
                f"(avg abs return {fmt_pct(calm_row['avg_abs_return'], 3)})."
            )
            top_minutes = intraday_top_map.get(ticker)
            if top_minutes is not None and not top_minutes.empty:
                descriptions = [
                    f"{row.Index} ({fmt_pct(row.avg_abs_return, 3)}, prob up {fmt_pct(row.prob_up, 2)})"
                    for row in top_minutes.sort_values("avg_abs_return", ascending=False).itertuples()
                ]
                add_para(f"Concentrated volatility windows: " + "; ".join(descriptions) + ".")
        spread_stats = spread_stats_map.get(ticker)
        if spread_stats:
            add_para(
                f"Regression spread: standard deviation {fmt_num(spread_stats.get('std', float('nan')), 6)} with +/-2 sigma envelope "
                f"{fmt_pct(spread_stats.get('two_sigma_return', float('nan')), 3)}. "
                f"{fmt_pct(spread_stats.get('pct_outside_2sigma', float('nan')), 2)} of observations left this envelope. "
                f"Estimated mean-reversion half-life {fmt_minutes(spread_stats.get('half_life_minutes', float('nan')))}."
            )
        coeff_df = spread_coeff_map.get(ticker)
        if coeff_df is not None and not coeff_df.empty:
            sorted_coeffs = coeff_df.reindex(coeff_df["coefficient"].abs().sort_values(ascending=False).index)
            coeff_descriptions = [
                f"{row.feature} beta {fmt_num(row.coefficient, 4)}"
                for row in sorted_coeffs.itertuples()
            ]
            add_para("Regression contributors ranked by influence: " + "; ".join(coeff_descriptions) + ".")
        speed_stats = speed_summary_map.get(ticker)
        if speed_stats:
            add_para(
                f"Trend persistence: median run length {fmt_num(speed_stats.get('median_duration_min', float('nan')), 2)} minutes with median speed "
                f"{fmt_pct(speed_stats.get('median_speed_per_min', float('nan')), 3)} per minute and {int(speed_stats.get('runs_count', 0))} runs analysed."
            )
        cyc = cyc_summary_map.get(ticker)
        if cyc is not None and not cyc.empty:
            peaks = [
                f"{fmt_num(row.period_minutes, 2)} min (amp {fmt_num(row.amplitude, 4)})"
                for row in cyc.sort_values("rank").itertuples()
            ]
            add_para(
                "Return cyclicality: dominant periods " + "; ".join(peaks) + ". "
                "These cycles can inform timing filters for entry and exit."
            )

    # Closing guidance
    add_heading("Applying the Results", level=1)
    add_para(
        "Volume-price analytics point to structurally important price levels; intraday statistics show when volatility usually concentrates; "
        "regression spreads quantify cross-instrument dislocations; speed-duration summaries describe trend persistence; cyclical analysis surfaces repeatable rhythms. "
        "Combine these findings to build trading rules, monitor risk, and validate hypotheses with data-driven evidence."
    )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=LETTER,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(story)


if __name__ == "__main__":
    build_document()
