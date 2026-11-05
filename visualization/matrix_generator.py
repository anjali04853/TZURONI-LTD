"""
Visual Matrix Output Generator for CrowdWisdomTrading.
Generates rich multi-panel publication figures and interactive HTML dashboards
displaying X sentiment dynamics against price evolution, multi-horizon forecasts, and quant equity curves.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VisualMatrixGenerator:
    """Generates visual matrix charts of sentiment changes vs price over time."""

    def __init__(self, output_dir: Optional[Path] = None):
        from config.settings import settings
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_static_matrix(
        self,
        merged_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        asset_symbol: str = "BTC-USD",
        filename: str = "visual_matrix.png"
    ) -> Path:
        """
        Creates a 4-row institutional visual matrix chart saved as high-res PNG.
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=False, gridspec_kw={"height_ratios": [3, 2, 2, 2.5]})
        plt.subplots_adjust(hspace=0.28)

        # Style configuration
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig.patch.set_facecolor("#0e1117")
        for ax in axes:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=9)
            ax.grid(True, color="#30363d", linestyle="--", alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        # -------------------------------------------------------------
        # Panel 1: Price Action Over Time with Sentiment Shading
        # -------------------------------------------------------------
        ax1 = axes[0]
        dates = pd.to_datetime(merged_df["timestamp_utc"])
        close_p = merged_df["close"].values
        sent = merged_df["sentiment_weighted"].values

        ax1.plot(dates, close_p, color="#58a6ff", linewidth=1.8, label=f"{asset_symbol} Spot Price")
        ax1.set_title(f"CrowdWisdomTrading: X Sentiment vs Price Matrix — {asset_symbol}", color="#f0f6fc", fontsize=14, fontweight="bold", pad=12)
        ax1.set_ylabel("Price ($)", color="#f0f6fc", fontsize=11)

        # Highlight bullish / bearish crowd regimes
        ax1.fill_between(dates, close_p.min(), close_p.max(), where=(sent > 0.15), color="#238636", alpha=0.15, label="Bullish Crowd Regime")
        ax1.fill_between(dates, close_p.min(), close_p.max(), where=(sent < -0.15), color="#da3633", alpha=0.15, label="Bearish Crowd Regime")
        ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

        # -------------------------------------------------------------
        # Panel 2: Engagement-Weighted X Sentiment & 24h Trend
        # -------------------------------------------------------------
        ax2 = axes[1]
        sent_ma24 = pd.Series(sent).rolling(24, min_periods=1).mean()
        
        # Color positive sentiment green, negative red
        pos_mask = sent >= 0
        neg_mask = sent < 0
        ax2.bar(dates[pos_mask], sent[pos_mask], width=0.04, color="#3fb950", alpha=0.7, label="Positive Social Sentiment")
        ax2.bar(dates[neg_mask], sent[neg_mask], width=0.04, color="#f85149", alpha=0.7, label="Negative Social Sentiment")
        ax2.plot(dates, sent_ma24, color="#f1e05a", linewidth=1.5, label="24h Sentiment EWMA")
        ax2.axhline(0, color="#8b949e", linestyle=":", linewidth=0.8)
        ax2.set_ylabel("Weighted Sentiment ($S_w$)", color="#f0f6fc", fontsize=10)
        ax2.set_ylim(-1.0, 1.0)
        ax2.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

        # -------------------------------------------------------------
        # Panel 3: Post Volume & Viral Velocity Spikes
        # -------------------------------------------------------------
        ax3 = axes[2]
        post_vol = merged_df.get("post_volume", pd.Series(0, index=merged_df.index)).values
        ax3.plot(dates, post_vol, color="#a371f7", linewidth=1.2, label="Post Volume per Hour")
        ax3.fill_between(dates, 0, post_vol, color="#a371f7", alpha=0.25)
        ax3.set_ylabel("Hourly Post Volume", color="#f0f6fc", fontsize=10)
        ax3.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

        # -------------------------------------------------------------
        # Panel 4: Multi-Horizon Walk-Forward Strategy Performance
        # -------------------------------------------------------------
        ax4 = axes[3]
        if not oos_df.empty and "strategy_equity" in oos_df.columns:
            oos_dates = pd.to_datetime(oos_df["timestamp_utc"])
            strat_eq = oos_df["strategy_equity"].values
            bench_eq = oos_df["benchmark_equity"].values
            
            ax4.plot(oos_dates, strat_eq, color="#2ea043", linewidth=2.0, label="Walk-Forward Sentiment ML Strategy ($100k base)")
            ax4.plot(oos_dates, bench_eq, color="#8b949e", linewidth=1.4, linestyle="--", label="Buy & Hold Benchmark")
            ax4.set_ylabel("Portfolio Value ($)", color="#f0f6fc", fontsize=10)
            ax4.set_xlabel("Time (UTC)", color="#f0f6fc", fontsize=10)
            ax4.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
        else:
            ax4.text(0.5, 0.5, "Out-of-sample trading simulation running...", color="#8b949e", ha="center")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Visual Matrix generated at: {out_path}")
        return out_path

    def generate_horizon_fan_chart(
        self,
        oos_df: pd.DataFrame,
        sample_size: int = 120,
        filename: str = "horizon_forecast_fans.png"
    ) -> Path:
        """
        Visualizes predicted trajectories (n+0 to n+5) against actual realization.
        """
        sub_df = oos_df.tail(sample_size).copy().reset_index(drop=True)
        dates = pd.to_datetime(sub_df["timestamp_utc"])
        
        fig, ax = plt.subplots(figsize=(15, 7))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.grid(True, color="#30363d", linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        # Plot Actual Price
        ax.plot(dates, sub_df["close"], color="#ffffff", linewidth=2.2, label="Actual Realized Spot Close")

        # Plot Forecasts for n+1, n+3, n+5
        if "pred_price_n1" in sub_df.columns:
            ax.plot(dates, sub_df["pred_price_n1"], color="#58a6ff", linestyle="--", linewidth=1.2, label="Forecast Horizon n+1")
        if "pred_price_n3" in sub_df.columns:
            ax.plot(dates, sub_df["pred_price_n3"], color="#bc8cff", linestyle="--", linewidth=1.2, label="Forecast Horizon n+3")
        if "pred_price_n5" in sub_df.columns:
            ax.plot(dates, sub_df["pred_price_n5"], color="#f0883e", linestyle="--", linewidth=1.2, label="Forecast Horizon n+5")

        ax.set_title("Multi-Horizon Forecast Paths (n+1, n+3, n+5) vs Actual Spot Realization", color="#f0f6fc", fontsize=13, fontweight="bold")
        ax.set_ylabel("Price ($)", color="#f0f6fc")
        ax.set_xlabel("Time (UTC)", color="#f0f6fc")
        ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path

    def generate_interactive_html(
        self,
        merged_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        asset_symbol: str = "BTC-USD",
        filename: str = "visual_matrix.html"
    ) -> Path:
        """
        Builds a comprehensive interactive Plotly dashboard.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.35, 0.22, 0.18, 0.25],
            subplot_titles=[
                f"{asset_symbol} Spot Price & Crowd Regimes",
                "X Engagement-Weighted Sentiment ($S_w$)",
                "Hourly Post Volume",
                "Out-of-Sample Walk-Forward Strategy Cumulative Equity"
            ]
        )

        dates = pd.to_datetime(merged_df["timestamp_utc"])
        
        # Row 1: Price
        fig.add_trace(go.Scatter(x=dates, y=merged_df["close"], name="Spot Price", line=dict(color="#58a6ff", width=2)), row=1, col=1)

        # Row 2: Sentiment
        sent = merged_df["sentiment_weighted"]
        colors = np.where(sent >= 0, "#3fb950", "#f85149")
        fig.add_trace(go.Bar(x=dates, y=sent, name="Sentiment", marker=dict(color=colors)), row=2, col=1)

        # Row 3: Volume
        vol = merged_df.get("post_volume", pd.Series(0, index=merged_df.index))
        fig.add_trace(go.Bar(x=dates, y=vol, name="Volume", marker=dict(color="#a371f7")), row=3, col=1)

        # Row 4: Equity
        if not oos_df.empty and "strategy_equity" in oos_df.columns:
            oos_dates = pd.to_datetime(oos_df["timestamp_utc"])
            fig.add_trace(go.Scatter(x=oos_dates, y=oos_df["strategy_equity"], name="ML Sentiment Strategy", line=dict(color="#2ea043", width=2.5)), row=4, col=1)
            fig.add_trace(go.Scatter(x=oos_dates, y=oos_df["benchmark_equity"], name="Benchmark", line=dict(color="#8b949e", dash="dash")), row=4, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=1000,
            title_text=f"CrowdWisdomTrading: X Sentiment Price Prediction Agent Matrix ({asset_symbol})",
            showlegend=True,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22"
        )

        out_path = self.output_dir / filename
        fig.write_html(str(out_path))
        logger.info(f"Interactive dashboard generated at: {out_path}")
        return out_path

