"""
Quantitative Performance Metrics Module for CrowdWisdomTrading.
Calculates institutional hedge fund metrics: Sharpe Ratio, Sortino Ratio,
Maximum Drawdown, Calmar Ratio, Win Rate, and Multi-Horizon Directional Accuracy.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class QuantMetricsCalculator:
    """Calculates risk-adjusted returns and forecast evaluation metrics."""

    def __init__(self, periods_per_year: int = 24 * 365, risk_free_rate: float = 0.04):
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        # Per-period risk-free return
        self.rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0

    def calculate_strategy_metrics(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Computes Sharpe, Sortino, Max Drawdown, Calmar, and Win Rate.
        """
        rets = np.asarray(strategy_returns, dtype=float)
        rets = rets[~np.isnan(rets)]

        if len(rets) == 0:
            return self._empty_metrics()

        n_periods = len(rets)
        cum_ret = float(np.prod(1.0 + rets) - 1.0)
        ann_factor = self.periods_per_year / n_periods if n_periods > 0 else 1.0
        ann_return = float((1.0 + cum_ret) ** ann_factor - 1.0) if cum_ret > -1.0 else -1.0

        # Mean excess return
        excess_rets = rets - self.rf_per_period
        mean_excess = float(np.mean(excess_rets))
        vol = float(np.std(rets, ddof=1)) if n_periods > 1 else 0.0
        ann_vol = vol * math.sqrt(self.periods_per_year)

        # 1. Sharpe Ratio
        sharpe = (mean_excess * self.periods_per_year) / (ann_vol + 1e-9) if ann_vol > 0 else 0.0

        # 2. Downside Volatility & Sortino Ratio
        downside_rets = np.minimum(excess_rets, 0.0)
        downside_vol = float(np.sqrt(np.mean(downside_rets ** 2))) * math.sqrt(self.periods_per_year)
        sortino = (mean_excess * self.periods_per_year) / (downside_vol + 1e-9) if downside_vol > 0 else 0.0

        # 3. Maximum Drawdown & Drawdown Duration
        equity_curve = np.cumprod(1.0 + rets)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        max_drawdown_pct = abs(max_drawdown) * 100.0

        # 4. Calmar Ratio
        calmar = (ann_return / (abs(max_drawdown) + 1e-9)) if abs(max_drawdown) > 0 else 0.0

        # 5. Win Rate and Profit Factor
        winning_trades = rets[rets > 0]
        losing_trades = rets[rets < 0]
        win_rate = float(len(winning_trades) / len(rets)) if len(rets) > 0 else 0.0

        sum_gains = float(np.sum(winning_trades)) if len(winning_trades) > 0 else 0.0
        sum_losses = float(np.abs(np.sum(losing_trades))) if len(losing_trades) > 0 else 0.0
        profit_factor = (sum_gains / sum_losses) if sum_losses > 0 else (10.0 if sum_gains > 0 else 1.0)

        # Benchmark comparison
        bench_cum_ret = 0.0
        bench_sharpe = 0.0
        if benchmark_returns is not None:
            b_rets = np.asarray(benchmark_returns, dtype=float)[:len(rets)]
            bench_cum_ret = float(np.prod(1.0 + b_rets) - 1.0)
            b_excess = b_rets - self.rf_per_period
            b_vol = float(np.std(b_rets, ddof=1)) * math.sqrt(self.periods_per_year)
            bench_sharpe = (float(np.mean(b_excess)) * self.periods_per_year) / (b_vol + 1e-9) if b_vol > 0 else 0.0

        return {
            "cumulative_return_pct": round(cum_ret * 100.0, 2),
            "annualized_return_pct": round(ann_return * 100.0, 2),
            "annualized_volatility_pct": round(ann_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "calmar_ratio": round(calmar, 3),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "profit_factor": round(profit_factor, 2),
            "benchmark_cumulative_pct": round(bench_cum_ret * 100.0, 2),
            "benchmark_sharpe": round(bench_sharpe, 3)
        }

    def calculate_horizon_forecasting_metrics(
        self,
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates forecast accuracy metrics for a specific prediction horizon.
        """
        y_pred = np.asarray(predicted_returns, dtype=float)
        y_true = np.asarray(actual_returns, dtype=float)

        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_p, y_t = y_pred[mask], y_true[mask]

        if len(y_p) == 0:
            return {"rmse": 0.0, "mae": 0.0, "directional_accuracy_pct": 50.0, "ic": 0.0}

        rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
        mae = float(np.mean(np.abs(y_p - y_t)))

        # Directional Accuracy (Hit Rate)
        dir_hits = (np.sign(y_p) == np.sign(y_t))
        dir_acc = float(np.mean(dir_hits)) * 100.0

        # Information Coefficient (Spearman / Pearson Rank Correlation)
        if len(y_p) > 2 and np.std(y_p) > 0 and np.std(y_t) > 0:
            ic = float(np.corrcoef(y_p, y_t)[0, 1])
            if np.isnan(ic):
                ic = 0.0
        else:
            ic = 0.0

        return {
            "rmse": round(rmse, 5),
            "mae": round(mae, 5),
            "directional_accuracy_pct": round(dir_acc, 2),
            "information_coefficient": round(ic, 4)
        }

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            "cumulative_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "benchmark_cumulative_pct": 0.0,
            "benchmark_sharpe": 0.0
        }

