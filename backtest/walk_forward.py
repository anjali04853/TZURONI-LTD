"""
Walk-Forward Validation Engine for CrowdWisdomTrading.
Performs rigorous rolling/expanding out-of-sample evaluation, preventing lookahead leakage,
and simulates realistic trading execution with commission and slippage friction.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from features.engineer import QuantFeatureEngineer
from models.multi_horizon import MultiHorizonForecaster
from models.rl_agent import RLHorizonTradingAgent
from backtest.metrics import QuantMetricsCalculator

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Executes walk-forward backtesting with purged embargo windows
    to validate multi-horizon price and return forecasts.
    """

    def __init__(
        self,
        algorithm: str = "lightgbm",
        train_window_bars: int = 1440,   # 60 days * 24h
        test_step_bars: int = 168,       # 7 days * 24h
        embargo_bars: int = 6,           # Equal to max forecast horizon n+5
        horizons: Optional[List[int]] = None,
        commission_bps: float = 5.0,
        slippage_bps: float = 2.0
    ):
        from config.settings import settings
        self.settings = settings
        self.algorithm = algorithm or settings.ALGORITHM
        self.train_window_bars = train_window_bars
        self.test_step_bars = test_step_bars
        self.embargo_bars = embargo_bars
        self.horizons = horizons if horizons is not None else [0, 1, 2, 3, 4, 5]
        self.cost_rate = (commission_bps + slippage_bps) / 10000.0  # e.g. 7 bps total friction
        
        self.feature_engineer = QuantFeatureEngineer(horizons=self.horizons)
        self.metrics_calc = QuantMetricsCalculator(periods_per_year=24 * 365, risk_free_rate=settings.RISK_FREE_RATE)

    def run_validation(self, df_merged: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes complete expanding window walk-forward validation across dataset.
        Returns:
            - performance_summary: Dict of Sharpe, Sortino, Max Drawdown, Calmar
            - horizon_metrics: Dict of per-horizon RMSE, MAE, and Directional Accuracy
            - oos_predictions_df: Consolidated DataFrame of all out-of-sample predictions
            - equity_curve_df: Strategy and benchmark portfolio paths over time
        """
        logger.info("Engineering features strictly from sentiment and price...")
        df_feat = self.feature_engineer.engineer_features(df_merged)
        feature_cols = self.feature_engineer.get_feature_column_names(df_feat)
        
        total_bars = len(df_feat)
        if total_bars < (self.train_window_bars + self.test_step_bars):
            raise ValueError(f"Total dataset bars ({total_bars}) insufficient for train window ({self.train_window_bars})")

        logger.info(f"Initiating Walk-Forward Validation: {total_bars} bars total, {len(feature_cols)} features...")

        oos_records = []
        train_start = 0
        train_end = self.train_window_bars
        fold_idx = 0

        while (train_end + self.embargo_bars) < total_bars:
            test_start = train_end + self.embargo_bars
            test_end = min(test_start + self.test_step_bars, total_bars)

            if test_start >= total_bars:
                break

            logger.info(f"  -> Walk-Forward Fold {fold_idx}: train=[{train_start}:{train_end}], test=[{test_start}:{test_end}]")

            train_df = df_feat.iloc[train_start:train_end]
            test_df = df_feat.iloc[test_start:test_end]

            # Features & Targets for Training
            X_train = train_df[feature_cols]
            y_train = {h: train_df[f"target_ret_n{h}"] for h in self.horizons if f"target_ret_n{h}" in train_df.columns}

            # Train Forecaster
            forecaster = MultiHorizonForecaster(
                algorithm=self.algorithm,
                horizons=self.horizons,
                hyperparams=self.settings.HYPERPARAMS
            )
            forecaster.fit(X_train, y_train)

            # Predict on Test Set (Pure Out-of-Sample)
            X_test = test_df[feature_cols]
            current_close_test = test_df["close"].values
            
            pred_rets = forecaster.predict_returns(X_test)
            pred_prices = forecaster.predict_prices(X_test, test_df["close"])

            # Store OOS predictions
            for idx in range(len(test_df)):
                row_idx = test_df.index[idx]
                ts_utc = test_df["timestamp_utc"].iloc[idx]
                close_p = current_close_test[idx]
                sent_w = test_df["sentiment_weighted"].iloc[idx]

                record = {
                    "fold": fold_idx,
                    "timestamp_utc": ts_utc,
                    "close": close_p,
                    "sentiment_weighted": sent_w,
                    "realized_next_return": test_df["price_log_ret"].iloc[idx]
                }

                for h in self.horizons:
                    record[f"pred_ret_n{h}"] = pred_rets[h][idx]
                    record[f"pred_price_n{h}"] = pred_prices[h][idx]
                    record[f"true_ret_n{h}"] = test_df[f"target_ret_n{h}"].iloc[idx] if f"target_ret_n{h}" in test_df.columns else 0.0
                    record[f"true_price_n{h}"] = test_df[f"target_price_n{h}"].iloc[idx] if f"target_price_n{h}" in test_df.columns else close_p

                oos_records.append(record)

            fold_idx += 1
            train_end += self.test_step_bars  # Expanding window

        oos_df = pd.DataFrame(oos_records)
        if oos_df.empty:
            raise RuntimeError("Walk-forward produced 0 out-of-sample predictions.")

        oos_df.sort_values(by="timestamp_utc", inplace=True)
        oos_df.reset_index(drop=True, inplace=True)

        # -------------------------------------------------------------
        # Evaluate Forecast Accuracy Per Horizon
        # -------------------------------------------------------------
        horizon_metrics = {}
        for h in self.horizons:
            pred_r = oos_df[f"pred_ret_n{h}"].values
            true_r = oos_df[f"true_ret_n{h}"].values
            horizon_metrics[f"n+{h}"] = self.metrics_calc.calculate_horizon_forecasting_metrics(pred_r, true_r)

        # -------------------------------------------------------------
        # Simulate Out-of-Sample Trading Strategy
        # -------------------------------------------------------------
        # Composite alpha combining short-horizon predictions (n+1, n+2, n+3) with sentiment confirmation
        raw_signal = (
            0.40 * oos_df["pred_ret_n1"] +
            0.30 * oos_df["pred_ret_n2"] +
            0.20 * oos_df["pred_ret_n3"] +
            0.10 * (oos_df["sentiment_weighted"] * 0.005)
        )

        # Standardize signal rolling over 72h to adapt to local volatility
        sig_roll_mean = raw_signal.rolling(72, min_periods=12).mean().fillna(0.0)
        sig_roll_std = raw_signal.rolling(72, min_periods=12).std().fillna(1.0).replace(0, 1.0)
        z_signal = ((raw_signal - sig_roll_mean) / sig_roll_std).clip(-2.5, 2.5)

        # Target raw position with deadband: ignore noise within [-0.25, +0.25]
        target_positions = np.where(z_signal > 0.25, np.tanh(z_signal), np.where(z_signal < -0.25, np.tanh(z_signal), 0.0))

        # Position smoothing to prevent fee churning (reduce turnover)
        smoothed_pos = np.zeros(len(target_positions))
        curr = 0.0
        for i, tp in enumerate(target_positions):
            # Slow adaptation to avoid churning on noise
            curr = 0.88 * curr + 0.12 * tp
            smoothed_pos[i] = curr

        # Shift position by 1 bar to simulate trade entry strictly at next bar open without lookahead
        actual_positions = pd.Series(smoothed_pos).shift(1).fillna(0.0).values
        bench_returns = oos_df["realized_next_return"].values

        # Execution friction: turnover * cost_rate
        turnover = np.abs(np.diff(actual_positions, prepend=0.0))
        friction = turnover * self.cost_rate
        strat_returns = (actual_positions * bench_returns) - friction

        oos_df["position"] = actual_positions
        oos_df["strategy_return"] = strat_returns
        oos_df["benchmark_return"] = bench_returns
        oos_df["strategy_equity"] = np.cumprod(1.0 + strat_returns) * 100000.0
        oos_df["benchmark_equity"] = np.cumprod(1.0 + bench_returns) * 100000.0

        # Calculate high-level performance metrics
        perf_summary = self.metrics_calc.calculate_strategy_metrics(strat_returns, bench_returns)

        return {
            "performance_summary": perf_summary,
            "horizon_metrics": horizon_metrics,
            "oos_predictions_df": oos_df,
            "total_folds": fold_idx
        }

