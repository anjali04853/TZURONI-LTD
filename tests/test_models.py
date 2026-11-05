"""
Unit tests for Multi-Horizon Forecasting and Quant Backtest Metrics.
"""

import unittest
import numpy as np
import pandas as pd

from models.multi_horizon import MultiHorizonForecaster
from backtest.metrics import QuantMetricsCalculator


class TestModelsAndMetrics(unittest.TestCase):

    def test_multi_horizon_forecaster(self):
        np.random.seed(42)
        n = 150
        X = pd.DataFrame({
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n)
        })
        current_prices = pd.Series(50000.0 + 100 * np.arange(n))

        y_dict = {
            h: pd.Series(0.001 * h + 0.005 * X["feat_1"]) for h in range(6)
        }

        forecaster = MultiHorizonForecaster(algorithm="ridge", horizons=[0, 1, 2, 3, 4, 5])
        forecaster.fit(X, y_dict)

        pred_rets = forecaster.predict_returns(X)
        pred_prices = forecaster.predict_prices(X, current_prices)

        for h in range(6):
            self.assertIn(h, pred_rets)
            self.assertEqual(len(pred_rets[h]), n)
            self.assertIn(h, pred_prices)
            self.assertEqual(len(pred_prices[h]), n)

    def test_metrics_calculation(self):
        calc = QuantMetricsCalculator(periods_per_year=24 * 365, risk_free_rate=0.04)
        
        # Test positive returns stream
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.005, 1000)
        benchmark = np.random.normal(0.0001, 0.006, 1000)

        metrics = calc.calculate_strategy_metrics(returns, benchmark)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertGreater(metrics["sharpe_ratio"], 0.0)
        self.assertGreater(metrics["sortino_ratio"], 0.0)
        self.assertGreaterEqual(metrics["max_drawdown_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()

