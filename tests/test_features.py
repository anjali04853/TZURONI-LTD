"""
Unit tests for Quantitative Feature Engineering and Sentiment Processing.
Validates zero data leakage, target horizon creation, and pure sentiment-price feature constraint.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from features.engineer import QuantFeatureEngineer
from sentiment.analyzer import SentimentAnalyzer
from sentiment.aggregator import SentimentAggregator


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Create synthetic merged dataset
        now = datetime.now(timezone.utc)
        dates = [now - timedelta(hours=i) for i in reversed(range(100))]
        
        prices = [60000.0 * (1.0 + 0.001 * np.sin(i / 5.0)) for i in range(100)]
        sents = [0.2 * np.cos(i / 4.0) for i in range(100)]
        vols = [int(15 + 5 * np.sin(i / 3.0)) for i in range(100)]

        self.df = pd.DataFrame({
            "timestamp_utc": dates,
            "close": prices,
            "open": [p * 0.999 for p in prices],
            "high": [p * 1.002 for p in prices],
            "low": [p * 0.998 for p in prices],
            "volume": [150.0] * 100,
            "sentiment_weighted": sents,
            "post_volume": vols,
            "sentiment_std": [0.15] * 100
        })

    def test_sentiment_analyzer(self):
        analyzer = SentimentAnalyzer()
        res_bull = analyzer.score_text("Bitcoin breakout to new ATH! Extremely bullish! 🚀")
        self.assertGreater(res_bull["sentiment"], 0.2)
        self.assertEqual(res_bull["polarity"], "BULLISH")

        res_bear = analyzer.score_text("Huge crash and dump incoming! Liquidations everywhere. 🛑")
        self.assertLess(res_bear["sentiment"], -0.2)
        self.assertEqual(res_bear["polarity"], "BEARISH")

        res_neg = analyzer.score_text("This is not bullish at all, resistance holding firm.")
        self.assertLess(res_neg["sentiment"], 0.0)

    def test_feature_creation(self):
        fe = QuantFeatureEngineer()
        df_feat = fe.engineer_features(self.df)
        
        # Verify horizons n+0 to n+5 targets exist
        for h in [0, 1, 2, 3, 4, 5]:
            self.assertIn(f"target_price_n{h}", df_feat.columns)
            self.assertIn(f"target_ret_n{h}", df_feat.columns)

        # Verify key sentiment and price features exist
        self.assertIn("sentiment_price_divergence", df_feat.columns)
        self.assertIn("sent_velocity_1h", df_feat.columns)
        self.assertIn("price_realized_vol_24h", df_feat.columns)
        self.assertIn("sent_zscore_24h", df_feat.columns)

    def test_feature_columns_exclude_targets(self):
        fe = QuantFeatureEngineer()
        df_feat = fe.engineer_features(self.df)
        feat_names = fe.get_feature_column_names(df_feat)

        for col in feat_names:
            self.assertFalse(col.startswith("target_"))
            self.assertNotIn(col, ["close", "open", "high", "low", "volume", "timestamp_utc"])


if __name__ == "__main__":
    unittest.main()

