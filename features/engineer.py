"""
Quantitative Feature Engineering Pipeline for CrowdWisdomTrading.
Strictly constructs time-based features from SENTIMENT AND PRICE ONLY,
guaranteeing zero data leakage, rolling z-score scaling, and multi-horizon target labels (n+0 to n+5).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


class QuantFeatureEngineer:
    """
    Constructs quantitative alphas strictly from sentiment series and price series.
    No other exogenous data sources used, adhering strictly to technical instructions.
    """

    def __init__(
        self,
        sentiment_lags: Optional[List[int]] = None,
        price_lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        horizons: Optional[List[int]] = None,
    ):
        from config.settings import settings
        self.sentiment_lags = sentiment_lags or settings.SENTIMENT_LAGS
        self.price_lags = price_lags or settings.PRICE_LAGS
        self.rolling_windows = rolling_windows or settings.ROLLING_WINDOWS
        self.horizons = horizons or settings.HORIZONS

    def engineer_features(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        Takes merged price and aggregated sentiment DataFrame and computes all quant features.
        """
        df = df_merged.copy()
        if "close" not in df.columns or "sentiment_weighted" not in df.columns:
            raise ValueError("Input DataFrame must contain 'close' and 'sentiment_weighted' columns.")

        # Ensure sorted chronologically
        if "timestamp_utc" in df.columns:
            df.sort_values(by="timestamp_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        close = df["close"].astype(float)
        high = df.get("high", close).astype(float)
        low = df.get("low", close).astype(float)
        open_p = df.get("open", close).astype(float)

        sent = df["sentiment_weighted"].astype(float)
        post_vol = df.get("post_volume", pd.Series(0, index=df.index)).astype(float)

        # -------------------------------------------------------------
        # 1. PRICE-BASED FEATURES (Returns, Volatility, Momentum)
        # -------------------------------------------------------------
        # Single-period log return
        df["price_log_ret"] = np.log(close / close.shift(1)).fillna(0.0)

        # Lagged log returns
        for lag in self.price_lags:
            df[f"price_ret_lag_{lag}"] = df["price_log_ret"].shift(lag).fillna(0.0)

        # Cumulative price returns across multiple windows
        for w in [3, 6, 12, 24, 72]:
            df[f"price_cum_ret_{w}h"] = (close / close.shift(w) - 1.0).fillna(0.0)

        # Realized price volatility (standard deviation of hourly returns)
        for w in self.rolling_windows:
            df[f"price_realized_vol_{w}h"] = (
                df["price_log_ret"].rolling(window=w, min_periods=max(2, w // 4)).std().fillna(0.0)
            )

        # High-Low normalized spread (Parkinson range proxy)
        df["price_hl_spread"] = ((high - low) / close).fillna(0.0)
        df["price_co_spread"] = ((close - open_p) / open_p).fillna(0.0)

        # Price momentum vs rolling moving averages
        for w in [12, 24, 72]:
            ma = close.rolling(window=w, min_periods=w // 2).mean()
            df[f"price_dist_ma_{w}h"] = ((close - ma) / ma).fillna(0.0)

        # -------------------------------------------------------------
        # 2. SENTIMENT-BASED FEATURES (Consensus, Velocity, Volume)
        # -------------------------------------------------------------
        # Log post volume to compress heavy tails
        df["log_post_volume"] = np.log1p(post_vol)

        # Volume surge ratio vs 24h rolling average
        vol_mean_24h = post_vol.rolling(window=24, min_periods=4).mean().replace(0, 1.0)
        df["sentiment_vol_surge_ratio"] = (post_vol / vol_mean_24h).fillna(1.0)

        # Sentiment velocity and acceleration
        df["sent_velocity_1h"] = (sent - sent.shift(1)).fillna(0.0)
        df["sent_velocity_3h"] = (sent - sent.shift(3)).fillna(0.0)
        df["sent_velocity_6h"] = (sent - sent.shift(6)).fillna(0.0)
        df["sent_acceleration"] = (df["sent_velocity_1h"] - df["sent_velocity_1h"].shift(1)).fillna(0.0)

        # Sentiment lags
        for lag in self.sentiment_lags:
            df[f"sent_lag_{lag}"] = sent.shift(lag).fillna(0.0)
            df[f"post_vol_lag_{lag}"] = df["log_post_volume"].shift(lag).fillna(0.0)

        # Sentiment rolling moving averages and z-scores
        for w in self.rolling_windows:
            s_ma = sent.rolling(window=w, min_periods=max(2, w // 4)).mean()
            s_std = sent.rolling(window=w, min_periods=max(2, w // 4)).std().replace(0, np.nan)
            df[f"sent_ma_{w}h"] = s_ma.fillna(0.0)
            df[f"sent_zscore_{w}h"] = ((sent - s_ma) / s_std).fillna(0.0)

        # Sentiment dispersion / disagreement if present
        if "sentiment_std" in df.columns:
            df["crowd_disagreement"] = df["sentiment_std"].fillna(0.0)
        else:
            df["crowd_disagreement"] = 0.0

        # -------------------------------------------------------------
        # 3. SENTIMENT & PRICE INTERACTION FEATURES
        # -------------------------------------------------------------
        # Sentiment-Price Divergence:
        # Measures when crowd euphoria occurs on weak price, or panic on strong price
        p_zscore = df["price_dist_ma_24h"] / (df["price_realized_vol_24h"] + 1e-6)
        s_zscore = df["sent_zscore_24h"]
        df["sentiment_price_divergence"] = (s_zscore - p_zscore).fillna(0.0)

        # Interaction product
        df["sent_x_price_return"] = (sent * df["price_log_ret"]).fillna(0.0)
        df["volume_x_volatility"] = (df["log_post_volume"] * df["price_realized_vol_24h"]).fillna(0.0)

        # -------------------------------------------------------------
        # 4. MULTI-HORIZON TARGETS: n+0, n+1, n+2, n+3, n+4, n+5
        # -------------------------------------------------------------
        # Target Price at n+h and Target Return from period n
        for h in self.horizons:
            # Shift backwards (into the future) by h periods
            future_price = close.shift(-h)
            df[f"target_price_n{h}"] = future_price
            
            # Cumulative return from current close to future close
            # For h=0: immediate bar target return (e.g. close vs close, 0.0 or next bar open/close)
            if h == 0:
                df[f"target_ret_n{h}"] = 0.0
            else:
                df[f"target_ret_n{h}"] = ((future_price - close) / close).fillna(0.0)

        return df

    def get_feature_column_names(self, df: pd.DataFrame) -> List[str]:
        """
        Returns list of engineered feature names strictly excluding targets, timestamps, and identifiers.
        """
        excluded = {
            "timestamp_utc", "timestamp_est", "asset_symbol", "timeframe",
            "open", "high", "low", "close", "volume",
            "bar_timestamp", "text", "tweet_id", "author_username"
        }
        for h in self.horizons:
            excluded.add(f"target_price_n{h}")
            excluded.add(f"target_ret_n{h}")

        feature_cols = [c for c in df.columns if c not in excluded and not c.startswith("target_")]
        return feature_cols

