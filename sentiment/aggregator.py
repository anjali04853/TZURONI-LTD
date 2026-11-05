"""
Sentiment Aggregator Module for CrowdWisdomTrading.
Aggregates high-frequency tweet sentiments into discrete time bars (hourly/daily),
computes engagement-weighted consensus, sentiment velocity, and aligns with price bars.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytz
from typing import Optional


class SentimentAggregator:
    """Aggregates post-level sentiment into time bars and aligns with price bars."""

    def __init__(self, timeframe: str = "1h", timezone_str: str = "UTC"):
        self.timeframe = timeframe
        self.freq_alias = "1h" if timeframe == "1h" else "1D"
        self.tz = pytz.timezone(timezone_str)

    def aggregate_tweets(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups tweets into time intervals and computes summary sentiment statistics.
        """
        if tweets_df.empty:
            return pd.DataFrame()

        df = tweets_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["created_at_utc"]):
            df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True)
        else:
            df["created_at_utc"] = df["created_at_utc"].dt.tz_convert("UTC")

        df["bar_timestamp"] = df["created_at_utc"].dt.floor(self.freq_alias)

        def calc_bar_stats(group: pd.DataFrame) -> pd.Series:
            n_posts = len(group)
            raw_sents = group["sentiment_score"].values
            weights = group.get("engagement_weight", pd.Series(1.0, index=group.index)).values
            
            s_mean = float(np.mean(raw_sents))
            s_std = float(np.std(raw_sents)) if n_posts > 1 else 0.0
            
            sum_w = float(np.sum(weights))
            s_weighted = float(np.sum(raw_sents * weights) / sum_w) if sum_w > 0 else s_mean
            
            n_pos = int(np.sum(raw_sents > 0.1))
            n_neg = int(np.sum(raw_sents < -0.1))
            pos_ratio = n_pos / n_posts
            neg_ratio = n_neg / n_posts

            total_likes = int(group.get("likes_count", 0).sum())
            total_retweets = int(group.get("retweets_count", 0).sum())
            total_views = int(group.get("views_count", 0).sum())

            return pd.Series({
                "post_volume": n_posts,
                "sentiment_mean": s_mean,
                "sentiment_std": s_std,
                "sentiment_weighted": s_weighted,
                "sentiment_pos_ratio": pos_ratio,
                "sentiment_neg_ratio": neg_ratio,
                "total_likes": total_likes,
                "total_retweets": total_retweets,
                "total_views": total_views
            })

        agg_df = df.groupby("bar_timestamp").apply(calc_bar_stats, include_groups=False).reset_index()
        agg_df.rename(columns={"bar_timestamp": "timestamp_utc"}, inplace=True)
        agg_df.sort_values(by="timestamp_utc", inplace=True)
        agg_df.reset_index(drop=True, inplace=True)

        # Compute dynamic sentiment velocity and acceleration
        agg_df["sentiment_velocity"] = agg_df["sentiment_weighted"].diff().fillna(0.0)
        agg_df["sentiment_acceleration"] = agg_df["sentiment_velocity"].diff().fillna(0.0)

        # Rolling sentiment z-scores (24 periods)
        roll_mean = agg_df["sentiment_weighted"].rolling(window=24, min_periods=4).mean()
        roll_std = agg_df["sentiment_weighted"].rolling(window=24, min_periods=4).std().replace(0, np.nan)
        agg_df["sentiment_zscore_24h"] = ((agg_df["sentiment_weighted"] - roll_mean) / roll_std).fillna(0.0)

        return agg_df

    def merge_sentiment_and_prices(
        self,
        sentiment_agg_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Inner/left joins aggregated sentiment metrics onto price bars.
        Fills missing intervals with neutral decay instead of arbitrary forward filling.
        """
        if price_df.empty:
            raise ValueError("Price DataFrame cannot be empty for merging.")

        prices = price_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(prices["timestamp_utc"]):
            prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
        else:
            prices["timestamp_utc"] = prices["timestamp_utc"].dt.tz_convert("UTC")

        prices["timestamp_utc"] = prices["timestamp_utc"].dt.floor(self.freq_alias)
        prices.sort_values(by="timestamp_utc", inplace=True)
        prices.drop_duplicates(subset=["timestamp_utc"], inplace=True)

        if sentiment_agg_df.empty:
            # Create neutral dummy columns
            merged = prices.copy()
            merged["post_volume"] = 0
            merged["sentiment_mean"] = 0.0
            merged["sentiment_std"] = 0.0
            merged["sentiment_weighted"] = 0.0
            merged["sentiment_pos_ratio"] = 0.33
            merged["sentiment_neg_ratio"] = 0.33
            merged["sentiment_velocity"] = 0.0
            merged["sentiment_acceleration"] = 0.0
            merged["sentiment_zscore_24h"] = 0.0
            return merged

        sent = sentiment_agg_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(sent["timestamp_utc"]):
            sent["timestamp_utc"] = pd.to_datetime(sent["timestamp_utc"], utc=True)

        # Merge on exact timestamp_utc bar
        merged = pd.merge(prices, sent, on="timestamp_utc", how="left")

        # Fill volume with 0
        merged["post_volume"] = merged["post_volume"].fillna(0).astype(int)

        # Exponential decay towards 0.0 for intervals with no posts
        merged["sentiment_weighted"] = merged["sentiment_weighted"].ffill().fillna(0.0) * (0.95 ** (merged["post_volume"] == 0).astype(int))
        merged["sentiment_mean"] = merged["sentiment_mean"].ffill().fillna(0.0)
        merged["sentiment_std"] = merged["sentiment_std"].fillna(0.0)
        merged["sentiment_pos_ratio"] = merged["sentiment_pos_ratio"].fillna(0.33)
        merged["sentiment_neg_ratio"] = merged["sentiment_neg_ratio"].fillna(0.33)
        merged["sentiment_velocity"] = merged["sentiment_velocity"].fillna(0.0)
        merged["sentiment_acceleration"] = merged["sentiment_acceleration"].fillna(0.0)
        merged["sentiment_zscore_24h"] = merged["sentiment_zscore_24h"].fillna(0.0)

        # EST timestamp derivation
        est_zone = pytz.timezone("US/Eastern")
        merged["timestamp_est"] = merged["timestamp_utc"].dt.tz_convert(est_zone)

        merged.sort_values(by="timestamp_utc", inplace=True)
        merged.reset_index(drop=True, inplace=True)
        return merged

