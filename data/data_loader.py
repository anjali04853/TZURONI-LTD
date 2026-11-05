"""
Unified Data Loader and Orchestration Module for CrowdWisdomTrading.
Coordinates tweet scraping (Apify), news ingestion (Tavily/Exa), price bars collection,
Google Doc historical tweets import, and atomic database persistence.
"""

from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import pytz

from data.database import DatabaseManager
from data.apify_collector import ApifyTweetCollector
from data.tavily_collector import TavilyNewsCollector
from data.price_collector import PriceCollector

logger = logging.getLogger(__name__)


class UnifiedDataLoader:
    """End-to-end data pipeline orchestrator."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        from config.settings import settings
        self.settings = settings
        self.db = db_manager or DatabaseManager()
        self.tweet_collector = ApifyTweetCollector()
        self.news_collector = TavilyNewsCollector()
        self.price_collector = PriceCollector()

    def run_full_ingestion(
        self,
        symbol: Optional[str] = None,
        lookback_days: Optional[int] = None,
        timeframe: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, int]:
        """
        Executes full data ingestion pipeline:
        1. Ingests raw X posts for last 180 days via Apify.
        2. Ingests OHLCV price bars for the same period.
        3. Ingests recent asset news via Tavily.
        4. Joins and writes all records into database.
        """
        symbol = symbol or self.settings.SYMBOL
        lookback_days = lookback_days or self.settings.LOOKBACK_DAYS
        timeframe = timeframe or self.settings.TIMEFRAME

        logger.info(f"Starting ingestion pipeline for {symbol} (lookback={lookback_days}d, timeframe={timeframe})...")

        # 1. Ingest Tweets
        tweets_df = self.tweet_collector.scrape_tweets(
            symbol=symbol,
            lookback_days=lookback_days,
            use_cache=use_cache
        )
        tweets_stored = self.db.store_tweets(tweets_df)
        logger.info(f"Stored {tweets_stored} tweets in database.")

        # 2. Ingest Price Bars
        price_df = self.price_collector.get_historical_bars(
            symbol=symbol,
            timeframe=timeframe,
            lookback_days=lookback_days,
            use_cache=use_cache
        )
        bars_stored = self.db.store_price_bars(price_df, asset_symbol=symbol, timeframe=timeframe)
        logger.info(f"Stored {bars_stored} price bars in database.")

        # 3. Ingest News
        news_df = self.news_collector.fetch_recent_news(
            symbol=symbol,
            use_cache=use_cache
        )
        news_stored = self.db.store_news(news_df)
        logger.info(f"Stored {news_stored} news items in database.")

        return {
            "tweets_stored": tweets_stored,
            "bars_stored": bars_stored,
            "news_stored": news_stored
        }

    def import_google_doc_tweets(self, raw_text_or_path: str, symbol: str = "BTC-USD") -> int:
        """
        Parses historical tweets provided in Google Docs or exported text/csv files.
        Extracts author, timestamp, tweet text, likes, retweets, and followers.
        """
        content = ""
        p = Path(raw_text_or_path)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            content = raw_text_or_path

        records = []
        utc_zone = pytz.UTC
        est_zone = pytz.timezone("US/Eastern")
        now_utc = datetime.now(timezone.utc)

        # Regex heuristics for structured and unstructured social posts
        # Handles lines like: "@user (15k followers) [2024-05-01 14:00]: Bitcoin breaking out! [Likes: 140, RT: 32]"
        lines = content.split("\n")
        current_tweet = {}

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Look for username pattern
            user_match = re.search(r"@(\w+)", line)
            username = user_match.group(1) if user_match else f"doc_trader_{i}"

            # Look for dates
            date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)", line)
            if date_match:
                try:
                    dt_utc = pd.to_datetime(date_match.group(1), utc=True).to_pydatetime()
                except Exception:
                    dt_utc = now_utc
            else:
                dt_utc = now_utc

            dt_est = dt_utc.astimezone(est_zone)

            # Look for engagement counts
            likes_m = re.search(r"likes?[:\s]+(\d+)", line, re.IGNORECASE)
            likes = int(likes_m.group(1)) if likes_m else 25

            rt_m = re.search(r"(?:rt|retweets?)[:\s]+(\d+)", line, re.IGNORECASE)
            rt = int(rt_m.group(1)) if rt_m else 5

            followers_m = re.search(r"followers?[:\s]+(\d+)", line, re.IGNORECASE)
            followers = int(followers_m.group(1)) if followers_m else 2500

            text_cleaned = re.sub(r"\[.*?\]", "", line).strip()
            if len(text_cleaned) < 5:
                text_cleaned = line

            tweet_id = f"gdoc_tweet_{int(dt_utc.timestamp())}_{i}"
            engagement_score = float(likes * 1.0 + rt * 2.5)

            records.append({
                "tweet_id": tweet_id,
                "created_at_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "created_at_est": dt_est.strftime("%Y-%m-%d %H:%M:%S"),
                "asset_symbol": symbol,
                "author_username": username,
                "author_followers_count": followers,
                "author_verified": 0,
                "text": text_cleaned,
                "likes_count": likes,
                "retweets_count": rt,
                "replies_count": 0,
                "quotes_count": 0,
                "views_count": likes * 20,
                "sentiment_score": 0.0,
                "engagement_score": engagement_score,
                "ingested_at": now_utc.strftime("%Y-%m-%d %H:%M:%S")
            })

        if records:
            df = pd.DataFrame(records)
            return self.db.store_tweets(df)
        return 0

