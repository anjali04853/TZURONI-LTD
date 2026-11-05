"""
Database module for CrowdWisdomTrading X Sentiment Price Prediction Agent.
Provides unified SQLite storage for raw tweets, news articles, price bars,
aligned features, and multi-horizon predictions.
"""

from __future__ import annotations
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import pytz


class DatabaseManager:
    """Manages SQLite storage and joins for market, sentiment, and news data."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            from config.settings import settings
            self.db_path = settings.DATABASE_PATH
        else:
            self.db_path = Path(db_path)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def get_connection(self) -> sqlite3.Connection:
        """Create a connection with foreign keys and WAL journal mode enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_database(self) -> None:
        """Initializes database schema with optimized indices."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Raw X (Twitter) Posts Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_tweets (
                tweet_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                created_at_est TEXT NOT NULL,
                asset_symbol TEXT NOT NULL,
                author_username TEXT,
                author_followers_count INTEGER DEFAULT 0,
                author_verified INTEGER DEFAULT 0,
                text TEXT NOT NULL,
                likes_count INTEGER DEFAULT 0,
                retweets_count INTEGER DEFAULT 0,
                replies_count INTEGER DEFAULT 0,
                quotes_count INTEGER DEFAULT 0,
                views_count INTEGER DEFAULT 0,
                sentiment_score REAL DEFAULT 0.0,
                engagement_score REAL DEFAULT 1.0,
                ingested_at TEXT NOT NULL
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tweets_utc ON raw_tweets(created_at_utc);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tweets_asset ON raw_tweets(asset_symbol);")

            # 2. News Ingestion Table (Tavily / Exa)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_items (
                news_id TEXT PRIMARY KEY,
                published_at_utc TEXT NOT NULL,
                asset_symbol TEXT NOT NULL,
                title TEXT NOT NULL,
                snippet TEXT,
                url TEXT,
                source TEXT,
                sentiment_score REAL DEFAULT 0.0,
                relevance_score REAL DEFAULT 1.0,
                ingested_at TEXT NOT NULL
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_utc ON news_items(published_at_utc);")

            # 3. Historical Price Bars Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_bars (
                timestamp_utc TEXT NOT NULL,
                timestamp_est TEXT NOT NULL,
                asset_symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (timestamp_utc, asset_symbol, timeframe)
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_utc ON price_bars(timestamp_utc);")

            # 4. Model Predictions Audit Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                asset_symbol TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                target_timestamp_utc TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                predicted_return REAL NOT NULL,
                actual_price REAL,
                actual_return REAL,
                model_name TEXT NOT NULL,
                is_out_of_sample INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp_utc, horizon);")
            conn.commit()

    def store_tweets(self, df: pd.DataFrame) -> int:
        """Upsert scraped tweets into database."""
        if df.empty:
            return 0

        # Ensure datetime alignment
        df_copy = df.copy()
        if "created_at_utc" in df_copy.columns:
            df_copy["created_at_utc"] = pd.to_datetime(df_copy["created_at_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "created_at_est" in df_copy.columns:
            df_copy["created_at_est"] = pd.to_datetime(df_copy["created_at_est"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        elif "created_at_utc" in df_copy.columns:
            utc_series = pd.to_datetime(df_copy["created_at_utc"], utc=True)
            df_copy["created_at_est"] = utc_series.dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d %H:%M:%S")
        if "ingested_at" not in df_copy.columns:
            df_copy["ingested_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            records = df_copy.to_dict(orient="records")
            query = """
            INSERT INTO raw_tweets (
                tweet_id, created_at_utc, created_at_est, asset_symbol,
                author_username, author_followers_count, author_verified,
                text, likes_count, retweets_count, replies_count, quotes_count,
                views_count, sentiment_score, engagement_score, ingested_at
            ) VALUES (
                :tweet_id, :created_at_utc, :created_at_est, :asset_symbol,
                :author_username, :author_followers_count, :author_verified,
                :text, :likes_count, :retweets_count, :replies_count, :quotes_count,
                :views_count, :sentiment_score, :engagement_score, :ingested_at
            ) ON CONFLICT(tweet_id) DO UPDATE SET
                likes_count=excluded.likes_count,
                retweets_count=excluded.retweets_count,
                views_count=excluded.views_count,
                sentiment_score=excluded.sentiment_score,
                engagement_score=excluded.engagement_score;
            """
            cursor.executemany(query, records)
            conn.commit()
            return len(records)

    def store_price_bars(self, df: pd.DataFrame, asset_symbol: str, timeframe: str = "1h") -> int:
        """Store historical OHLCV price bars."""
        if df.empty:
            return 0

        df_copy = df.copy()
        if "timestamp_utc" not in df_copy.columns:
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy["timestamp_utc"] = df_copy.index
            else:
                raise ValueError("DataFrame must contain timestamp_utc column or DatetimeIndex")

        utc_series = pd.to_datetime(df_copy["timestamp_utc"], utc=True)
        df_copy["timestamp_utc"] = utc_series.dt.strftime("%Y-%m-%d %H:%M:%S")
        df_copy["timestamp_est"] = utc_series.dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d %H:%M:%S")
        df_copy["asset_symbol"] = asset_symbol
        df_copy["timeframe"] = timeframe

        cols = ["timestamp_utc", "timestamp_est", "asset_symbol", "timeframe", "open", "high", "low", "close", "volume"]
        df_insert = df_copy[cols]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            records = df_insert.to_dict(orient="records")
            query = """
            INSERT INTO price_bars (
                timestamp_utc, timestamp_est, asset_symbol, timeframe,
                open, high, low, close, volume
            ) VALUES (
                :timestamp_utc, :timestamp_est, :asset_symbol, :timeframe,
                :open, :high, :low, :close, :volume
            ) ON CONFLICT(timestamp_utc, asset_symbol, timeframe) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume;
            """
            cursor.executemany(query, records)
            conn.commit()
            return len(records)

    def store_news(self, df: pd.DataFrame) -> int:
        """Store news items from Tavily/Exa."""
        if df.empty:
            return 0
        df_copy = df.copy()
        if "ingested_at" not in df_copy.columns:
            df_copy["ingested_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            records = df_copy.to_dict(orient="records")
            query = """
            INSERT INTO news_items (
                news_id, published_at_utc, asset_symbol, title,
                snippet, url, source, sentiment_score, relevance_score, ingested_at
            ) VALUES (
                :news_id, :published_at_utc, :asset_symbol, :title,
                :snippet, :url, :source, :sentiment_score, :relevance_score, :ingested_at
            ) ON CONFLICT(news_id) DO UPDATE SET
                sentiment_score=excluded.sentiment_score,
                relevance_score=excluded.relevance_score;
            """
            cursor.executemany(query, records)
            conn.commit()
            return len(records)

    def store_predictions(self, df: pd.DataFrame) -> int:
        """Store multi-horizon predictions for audit and evaluation."""
        if df.empty:
            return 0
        df_copy = df.copy()
        if "created_at" not in df_copy.columns:
            df_copy["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with self.get_connection() as conn:
            df_copy.to_sql("predictions", conn, if_exists="append", index=False)
            return len(df_copy)

    def load_tweets(self, asset_symbol: Optional[str] = None, start_utc: Optional[str] = None) -> pd.DataFrame:
        """Retrieve tweets filtered by asset and optional start date."""
        query = "SELECT * FROM raw_tweets WHERE 1=1"
        params: List[Any] = []
        if asset_symbol:
            query += " AND asset_symbol = ?"
            params.append(asset_symbol)
        if start_utc:
            query += " AND created_at_utc >= ?"
            params.append(start_utc)
        query += " ORDER BY created_at_utc ASC"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], utc=True)
                df["created_at_est"] = pd.to_datetime(df["created_at_est"])
            return df

    def load_price_bars(self, asset_symbol: str, timeframe: str = "1h", start_utc: Optional[str] = None) -> pd.DataFrame:
        """Retrieve price bars ordered chronologically."""
        query = "SELECT * FROM price_bars WHERE asset_symbol = ? AND timeframe = ?"
        params: List[Any] = [asset_symbol, timeframe]
        if start_utc:
            query += " AND timestamp_utc >= ?"
            params.append(start_utc)
        query += " ORDER BY timestamp_utc ASC"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
                df["timestamp_est"] = pd.to_datetime(df["timestamp_est"])
                df.set_index("timestamp_utc", inplace=False)
            return df

    def load_news(self, asset_symbol: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Retrieve recent news items."""
        query = "SELECT * FROM news_items WHERE 1=1"
        params: List[Any] = []
        if asset_symbol:
            query += " AND asset_symbol = ?"
            params.append(asset_symbol)
        query += f" ORDER BY published_at_utc DESC LIMIT {limit}"

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
