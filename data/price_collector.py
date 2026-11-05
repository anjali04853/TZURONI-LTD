"""
Price Collector Module for CrowdWisdomTrading.
Fetches historical OHLCV price bars for 180 days with hourly/daily resolution,
aligns timezones across UTC and EST, and provides continuous fallback data.
"""

from __future__ import annotations
import logging
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pytz
import requests

logger = logging.getLogger(__name__)


class PriceCollector:
    """Collects historical price bars from Binance, Yahoo Finance, or synthetic generator."""

    def __init__(self, cache_dir: Optional[Path] = None):
        from config.settings import settings
        self.cache_dir = cache_dir or settings.DATA_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_historical_bars(
        self,
        symbol: str = "BTC-USD",
        timeframe: str = "1h",
        lookback_days: int = 180,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Retrieves historical OHLCV bars with dual UTC and EST timestamp indices.
        """
        clean_sym = symbol.replace("/", "_").replace("-", "_")
        cache_path = self.cache_dir / f"price_bars_{clean_sym}_{timeframe}_{lookback_days}d.csv"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached price bars from {cache_path}")
            df = pd.read_csv(cache_path)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            df["timestamp_est"] = pd.to_datetime(df["timestamp_est"])
            return df

        df = pd.DataFrame()

        # Try Binance public API (ideal for BTC, ETH, crypto assets)
        if "BTC" in symbol or "ETH" in symbol or "crypto" in symbol.lower():
            try:
                df = self._fetch_binance_klines(symbol, timeframe, lookback_days)
            except Exception as e:
                logger.warning(f"Binance fetch failed: {e}")

        # Try Yahoo Finance if Binance failed or if asset is equity
        if df.empty:
            try:
                df = self._fetch_yfinance_bars(symbol, timeframe, lookback_days)
            except Exception as e:
                logger.warning(f"Yahoo Finance fetch failed: {e}")

        # Fallback to realistic stochastic market process (jump-diffusion)
        if df.empty:
            logger.info(f"Generating realistic 180-day price path with jump diffusion for {symbol}...")
            df = self._generate_realistic_price_series(symbol, timeframe, lookback_days)

        # Save to cache
        try:
            df.to_csv(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to cache price bars: {e}")

        return df

    def _fetch_binance_klines(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        """Fetch 1h klines from Binance public REST API in batches."""
        binance_sym = "BTCUSDT" if "BTC" in symbol else "ETHUSDT"
        interval = "1h" if timeframe == "1h" else "1d"
        limit = 1000

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)

        all_bars = []
        current_start = start_ms

        while current_start < now_ms:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": binance_sym,
                "interval": interval,
                "startTime": current_start,
                "endTime": now_ms,
                "limit": limit
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data or not isinstance(data, list):
                break
            all_bars.extend(data)
            current_start = data[-1][6] + 1  # Next close time + 1ms

        if not all_bars:
            return pd.DataFrame()

        rows = []
        utc_zone = pytz.UTC
        est_zone = pytz.timezone("US/Eastern")

        for k in all_bars:
            open_time_ms = k[0]
            dt_utc = datetime.fromtimestamp(open_time_ms / 1000.0, tz=utc_zone)
            dt_est = dt_utc.astimezone(est_zone)
            rows.append({
                "timestamp_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_est": dt_est.strftime("%Y-%m-%d %H:%M:%S"),
                "asset_symbol": symbol,
                "timeframe": timeframe,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })

        df = pd.DataFrame(rows)
        df.drop_duplicates(subset=["timestamp_utc"], inplace=True)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp_est"] = pd.to_datetime(df["timestamp_est"])
        return df

    def _fetch_yfinance_bars(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        """Fetch bars using yfinance library."""
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        period = f"{min(lookback_days, 729)}d"
        interval = "1h" if timeframe == "1h" else "1d"
        raw_df = ticker.history(period=period, interval=interval)
        if raw_df.empty:
            return pd.DataFrame()

        raw_df.reset_index(inplace=True)
        date_col = "Datetime" if "Datetime" in raw_df.columns else "Date"
        raw_df["timestamp_utc"] = pd.to_datetime(raw_df[date_col], utc=True)
        raw_df["timestamp_est"] = raw_df["timestamp_utc"].dt.tz_convert("US/Eastern")

        df = pd.DataFrame({
            "timestamp_utc": raw_df["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_est": raw_df["timestamp_est"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "asset_symbol": symbol,
            "timeframe": timeframe,
            "open": raw_df["Open"].astype(float),
            "high": raw_df["High"].astype(float),
            "low": raw_df["Low"].astype(float),
            "close": raw_df["Close"].astype(float),
            "volume": raw_df["Volume"].astype(float)
        })
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp_est"] = pd.to_datetime(df["timestamp_est"])
        return df

    def _generate_realistic_price_series(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        """
        Generates realistic 180-day hourly asset prices with Merton jump-diffusion,
        volatility clustering (GARCH-like), and intraday seasonality.
        """
        np.random.seed(42)
        total_hours = lookback_days * 24
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_utc = now_utc - timedelta(hours=total_hours - 1)

        dates_utc = [start_utc + timedelta(hours=i) for i in range(total_hours)]
        est_zone = pytz.timezone("US/Eastern")
        dates_est = [d.astimezone(est_zone) for d in dates_utc]

        # Parameters calibrated to Bitcoin historical hourly dynamics
        s0 = 62500.0 if "BTC" in symbol else (120.0 if "NVDA" in symbol else 220.0)
        mu = 0.00005        # ~40% annualized drift
        sigma_base = 0.004  # ~60% annualized hourly volatility
        lambda_jump = 0.008 # Average jump frequency per hour
        jump_mu = -0.001
        jump_sigma = 0.015

        prices = [s0]
        vols = [sigma_base]

        for i in range(1, total_hours):
            days_passed = i / 24.0
            # Macro sentiment regime drift (+0.0006 in bull regime, -0.0006 in bear regime)
            regime_drift = 0.0005 * math.sin(days_passed * 2 * math.pi / 45.0) + 0.0002 * math.sin(days_passed * 2 * math.pi / 7.0)

            # Volatility clustering (GARCH(1,1) proxy)
            prev_ret = (prices[-1] - prices[-2]) / prices[-2] if i > 1 else 0.0
            vol = math.sqrt(0.000001 + 0.88 * (vols[-1]**2) + 0.10 * (prev_ret**2))
            vol = min(max(vol, 0.002), 0.018)
            vols.append(vol)

            # Diffusion component
            z = np.random.normal(0, 1)
            diff_ret = (mu + regime_drift) - 0.5 * (vol**2) + vol * z

            # Jump component
            jump_occurred = np.random.poisson(lambda_jump) > 0
            jump_ret = np.random.normal(jump_mu, jump_sigma) if jump_occurred else 0.0

            hourly_ret = diff_ret + jump_ret
            new_p = max(prices[-1] * math.exp(hourly_ret), 1.0)
            prices.append(new_p)

        # Build OHLCV bars
        rows = []
        for i in range(total_hours):
            c = prices[i]
            # Intra-hour range
            h_range = c * vols[i] * np.random.uniform(0.6, 1.8)
            o = prices[i - 1] if i > 0 else c
            h = max(o, c) + h_range * np.random.uniform(0.1, 0.5)
            l = min(o, c) - h_range * np.random.uniform(0.1, 0.5)
            v = float(np.random.lognormal(8.0, 0.6) * (1.0 + abs(c - o) / (o * 0.01)))

            rows.append({
                "timestamp_utc": dates_utc[i].strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_est": dates_est[i].strftime("%Y-%m-%d %H:%M:%S"),
                "asset_symbol": symbol,
                "timeframe": timeframe,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": round(v, 2)
            })

        df = pd.DataFrame(rows)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["timestamp_est"] = pd.to_datetime(df["timestamp_est"])
        return df

