"""
Configuration settings module for CrowdWisdomTrading X Sentiment Price Prediction Agent.
Loads configuration from YAML and environment variables with comprehensive fallback defaults.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Enforce thread safety and prevent subprocess CPU counting hangs in sandboxed execution
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
ENV_PATH = BASE_DIR / ".env"


def load_env_file(filepath: Path = ENV_PATH) -> None:
    """Manually parse .env file if present, avoiding hard dependency on python-dotenv."""
    if not filepath.exists():
        return
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"Warning: Failed to parse .env file: {e}")


load_env_file()


def load_yaml_config(filepath: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration YAML file."""
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_YAML_CONFIG = load_yaml_config()


class Settings:
    """Central configuration class with typed access and environment overrides."""

    BASE_DIR: Path = BASE_DIR

    # Project Information
    PROJECT_NAME: str = "CrowdWisdomTrading X Sentiment Price Prediction Agent"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Quantitative Data Science Candidate"
    SUPERVISOR: str = "Gilad Bar-Ilan (CEO, CrowdWisdomTrading)"
    EMAIL: str = "gilad@crowdwisdomtrading.com"

    # API Keys & Scraper Credentials
    APIFY_API_TOKEN: str = os.getenv("APIFY_API_TOKEN", "")
    APIFY_ACTOR_ID: str = os.getenv("APIFY_ACTOR_ID", _YAML_CONFIG.get("apify", {}).get("actor_id", "apidojo/tweet-scraper"))
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
    ALLOW_MOCK_FALLBACK: bool = os.getenv("ALLOW_MOCK_FALLBACK", "True").lower() in ("true", "1", "yes")

    # Database
    DATABASE_PATH: Path = BASE_DIR / os.getenv("DATABASE_PATH", "data_cache/market_sentiment.db")

    # Asset Specs
    asset_cfg: Dict[str, Any] = _YAML_CONFIG.get("asset", {})
    SYMBOL: str = os.getenv("TARGET_ASSET", asset_cfg.get("symbol", "BTC-USD"))
    ASSET_NAME: str = asset_cfg.get("name", "Bitcoin")
    CASHTAGS: List[str] = asset_cfg.get("cashtags", ["$BTC", "$BITCOIN"])
    KEYWORDS: List[str] = asset_cfg.get("keywords", ["Bitcoin", "BTC", "crypto"])
    LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", asset_cfg.get("lookback_days", 180)))
    TIMEFRAME: str = os.getenv("TARGET_TIMEFRAME", asset_cfg.get("timeframe", "1h"))
    TIMEZONE: str = os.getenv("TIMEZONE", asset_cfg.get("timezone", "UTC"))

    # Feature & Model Specs
    features_cfg: Dict[str, Any] = _YAML_CONFIG.get("features", {})
    HORIZONS: List[int] = features_cfg.get("horizons", [0, 1, 2, 3, 4, 5])
    SENTIMENT_LAGS: List[int] = features_cfg.get("sentiment_lags", [1, 2, 3, 6, 12, 24])
    PRICE_LAGS: List[int] = features_cfg.get("price_return_lags", [1, 2, 3, 5, 8, 13, 21])
    ROLLING_WINDOWS: List[int] = features_cfg.get("rolling_windows", [6, 12, 24, 72])
    ENGAGEMENT_WEIGHTS: Dict[str, float] = features_cfg.get("engagement_weights", {
        "likes": 1.0,
        "retweets": 2.0,
        "replies": 1.5,
        "quotes": 1.5,
        "views": 0.05
    })

    # Model Algorithm
    model_cfg: Dict[str, Any] = _YAML_CONFIG.get("model", {})
    ALGORITHM: str = model_cfg.get("algorithm", "lightgbm")
    HYPERPARAMS: Dict[str, Any] = model_cfg.get("hyperparameters", {})

    # Validation
    val_cfg: Dict[str, Any] = _YAML_CONFIG.get("validation", {})
    TRAIN_WINDOW_DAYS: int = val_cfg.get("train_window_days", 60)
    TEST_STEP_DAYS: int = val_cfg.get("test_step_days", 7)
    EMBARGO_PERIODS: int = val_cfg.get("embargo_periods", 6)

    # Trading Execution Simulation
    trade_cfg: Dict[str, Any] = _YAML_CONFIG.get("trading", {})
    INITIAL_CAPITAL: float = trade_cfg.get("initial_capital", 100000.0)
    COMMISSION_BPS: float = trade_cfg.get("commission_bps", 5.0)
    SLIPPAGE_BPS: float = trade_cfg.get("slippage_bps", 2.0)
    RISK_FREE_RATE: float = trade_cfg.get("risk_free_rate", 0.04)

    # Directories
    OUTPUT_DIR: Path = BASE_DIR / "output"
    DATA_CACHE_DIR: Path = BASE_DIR / "data_cache"
    REPORTS_DIR: Path = BASE_DIR / "reports"


settings = Settings()
