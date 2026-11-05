"""
Model Trainer and Orchestrator for CrowdWisdomTrading.
Coordinates training of multi-horizon models, feature importance extraction,
and model persistence.
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from features.engineer import QuantFeatureEngineer
from models.multi_horizon import MultiHorizonForecaster
from models.rl_agent import RLHorizonTradingAgent

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Manages training pipelines and model artifact lifecycles."""

    def __init__(
        self,
        algorithm: str = "lightgbm",
        horizons: Optional[List[int]] = None,
        model_dir: Optional[Path] = None
    ):
        from config.settings import settings
        self.settings = settings
        self.algorithm = algorithm or settings.ALGORITHM
        self.horizons = horizons or settings.HORIZONS
        self.model_dir = model_dir or (settings.DATA_CACHE_DIR / "models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_engineer = QuantFeatureEngineer(horizons=self.horizons)
        self.forecaster = MultiHorizonForecaster(
            algorithm=self.algorithm,
            horizons=self.horizons,
            hyperparams=settings.HYPERPARAMS
        )
        self.rl_agent = RLHorizonTradingAgent(cost_bps=settings.COMMISSION_BPS / 10000.0)

    def prepare_data(self, df_merged: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, pd.Series], pd.Series]:
        """
        Engineers features and returns X, targets dictionary for n+0..n+5, and close prices.
        """
        df_feat = self.feature_engineer.engineer_features(df_merged)
        feature_cols = self.feature_engineer.get_feature_column_names(df_feat)
        
        X = df_feat[feature_cols].copy()
        current_prices = df_feat["close"].copy()

        y_dict = {}
        for h in self.horizons:
            target_col = f"target_ret_n{h}"
            if target_col in df_feat.columns:
                y_dict[h] = df_feat[target_col].copy()

        return X, y_dict, current_prices

    def train(self, df_merged: pd.DataFrame) -> MultiHorizonForecaster:
        """
        Executes end-to-end training of all multi-horizon direct models.
        """
        logger.info(f"Training Multi-Horizon {self.algorithm.upper()} Forecaster on {len(df_merged)} bars...")
        X, y_dict, _ = self.prepare_data(df_merged)
        self.forecaster.fit(X, y_dict)
        return self.forecaster

    def save_models(self, filename: str = "forecaster.pkl") -> Path:
        """Serializes trained model to disk."""
        out_path = self.model_dir / filename
        with open(out_path, "wb") as f:
            pickle.dump(self.forecaster, f)
        logger.info(f"Saved model to {out_path}")
        return out_path

    def load_models(self, filename: str = "forecaster.pkl") -> MultiHorizonForecaster:
        """Loads trained model from disk."""
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found.")
        with open(path, "rb") as f:
            self.forecaster = pickle.load(f)
        return self.forecaster

