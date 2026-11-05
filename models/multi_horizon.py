"""
Multi-Horizon Predictive Model Module for CrowdWisdomTrading.
Implements Direct Multi-Horizon Forecasting for n+0, n+1, n+2, n+3, n+4, and n+5
using LightGBM, XGBoost, and Ridge with strictly sentiment and price features.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


class MultiHorizonForecaster:
    """
    Direct Multi-Horizon Regressor that forecasts future prices and returns
    at horizons h in [0, 1, 2, 3, 4, 5].
    """

    def __init__(
        self,
        algorithm: str = "lightgbm",
        horizons: Optional[List[int]] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ):
        self.algorithm = algorithm.lower()
        self.horizons = horizons if horizons is not None else [0, 1, 2, 3, 4, 5]
        self.hyperparams = hyperparams or {}
        self.models: Dict[int, Any] = {}
        self.feature_names: List[str] = []
        self.feature_importances: Dict[int, pd.Series] = {}

    def _create_base_model(self, horizon: int) -> Any:
        """Instantiates algorithm estimator with horizon-appropriate defaults."""
        if self.algorithm in ("lightgbm", "histgbm", "gradient_boosting"):
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                params = {
                    "max_iter": self.hyperparams.get("n_estimators", 70),
                    "learning_rate": self.hyperparams.get("learning_rate", 0.03),
                    "max_depth": self.hyperparams.get("max_depth", 5),
                    "random_state": 42 + horizon
                }
                return HistGradientBoostingRegressor(**params)
            except Exception:
                pass

        if self.algorithm == "xgboost":
            try:
                import xgboost as xgb
                params = {
                    "n_estimators": self.hyperparams.get("n_estimators", 70),
                    "learning_rate": self.hyperparams.get("learning_rate", 0.03),
                    "max_depth": self.hyperparams.get("max_depth", 5),
                    "random_state": 42 + horizon,
                    "verbosity": 0,
                    "n_jobs": 1
                }
                return xgb.XGBRegressor(**params)
            except ImportError:
                logger.warning("XGBoost not found, falling back to Ridge.")
                self.algorithm = "ridge"

        # Standardized Ridge Regression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        alpha = self.hyperparams.get("alpha", 50.0)
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=42 + horizon))

    def fit(self, X: pd.DataFrame, y_dict: Dict[int, pd.Series]) -> MultiHorizonForecaster:
        """
        Fits separate direct estimators for each horizon to prevent error compounding.
        X: features DataFrame (sentiment and price features only)
        y_dict: dictionary mapping horizon h -> target return series
        """
        self.feature_names = list(X.columns)

        for h in self.horizons:
            if h == 0:
                continue
            if h not in y_dict:
                continue

            y_h = y_dict[h]
            # Drop NaN pairs
            valid_mask = ~(X.isna().any(axis=1) | y_h.isna())
            X_clean = X.loc[valid_mask]
            y_clean = y_h.loc[valid_mask]

            if len(X_clean) < 10:
                logger.warning(f"Not enough samples to train horizon n+{h}")
                continue

            model = self._create_base_model(h)
            model.fit(X_clean, y_clean)
            self.models[h] = model

            # Store feature importance
            if hasattr(model, "feature_importances_"):
                self.feature_importances[h] = pd.Series(
                    model.feature_importances_, index=self.feature_names
                ).sort_values(ascending=False)
            elif hasattr(model, "named_steps") and "ridge" in model.named_steps:
                coef = np.abs(model.named_steps["ridge"].coef_)
                self.feature_importances[h] = pd.Series(
                    coef, index=self.feature_names
                ).sort_values(ascending=False)
            elif hasattr(model, "coef_"):
                self.feature_importances[h] = pd.Series(
                    np.abs(model.coef_), index=self.feature_names
                ).sort_values(ascending=False)
            else:
                corrs = [
                    abs(np.corrcoef(X_clean[col], y_clean)[0, 1]) if np.std(X_clean[col]) > 0 else 0.0
                    for col in self.feature_names
                ]
                self.feature_importances[h] = pd.Series(
                    np.nan_to_num(corrs, nan=0.0), index=self.feature_names
                ).sort_values(ascending=False)

        return self

    def predict_returns(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Forecasts expected return R_{n+h} for each horizon h.
        """
        predictions: Dict[int, np.ndarray] = {}
        X_eval = X[self.feature_names].fillna(0.0)

        for h in self.horizons:
            if h == 0:
                # n+0 corresponds to the current period close
                predictions[h] = np.zeros(len(X_eval))
                continue

            if h in self.models:
                preds = self.models[h].predict(X_eval)
                predictions[h] = np.array(preds)
            else:
                predictions[h] = np.zeros(len(X_eval))

        return predictions

    def predict_prices(self, X: pd.DataFrame, current_prices: pd.Series) -> Dict[int, np.ndarray]:
        """
        Converts forecasted returns into forecasted spot prices: P_{n+h} = P_n * (1 + R_{n+h}).
        """
        ret_preds = self.predict_returns(X)
        price_preds: Dict[int, np.ndarray] = {}
        p_arr = np.asarray(current_prices)

        for h in self.horizons:
            price_preds[h] = p_arr * (1.0 + ret_preds[h])

        return price_preds

    def get_aggregate_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Summarizes feature importance across all predicted horizons.
        """
        if not self.feature_importances:
            return pd.DataFrame()

        df_all = pd.DataFrame(self.feature_importances)
        # Normalize per horizon to sum to 100%
        df_norm = df_all.div(df_all.sum(axis=0) + 1e-9, axis=1) * 100.0
        df_norm["mean_importance"] = df_norm.mean(axis=1)
        df_norm.sort_values(by="mean_importance", ascending=False, inplace=True)
        return df_norm.head(top_n)

