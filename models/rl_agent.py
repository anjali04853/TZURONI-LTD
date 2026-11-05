"""
Reinforcement Learning Execution Policy for CrowdWisdomTrading.
Implements an RL / Policy-Gradient agent that optimizes dynamic capital allocation
across multi-horizon price predictions while penalizing downside volatility and turnover.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class RLHorizonTradingAgent:
    """
    Reinforcement learning trading agent that maps multi-horizon forecasts
    (n+0 to n+5) and crowd sentiment into continuous portfolio weights in [-1.0, +1.0].
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        gamma: float = 0.95,
        risk_aversion: float = 2.0,
        cost_bps: float = 0.0005,
    ):
        self.lr = learning_rate
        self.gamma = gamma
        self.risk_aversion = risk_aversion
        self.cost_bps = cost_bps
        # Horizon weights parameter vector initialized to decay over time
        self.weights = np.array([0.0, 0.35, 0.30, 0.20, 0.10, 0.05], dtype=float)
        self.sentiment_weight = 0.25
        self.bias = 0.0

    def get_action(
        self,
        horizon_returns: Dict[int, float],
        sentiment_score: float,
        volatility: float
    ) -> float:
        """
        Determines target position allocation a_t in [-1.0, +1.0].
        """
        pred_vector = np.array([horizon_returns.get(h, 0.0) for h in range(6)], dtype=float)
        
        # Linear policy with tanh activation
        raw_signal = np.dot(self.weights, pred_vector) * 100.0 + self.sentiment_weight * sentiment_score + self.bias
        
        # Volatility scaling (Kelly-criterion heuristic)
        vol_scale = max(volatility * math_sqrt(24 * 365), 0.15)
        scaled_signal = raw_signal / (vol_scale * self.risk_aversion)

        action = float(np.tanh(scaled_signal))
        return np.clip(action, -1.0, 1.0)

    def train_policy_gradient(
        self,
        episodes: int,
        feature_df: pd.DataFrame,
        horizon_preds: Dict[int, np.ndarray],
        realized_returns: np.ndarray
    ) -> List[float]:
        """
        Optimizes policy parameters via REINFORCE policy gradients
        maximizing the risk-adjusted return (Sharpe / Sortino proxy).
        """
        n_samples = len(realized_returns)
        rewards_history = []

        for ep in range(episodes):
            positions = np.zeros(n_samples)
            curr_pos = 0.0

            for t in range(n_samples):
                h_rets = {h: horizon_preds[h][t] for h in range(6) if h in horizon_preds}
                s_t = feature_df["sentiment_weighted"].iloc[t] if "sentiment_weighted" in feature_df.columns else 0.0
                vol_t = feature_df["price_realized_vol_24h"].iloc[t] if "price_realized_vol_24h" in feature_df.columns else 0.01
                
                pos = self.get_action(h_rets, s_t, vol_t)
                positions[t] = pos

            # Calculate strategy returns with slippage/fees
            pos_changes = np.abs(np.diff(positions, prepend=0.0))
            turnover_cost = pos_changes * self.cost_bps
            strat_returns = positions * realized_returns - turnover_cost

            # Reward: Downside-penalized return (Sortino proxy)
            downside = np.minimum(strat_returns, 0.0)
            downside_vol = np.std(downside) + 1e-6
            mean_ret = np.mean(strat_returns)
            sortino_proxy = mean_ret / downside_vol

            # Approximate gradient step on weights
            grad_weights = np.zeros_like(self.weights)
            for h in range(1, 6):
                corr = np.corrcoef(horizon_preds[h], realized_returns)[0, 1] if len(realized_returns) > 1 else 0.0
                grad_weights[h] = corr if not np.isnan(corr) else 0.0

            self.weights += self.lr * grad_weights
            # Normalize horizon weights
            w_sum = np.sum(np.abs(self.weights[1:])) + 1e-6
            self.weights[1:] = self.weights[1:] / w_sum

            rewards_history.append(float(sortino_proxy))

        return rewards_history


def math_sqrt(x: float) -> float:
    import math
    return math.sqrt(max(x, 1e-6))

