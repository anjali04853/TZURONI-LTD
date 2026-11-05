"""
Sentiment Analyzer Module for CrowdWisdomTrading.
Implements a specialized Financial & Crypto Social Sentiment Engine (FinLexicon)
calibrated with market vernacular, engagement weighting, and polarity scoring.
"""

from __future__ import annotations
import math
import re
from typing import Dict, Any, List, Tuple
import pandas as pd


# Financial & Trading Social Lexicon calibrated to market sentiment
FINANCIAL_LEXICON = {
    # High Bullish (+2.0 to +3.5)
    "breakout": 2.5, "ath": 3.0, "all-time-high": 3.0, "moon": 2.8, "mooning": 3.0,
    "bullish": 2.6, "bull": 1.8, "rally": 2.2, "rallying": 2.4, "surge": 2.5,
    "surging": 2.5, "pump": 2.0, "pumping": 2.2, "accumulate": 2.0, "accumulating": 2.2,
    "accumulation": 2.2, "inflows": 2.0, "inflow": 1.8, "squeeze": 2.2, "short squeeze": 3.0,
    "hodl": 1.8, "diamond hands": 2.5, "generational wealth": 2.8, "long": 1.5,
    "buy": 1.5, "buying": 1.8, "bought": 1.4, "reversal upward": 2.2, "oversold": 1.8,
    "undervalued": 2.0, "support holding": 2.2, "support holds": 2.2, "higher high": 2.0,
    "golden cross": 2.8, "parabolic": 2.8, "rocket": 2.5, "gem": 2.0, "win": 1.5,
    "green candle": 2.0, "god candle": 3.5, "etf approved": 3.2, "institutional adoption": 2.6,

    # Moderate Bullish (+0.5 to +1.5)
    "gain": 1.2, "gains": 1.4, "profit": 1.4, "profitable": 1.4, "solid": 1.0,
    "strong": 1.2, "strength": 1.2, "promising": 1.0, "optimistic": 1.2, "rebound": 1.5,
    "bounce": 1.4, "bouncing": 1.5, "recovery": 1.4, "break upward": 1.8,

    # High Bearish (-2.0 to -3.5)
    "dump": -2.8, "dumping": -3.0, "crash": -3.2, "crashing": -3.2, "bearish": -2.6,
    "bear": -1.8, "breakdown": -2.5, "breaking down": -2.6, "flash crash": -3.5,
    "rug": -3.5, "rugpull": -3.5, "scam": -3.0, "selloff": -2.5, "selling": -1.8,
    "capitulation": -3.0, "liquidated": -2.8, "liquidations": -2.6, "long squeeze": -2.8,
    "short": -1.5, "shorting": -1.8, "fud": -1.8, "panic": -2.5, "panic selling": -3.0,
    "paper hands": -2.0, "lower low": -2.0, "death cross": -2.8, "reject": -2.0,
    "rejected": -2.2, "rejection": -2.2, "resistance holding": -1.8, "overbought": -1.8,
    "overvalued": -2.0, "plummet": -2.8, "drop": -1.6, "dropping": -1.8, "collapse": -3.2,
    "bleeding": -2.4, "red candle": -1.8, "rekt": -3.0, "loss": -1.6, "losses": -1.8,

    # Moderate Bearish (-0.5 to -1.5)
    "down": -1.0, "weak": -1.2, "weakness": -1.4, "cautious": -0.8, "risk": -0.8,
    "risky": -1.0, "headwind": -1.2, "headwinds": -1.4, "concern": -1.0, "worried": -1.2,
    "dip": -0.6, "slowdown": -1.0, "stagnant": -0.6
}

# Negation and Booster modifiers
NEGATION_WORDS = {"not", "never", "no", "cant", "cannot", "won't", "wont", "hardly", "barely", "scarcely", "neither", "nor"}
BOOSTER_WORDS = {
    "extremely": 1.5, "very": 1.3, "massively": 1.6, "huge": 1.4, "super": 1.3,
    "heavily": 1.4, "insanely": 1.5, "hard": 1.3, "highly": 1.3, "slightly": 0.6, "somewhat": 0.7
}


class SentimentAnalyzer:
    """Quantitative Sentiment Engine for financial social posts and news headlines."""

    def __init__(self, use_finbert_if_available: bool = False):
        self.use_finbert = use_finbert_if_available
        self.lexicon = FINANCIAL_LEXICON
        self._emoji_map = {
            "🚀": 2.5, "🔥": 1.5, "📈": 2.0, "💎": 2.0, "🟢": 1.5, "🐂": 2.0,
            "📉": -2.0, "🛑": -2.0, "⚠️": -1.5, "💀": -2.5, "🩸": -2.5, "🐻": -2.0, "🚨": -1.5
        }

    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Computes normalized sentiment score in [-1.0, +1.0], confidence, and polarity.
        Handles negations ('not bullish' -> bearish), boosters ('extremely bearish'),
        and financial emojis.
        """
        if not text or not isinstance(text, str):
            return {"sentiment": 0.0, "confidence": 0.0, "polarity": "NEUTRAL"}

        # Tokenize and preserve emojis
        words = re.findall(r"[\w'-]+|[^\w\s]", text.lower())
        total_score = 0.0
        word_count = 0
        hits = 0

        # Check emojis in raw text
        for emo, val in self._emoji_map.items():
            count = text.count(emo)
            if count > 0:
                total_score += val * count
                hits += count

        i = 0
        n = len(words)
        while i < n:
            token = words[i]
            
            # Check 2-word ngrams first (e.g., 'short squeeze', 'death cross')
            if i < n - 1:
                bigram = f"{token} {words[i+1]}"
                if bigram in self.lexicon:
                    score = self.lexicon[bigram]
                    # Check negation in preceding 2 tokens
                    negated = any(words[max(0, i-2):i][k] in NEGATION_WORDS for k in range(len(words[max(0, i-2):i])))
                    if negated:
                        score *= -0.75
                    total_score += score
                    hits += 1
                    i += 2
                    continue

            if token in self.lexicon:
                score = self.lexicon[token]
                # Check negation
                negated = any(words[max(0, i-2):i][k] in NEGATION_WORDS for k in range(len(words[max(0, i-2):i])))
                if negated:
                    score *= -0.75
                # Check booster
                booster = 1.0
                if i > 0 and words[i-1] in BOOSTER_WORDS:
                    booster = BOOSTER_WORDS[words[i-1]]
                total_score += score * booster
                hits += 1

            i += 1

        if hits == 0:
            return {"sentiment": 0.0, "confidence": 0.1, "polarity": "NEUTRAL"}

        # Normalize via hyperbolic tangent
        norm_score = math.tanh(total_score / 3.0)
        norm_score = max(min(norm_score, 1.0), -1.0)
        confidence = min(1.0, 0.3 + 0.15 * hits)

        if norm_score > 0.15:
            polarity = "BULLISH"
        elif norm_score < -0.15:
            polarity = "BEARISH"
        else:
            polarity = "NEUTRAL"

        return {
            "sentiment": round(norm_score, 4),
            "confidence": round(confidence, 3),
            "polarity": polarity,
            "raw_hits": hits
        }

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Enriches a DataFrame of tweets/news with sentiment scores and engagement weights.
        """
        if df.empty or text_column not in df.columns:
            return df

        df_out = df.copy()
        sentiments = []

        for txt in df_out[text_column]:
            res = self.score_text(str(txt))
            sentiments.append(res["sentiment"])

        df_out["sentiment_score"] = sentiments

        # Recalculate engagement-weighted sentiment score if engagement columns exist
        if "likes_count" in df_out.columns and "retweets_count" in df_out.columns:
            likes = df_out["likes_count"].fillna(0).astype(float)
            retweets = df_out["retweets_count"].fillna(0).astype(float)
            views = df_out.get("views_count", pd.Series(0, index=df_out.index)).fillna(0).astype(float)
            followers = df_out.get("author_followers_count", pd.Series(100, index=df_out.index)).fillna(100).astype(float)

            # Compute engagement multiplier
            weights = (
                1.0 +
                np_log_safe(likes) * 1.0 +
                np_log_safe(retweets) * 2.0 +
                np_log_safe(views) * 0.2 +
                np_log_safe(followers) * 0.5
            )
            df_out["engagement_weight"] = weights
            df_out["weighted_sentiment"] = df_out["sentiment_score"] * weights
        else:
            df_out["engagement_weight"] = 1.0
            df_out["weighted_sentiment"] = df_out["sentiment_score"]

        return df_out


def np_log_safe(series: pd.Series) -> pd.Series:
    """Safe natural log handling non-negative numbers."""
    import numpy as np
    return np.log1p(np.maximum(series, 0))

