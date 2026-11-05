"""
Master Pipeline Runner for CrowdWisdomTrading X Sentiment Price Prediction Agent.
Executes the complete end-to-end workflow:
1. Ingestion of X posts (Apify), news (Tavily/Exa), and prices into SQLite database.
2. High-precision sentiment scoring with engagement weighting and time-bar aggregation.
3. Automated generation of the 1-page current sentiment outlook.
4. Quantitative feature engineering (strictly sentiment and price only).
5. Multi-horizon direct predictive modeling for n+0, n+1, n+2, n+3, n+4, n+5.
6. Purged Walk-Forward Validation with out-of-sample trading simulation.
7. Calculation of Sharpe Ratio, Sortino Ratio, Max Drawdown, and Horizon Accuracies.
8. Visual Matrix output generation (static PNG & interactive HTML).
9. Comprehensive Evaluation Report compilation (EVALUATION_REPORT.md).
"""

from __future__ import annotations
import os
import sys

# Prevent OpenMP and multiprocessing deadlocks in sandboxed Windows execution
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add workspace to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from data.database import DatabaseManager
from data.data_loader import UnifiedDataLoader
from sentiment.analyzer import SentimentAnalyzer
from sentiment.aggregator import SentimentAggregator
from sentiment.summary_generator import SentimentSummaryGenerator
from features.engineer import QuantFeatureEngineer
from models.trainer import ModelTrainer
from backtest.walk_forward import WalkForwardValidator
from visualization.matrix_generator import VisualMatrixGenerator
from reports.generate_report import EvaluationReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CWT-Pipeline")


def run_end_to_end_pipeline(
    symbol: str = "BTC-USD",
    lookback_days: int = 180,
    timeframe: str = "1h",
    algorithm: str = "lightgbm",
    use_cache: bool = True
) -> None:
    """Executes the entire quantitative pipeline from raw scraping to final report."""
    logger.info("=" * 80)
    logger.info(f"CROWDWISDOMTRADING: X SENTIMENT PRICE PREDICTION AGENT")
    logger.info(f"Asset: {symbol} | Timeframe: {timeframe} | Lookback: {lookback_days} days | Model: {algorithm.upper()}")
    logger.info("=" * 80)

    db = DatabaseManager()
    loader = UnifiedDataLoader(db)

    # -------------------------------------------------------------
    # Step 1: Ingestion (Apify X Posts, Tavily News, Price Bars)
    # -------------------------------------------------------------
    logger.info("\n[STEP 1/7] Ingesting and Joining Datasets in Database...")
    ingest_stats = loader.run_full_ingestion(
        symbol=symbol,
        lookback_days=lookback_days,
        timeframe=timeframe,
        use_cache=use_cache
    )
    logger.info(f"Ingestion complete: {ingest_stats}")

    raw_tweets = db.load_tweets(symbol)
    price_bars = db.load_price_bars(symbol, timeframe)
    recent_news = db.load_news(symbol)

    if price_bars.empty:
        raise RuntimeError("No price bars found in database. Check price ingestion.")

    # -------------------------------------------------------------
    # Step 2: Sentiment Scoring & Engagement-Weighted Aggregation
    # -------------------------------------------------------------
    logger.info("\n[STEP 2/7] Scoring Sentiment & Calculating Engagement Weights...")
    analyzer = SentimentAnalyzer()
    scored_tweets = analyzer.process_dataframe(raw_tweets, text_column="text")
    
    # Update scored tweets in database
    db.store_tweets(scored_tweets)

    aggregator = SentimentAggregator(timeframe=timeframe)
    sent_aggregated = aggregator.aggregate_tweets(scored_tweets)
    merged_df = aggregator.merge_sentiment_and_prices(sent_aggregated, price_bars)
    logger.info(f"Merged aligned dataset created: {len(merged_df)} time bars.")

    # -------------------------------------------------------------
    # Step 3: 1-Page Current Sentiment Outlook Summary
    # -------------------------------------------------------------
    logger.info("\n[STEP 3/7] Generating 1-Page Current Sentiment Outlook Summary...")
    summary_gen = SentimentSummaryGenerator()
    current_price = float(merged_df["close"].iloc[-1])
    top_engagement_tweets = scored_tweets.sort_values(by="engagement_score", ascending=False).head(5)
    
    summary_text = summary_gen.generate_outlook(
        asset_symbol=symbol,
        current_price=current_price,
        recent_sentiment_df=merged_df.tail(72),
        top_tweets=top_engagement_tweets,
        recent_news=recent_news
    )
    logger.info("1-Page Sentiment Outlook saved to: SENTIMENT_OUTLOOK_1PAGE.md")

    # -------------------------------------------------------------
    # Step 4: Quantitative Feature Engineering (Sentiment & Price ONLY)
    # -------------------------------------------------------------
    logger.info("\n[STEP 4/7] Engineering Quant Features strictly from Sentiment & Price...")
    feat_engineer = QuantFeatureEngineer()
    df_feat = feat_engineer.engineer_features(merged_df)
    feat_names = feat_engineer.get_feature_column_names(df_feat)
    logger.info(f"Engineered {len(feat_names)} pure sentiment-and-price alpha features.")

    # -------------------------------------------------------------
    # Step 5: Multi-Horizon Modeling & Purged Walk-Forward Validation
    # -------------------------------------------------------------
    logger.info("\n[STEP 5/7] Executing Walk-Forward Validation for n+0 to n+5...")
    validator = WalkForwardValidator(
        algorithm=algorithm,
        train_window_bars=min(1440, int(len(df_feat) * 0.45)),  # 60 days
        test_step_bars=168,                                      # 7 days
        embargo_bars=6,                                          # Embargo equal to max horizon
        horizons=settings.HORIZONS,
        commission_bps=settings.COMMISSION_BPS,
        slippage_bps=settings.SLIPPAGE_BPS
    )

    wf_results = validator.run_validation(merged_df)
    perf = wf_results["performance_summary"]
    h_metrics = wf_results["horizon_metrics"]
    oos_df = wf_results["oos_predictions_df"]

    logger.info("\n>>> OUT-OF-SAMPLE PERFORMANCE METRICS <<<")
    logger.info(f"Sharpe Ratio:        {perf['sharpe_ratio']:.3f} (Benchmark: {perf['benchmark_sharpe']:.3f})")
    logger.info(f"Sortino Ratio:       {perf['sortino_ratio']:.3f}")
    logger.info(f"Maximum Drawdown:    {perf['max_drawdown_pct']:.2f}%")
    logger.info(f"Calmar Ratio:        {perf['calmar_ratio']:.3f}")
    logger.info(f"Strategy Return:     {perf['cumulative_return_pct']:.2f}% (Benchmark: {perf['benchmark_cumulative_pct']:.2f}%)")
    logger.info(f"Win Rate:            {perf['win_rate_pct']:.2f}%")
    logger.info(f"Profit Factor:       {perf['profit_factor']:.2f}")

    # -------------------------------------------------------------
    # Step 6: Visual Matrix Output Generation
    # -------------------------------------------------------------
    logger.info("\n[STEP 6/7] Generating Visual Matrix Charts...")
    viz_gen = VisualMatrixGenerator()
    viz_path = viz_gen.generate_static_matrix(merged_df, oos_df, asset_symbol=symbol)
    fan_path = viz_gen.generate_horizon_fan_chart(oos_df)
    html_path = viz_gen.generate_interactive_html(merged_df, oos_df, asset_symbol=symbol)
    logger.info(f"Visual Matrix PNG saved: {viz_path}")
    logger.info(f"Horizon Fan Chart saved: {fan_path}")
    logger.info(f"Interactive HTML saved:   {html_path}")

    # -------------------------------------------------------------
    # Step 7: Comprehensive Evaluation Report & Final Outlook
    # -------------------------------------------------------------
    logger.info("\n[STEP 7/7] Compiling Formal Evaluation Report...")
    report_gen = EvaluationReportGenerator()
    trainer = ModelTrainer(algorithm=algorithm)
    trainer.train(merged_df)
    feat_imp = trainer.forecaster.get_aggregate_feature_importance(top_n=12)

    # Refresh 1-page summary with actual trained model forecasts
    latest_feat_row = df_feat[feat_names].iloc[-1:]
    latest_price_preds = trainer.forecaster.predict_prices(latest_feat_row, pd.Series([current_price]))
    horizon_forecasts = {f"n+{h}": float(latest_price_preds[h][0]) for h in settings.HORIZONS}
    summary_gen.generate_outlook(
        asset_symbol=symbol,
        current_price=current_price,
        recent_sentiment_df=merged_df.tail(72),
        top_tweets=top_engagement_tweets,
        recent_news=recent_news,
        horizon_forecasts=horizon_forecasts
    )

    report_path = report_gen.generate_markdown_report(
        asset_symbol=symbol,
        perf_summary=perf,
        horizon_metrics=h_metrics,
        feature_importance_df=feat_imp,
        ingest_stats=ingest_stats,
        total_oos_bars=len(oos_df)
    )
    logger.info(f"Evaluation report generated: {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"1. Repository & Codebase: Ready for submission to gilad@crowdwisdomtrading.com")
    logger.info(f"2. 1-Page Sentiment Outlook: {summary_gen.output_path}")
    logger.info(f"3. Evaluation Report: {report_path}")
    logger.info(f"4. Visual Matrix: {viz_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrowdWisdomTrading X Sentiment Prediction Pipeline")
    parser.add_argument("--asset", type=str, default="BTC-USD", help="Asset ticker (e.g. BTC-USD, NVDA, TSLA)")
    parser.add_argument("--days", type=int, default=180, help="Lookback window in days (default: 180)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Bar resolution: 1h or 1d")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "ridge"], help="Prediction algorithm")
    parser.add_argument("--no-cache", action="store_true", help="Force re-fetch/re-computation")

    args = parser.parse_args()
    run_end_to_end_pipeline(
        symbol=args.asset,
        lookback_days=args.days,
        timeframe=args.timeframe,
        algorithm=args.model,
        use_cache=not args.no_cache
    )

