"""Tools module for AutFin agent capabilities."""

from src.autfin.tools.kite_tools import KiteTools, create_kite_tools
from src.autfin.tools.sentiment_tools import SentimentTools
from src.autfin.tools.memory_tools import MemoryTools
from src.autfin.tools.fundamental_tools import FundamentalTools
from src.autfin.tools.technical_tools import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_obv,
    calculate_vwap,
    detect_support_resistance,
    detect_crossover,
    calculate_stochastic,
    fetch_technical_indicators,
    compute_custom_indicator,
    detect_patterns,
)
from src.autfin.tools.scraper_tools import ScraperTools
from src.autfin.tools.screening import (
    NIFTY_500_SYMBOLS,
    liquidity_filter,
    magic_formula_rank,
    piotroski_f_score,
    indian_red_flags,
    fetch_yfinance_bulk,
    run_screening_pipeline,
    detect_market_regime,
    get_fscore_threshold,
)

__all__ = [
    "KiteTools",
    "create_kite_tools",
    "SentimentTools",
    "MemoryTools",
    "FundamentalTools",
    "ScraperTools",
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_obv",
    "calculate_vwap",
    "detect_support_resistance",
    "detect_crossover",
    "calculate_stochastic",
    "fetch_technical_indicators",
    "compute_custom_indicator",
    "detect_patterns",
    "NIFTY_500_SYMBOLS",
    "liquidity_filter",
    "magic_formula_rank",
    "piotroski_f_score",
    "indian_red_flags",
    "fetch_yfinance_bulk",
    "run_screening_pipeline",
    "detect_market_regime",
    "get_fscore_threshold",
]
