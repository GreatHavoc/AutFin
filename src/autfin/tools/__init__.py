"""Tools module for AutFin agent capabilities.

Uses lazy imports to avoid circular dependencies and reduce startup time.
All tool classes and functions are available via module-level attributes.
"""

# Lazy import map: attribute name -> (module_path, attribute_name)
_LAZY_IMPORTS = {
    # Kite tools
    "KiteTools": ("src.autfin.tools.kite_tools", "KiteTools"),
    "create_kite_tools": ("src.autfin.tools.kite_tools", "create_kite_tools"),
    # Sentiment tools
    "SentimentTools": ("src.autfin.tools.sentiment_tools", "SentimentTools"),
    # Memory tools
    "MemoryTools": ("src.autfin.tools.memory_tools", "MemoryTools"),
    # Fundamental tools
    "FundamentalTools": ("src.autfin.tools.fundamental_tools", "FundamentalTools"),
    # Scraper tools
    "ScraperTools": ("src.autfin.tools.scraper_tools", "ScraperTools"),
    # Technical indicators
    "calculate_sma": ("src.autfin.tools.technical_tools", "calculate_sma"),
    "calculate_ema": ("src.autfin.tools.technical_tools", "calculate_ema"),
    "calculate_rsi": ("src.autfin.tools.technical_tools", "calculate_rsi"),
    "calculate_macd": ("src.autfin.tools.technical_tools", "calculate_macd"),
    "calculate_bollinger_bands": ("src.autfin.tools.technical_tools", "calculate_bollinger_bands"),
    "calculate_atr": ("src.autfin.tools.technical_tools", "calculate_atr"),
    "calculate_obv": ("src.autfin.tools.technical_tools", "calculate_obv"),
    "calculate_vwap": ("src.autfin.tools.technical_tools", "calculate_vwap"),
    "detect_support_resistance": ("src.autfin.tools.technical_tools", "detect_support_resistance"),
    "detect_crossover": ("src.autfin.tools.technical_tools", "detect_crossover"),
    "calculate_stochastic": ("src.autfin.tools.technical_tools", "calculate_stochastic"),
    "fetch_technical_indicators": ("src.autfin.tools.technical_tools", "fetch_technical_indicators"),
    "compute_custom_indicator": ("src.autfin.tools.technical_tools", "compute_custom_indicator"),
    "detect_patterns": ("src.autfin.tools.technical_tools", "detect_patterns"),
    # Screening tools
    "NIFTY_500_SYMBOLS": ("src.autfin.tools.screening", "NIFTY_500_SYMBOLS"),
    "liquidity_filter": ("src.autfin.tools.screening", "liquidity_filter"),
    "magic_formula_rank": ("src.autfin.tools.screening", "magic_formula_rank"),
    "piotroski_f_score": ("src.autfin.tools.screening", "piotroski_f_score"),
    "indian_red_flags": ("src.autfin.tools.screening", "indian_red_flags"),
    "fetch_yfinance_bulk": ("src.autfin.tools.screening", "fetch_yfinance_bulk"),
    "run_screening_pipeline": ("src.autfin.tools.screening", "run_screening_pipeline"),
    "detect_market_regime": ("src.autfin.tools.screening", "detect_market_regime"),
    "get_fscore_threshold": ("src.autfin.tools.screening", "get_fscore_threshold"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):
    """Lazy import: load module on first attribute access."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return available attributes for tab-completion."""
    return __all__
