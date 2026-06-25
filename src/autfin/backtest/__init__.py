"""AutFin Backtesting Engine."""

from .engine import BacktestEngine, BacktestConfig, BacktestResult, TradeRecord, fetch_historical_data
from .metrics import BacktestMetrics, calculate_metrics
from .strategies import (
    BaseStrategy,
    StrategySignal,
    MomentumStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
    OpeningRangeBreakoutStrategy,
    BreakoutStrategy,
    get_strategy,
    STRATEGY_MAP,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "TradeRecord",
    "fetch_historical_data",
    "BacktestMetrics",
    "calculate_metrics",
    "BaseStrategy",
    "StrategySignal",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
    "OpeningRangeBreakoutStrategy",
    "BreakoutStrategy",
    "get_strategy",
    "STRATEGY_MAP",
]
