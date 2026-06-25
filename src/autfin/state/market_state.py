"""
MarketSnapshot - Market data types
"""
from typing import TypedDict


class PriceQuote(TypedDict):
    """Price quote for a stock."""
    symbol: str
    ltp: float                      # Last traded price
    change: float                   # Absolute change
    change_pct: float               # Percentage change
    volume: int
    avg_volume: float               # 20-day average
    high_52w: float
    low_52w: float


class MarketSnapshot(TypedDict):
    """Live market data snapshot."""
    timestamp: str                  # ISO 8601
    indices: dict[str, float]       # {"NIFTY 50": 24500.0, "SENSEX": 81000.0}
    quotes: dict[str, PriceQuote]   # {symbol: quote}
    advance_decline: dict           # Market breadth