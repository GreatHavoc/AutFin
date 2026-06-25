"""
MacroState - Global macro indicator types
"""
from typing import TypedDict


class MacroIndicator(TypedDict):
    """Macro indicator with India impact."""
    name: str
    value: float
    change: float
    impact_on_india: float          # -1.0 to +1.0


class MacroSignals(TypedDict):
    """Global macro-to-India impact analysis."""
    timestamp: str
    indicators: dict[str, MacroIndicator]
    composite_india_impact: float   # -1.0 (bearish) to +1.0 (bullish)
    sector_impacts: dict[str, float]  # {"IT": +0.5, "OIL_GAS": -0.3}
    narrative: str                  # 2-3 sentence summary