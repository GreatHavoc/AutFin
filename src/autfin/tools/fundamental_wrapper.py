"""AutFin v7.0 — Fundamental Tools Wrapper

Wraps existing FundamentalTools class as async functions for Pydantic AI agents.
"""
from __future__ import annotations

from typing import Any

from autfin.tools.fundamental_tools import FundamentalTools

# Global singleton
_fundamental_instance: FundamentalTools | None = None


def get_fundamental_instance() -> FundamentalTools:
    """Get or create FundamentalTools singleton."""
    global _fundamental_instance
    if _fundamental_instance is None:
        _fundamental_instance = FundamentalTools()
    return _fundamental_instance


async def get_fundamental_data(symbol: str, as_of: str | None = None) -> dict[str, Any]:
    """Get fundamental data for a symbol."""
    tools = get_fundamental_instance()
    return await tools.get_financials(symbol)


async def get_ratios(symbol: str, as_of: str | None = None) -> dict[str, float]:
    """Get clean financial ratios for a symbol."""
    tools = get_fundamental_instance()
    return await tools.get_ratios(symbol)


async def compute_health_score(symbol: str, as_of: str | None = None) -> dict[str, Any]:
    """Compute financial health score for a symbol."""
    tools = get_fundamental_instance()
    return await tools.compute_health_score(symbol)


async def get_upcoming_catalysts(symbol: str, as_of: str | None = None) -> list[str]:
    """Get upcoming catalysts for a symbol (earnings, events, etc.)."""
    # This would integrate with a calendar API
    # For now, return placeholder
    return [
        f"{symbol}: Quarterly earnings expected within 2 weeks",
        f"{symbol}: Board meeting scheduled for dividend announcement",
    ]