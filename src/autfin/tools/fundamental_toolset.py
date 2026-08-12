"""AutFin v7.0 — Fundamental Toolset

Wraps existing FundamentalTools as a reusable FunctionToolset for Pydantic AI agents.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, RunContext

from autfin.tools.fundamental_tools import FundamentalTools
from autfin.types import AgentDeps


def create_fundamental_toolset() -> FunctionToolset[AgentDeps]:
    """Create a FunctionToolset with all fundamental analysis tools."""
    toolset = FunctionToolset[AgentDeps]()
    
    # Get FundamentalTools instance
    def _get_fundamental() -> FundamentalTools:
        return FundamentalTools()
    
    @toolset.tool
    async def get_financials(ctx: RunContext[AgentDeps], symbol: str) -> dict[str, Any]:
        """Fetch financial statements from yfinance and compute key ratios.
        
        Returns: pe, pb, roe, roic, debt_equity, current_ratio, fcf_yield,
        earnings_yield, dividend_yield, market_cap, enterprise_value,
        revenue_growth_3y, profit_growth_3y, promoter_holding, pledged_pct.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        tools = _get_fundamental()
        return await tools.get_financials(symbol)

    @toolset.tool
    async def get_ratios(ctx: RunContext[AgentDeps], symbol: str) -> dict[str, float]:
        """Get clean financial ratios dict. Returns only numeric values.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        tools = _get_fundamental()
        return await tools.get_ratios(symbol)

    @toolset.tool
    async def compute_health_score(ctx: RunContext[AgentDeps], symbol: str) -> dict[str, Any]:
        """Compute a composite financial health score (0-100).
        
        Components (each 0-20 points):
        - Profitability (ROE > 15% = 20pts, 10-15% = 15pts, 0-10% = 10pts, negative = 0pts)
        - Debt management (D/E < 0.3 = 20pts, 0.3-0.7 = 15pts, 0.7-1.5 = 10pts, >1.5 = 5pts)
        - Growth (Revenue growth > 20% = 20pts, 10-20% = 15pts, 0-10% = 10pts, negative = 5pts)
        - Cash generation (FCF yield > 5% = 20pts, 2-5% = 15pts, 0-2% = 10pts, negative = 5pts)
        - Valuation (PE < 12 = 20pts, 12-18 = 15pts, 18-25 = 10pts, >25 or negative = 5pts)
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        tools = _get_fundamental()
        return await tools.compute_health_score(symbol)

    @toolset.tool
    async def get_upcoming_catalysts(ctx: RunContext[AgentDeps], symbol: str) -> list[str]:
        """Get upcoming catalysts for a symbol (earnings, events, policy).
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        # Placeholder - would integrate with calendar API
        return [
            f"{symbol}: Quarterly earnings expected within 2 weeks",
            f"{symbol}: Board meeting scheduled for dividend announcement",
        ]

    return toolset


# Create the default toolset instance
fundamental_toolset = create_fundamental_toolset()


def get_fundamental_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared fundamental toolset instance."""
    return fundamental_toolset