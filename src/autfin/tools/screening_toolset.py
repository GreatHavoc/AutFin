"""AutFin v7.0 — Screening Toolset

Wraps screening pipeline as a reusable FunctionToolset for Pydantic AI agents.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, RunContext

from autfin.tools.fundamental_tools import FundamentalTools
from autfin.tools.screening import (
    NIFTY_500_SYMBOLS,
    apply_sector_rotation,
    indian_red_flags,
    liquidity_filter,
    magic_formula_rank,
)
from autfin.types import AgentDeps


def create_screening_toolset() -> FunctionToolset[AgentDeps]:
    """Create a FunctionToolset with all screening tools."""
    toolset = FunctionToolset[AgentDeps]()
    
    # Fundamental tools instance for fetching data
    fundamental_tools = FundamentalTools()
    
    @toolset.tool
    async def screen_stocks(
        ctx: RunContext[AgentDeps],
        criteria: dict[str, Any] | None = None,
        sector: str | None = None,
        max_results: int = 20,
        universe: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Screen stocks based on fundamental and technical criteria.
        
        Args:
            criteria: Optional dict of screening criteria
            sector: Optional sector filter
            max_results: Maximum number of results to return
            universe: Optional custom universe of symbols
        """
        # Use NIFTY 500 as default universe
        symbols = universe or NIFTY_500_SYMBOLS
        
        # Fetch fundamental data for all symbols
        stocks_data = []
        for symbol in symbols:
            try:
                data = await fundamental_tools.get_financials(symbol)
                if "error" not in data:
                    data["symbol"] = symbol
                    stocks_data.append(data)
            except Exception:
                continue
        
        # Apply filters
        filtered = liquidity_filter(stocks_data)
        filtered = [s for s in filtered if not indian_red_flags(s)]
        
        # Apply ranking
        ranked = magic_formula_rank(filtered)
        # Note: piotroski_f_score needs prior year data - simplified for now
        ranked = apply_sector_rotation(ranked)
        
        # Sector filter
        if sector:
            ranked = [s for s in ranked if s.get("sector") == sector]
        
        return ranked[:max_results]

    @toolset.tool
    async def get_nifty500_universe(ctx: RunContext[AgentDeps]) -> list[str]:
        """Get the NIFTY 500 universe symbols."""
        return NIFTY_500_SYMBOLS

    @toolset.tool
    async def apply_liquidity_filter(
        ctx: RunContext[AgentDeps],
        stocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply liquidity filter (volume >= 1Cr, market cap >= 500Cr)."""
        return liquidity_filter(stocks)

    @toolset.tool
    async def apply_magic_formula(
        ctx: RunContext[AgentDeps],
        stocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply Magic Formula ranking (Earnings Yield + ROIC)."""
        return magic_formula_rank(stocks)

    @toolset.tool
    async def apply_piotroski(
        ctx: RunContext[AgentDeps],
        stocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply Piotroski F-Score (0-9) - requires prior year data."""
        # This would need prior year financials
        # For now, return stocks with placeholder
        for s in stocks:
            s["f_score"] = 0  # placeholder
        return stocks

    @toolset.tool
    async def apply_red_flags(
        ctx: RunContext[AgentDeps],
        stocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply red flags filter (promoter pledging, auditor changes, etc.)."""
        return [s for s in stocks if not indian_red_flags(s)]

    @toolset.tool
    async def apply_sector_rotation_filter(
        ctx: RunContext[AgentDeps],
        stocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply sector rotation filter."""
        return apply_sector_rotation(stocks)

    return toolset


# Create the default toolset instance
screening_toolset = create_screening_toolset()


def get_screening_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared screening toolset instance."""
    return screening_toolset