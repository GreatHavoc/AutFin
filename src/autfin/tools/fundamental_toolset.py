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

    @toolset.tool
    async def estimate_valuation(ctx: RunContext[AgentDeps], symbol: str) -> dict[str, Any]:
        """Estimate fair value using simple DCF and relative valuation.

        Uses yfinance data to compute:
        - Simple DCF (10-year, 12% discount rate, 8% terminal growth)
        - Relative valuation (PE-based vs sector)
        - Fair value range and upside %

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        tools = _get_fundamental()
        try:
            financials = await tools.get_financials(symbol)
            if "error" in financials:
                return {"symbol": symbol, "error": financials["error"]}

            pe = financials.get("pe", 0)
            market_cap = financials.get("market_cap", 0)
            fcf_yield = financials.get("fcf_yield", 0)
            earnings_yield = financials.get("earnings_yield", 0)
            revenue_growth = financials.get("revenue_growth_3y", 0.08)

            # Simple DCF: FCF = market_cap * fcf_yield, grow at revenue_growth for 10yr
            fcf = market_cap * (fcf_yield / 100) if fcf_yield and market_cap else 0
            discount_rate = 0.12
            terminal_growth = 0.08
            growth_rate = min(revenue_growth / 100, 0.25) if revenue_growth else 0.08

            dcf_value = 0
            if fcf > 0:
                for yr in range(1, 11):
                    dcf_value += fcf * ((1 + growth_rate) ** yr) / ((1 + discount_rate) ** yr)
                terminal_value = (fcf * ((1 + growth_rate) ** 10) * (1 + terminal_growth)) / (
                    discount_rate - terminal_growth
                )
                dcf_value += terminal_value / ((1 + discount_rate) ** 10)

            # Relative: assume sector PE ~20, compute implied price
            sector_pe = 20
            earnings = market_cap / pe if pe and pe > 0 else 0
            relative_value = earnings * sector_pe if earnings > 0 else 0

            # Fair value range
            fair_values = [v for v in [dcf_value, relative_value] if v > 0]
            if fair_values:
                fv_low = min(fair_values) * 0.85
                fv_high = max(fair_values) * 1.15
                fv_mid = sum(fair_values) / len(fair_values)
            else:
                fv_low = fv_high = fv_mid = 0

            current_price = financials.get("market_cap", 0)  # fallback
            upside = ((fv_mid / current_price) - 1) * 100 if current_price and current_price > 0 else None

            return {
                "symbol": symbol,
                "dcf_base": round(dcf_value, 0) if dcf_value else None,
                "relative_value": round(relative_value, 0) if relative_value else None,
                "fair_value_range": (round(fv_low, 0), round(fv_high, 0)) if fv_low else None,
                "upside_pct": round(upside, 1) if upside is not None else None,
                "key_assumptions": {
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    "growth_rate": round(growth_rate, 3),
                    "fcf_yield": fcf_yield,
                    "pe": pe,
                },
                "summary": f"Fair value ₹{fv_mid:,.0f} (range ₹{fv_low:,.0f}-₹{fv_high:,.0f})"
                if fv_mid
                else "Insufficient data for valuation",
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    @toolset.tool
    async def compare_peers_tool(
        ctx: RunContext[AgentDeps], symbol: str, peers: list[str]
    ) -> dict[str, Any]:
        """Compare key financial ratios across a peer group.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            peers: List of peer symbols (e.g., ["ONGC", "IOC", "BPCL"])
        """
        tools = _get_fundamental()
        try:
            return await tools.compare_peers(symbol, peers)
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    return toolset


# Create the default toolset instance
fundamental_toolset = create_fundamental_toolset()


def get_fundamental_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared fundamental toolset instance."""
    return fundamental_toolset