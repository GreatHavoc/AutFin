"""
DCF Valuation Agent — Professional-grade discounted cash flow analysis.

Builds Base/Upside/Downside scenarios with implied per-share price targets,
computes asymmetric risk/reward ratio, and produces a professional stock
pitch valuation section.

Architecture:
  yfinance data → scenario builder → DCF computation → implied price targets
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCF Parameters
# ---------------------------------------------------------------------------

DEFAULT_WACC = 0.10           # 10% WACC for Indian equities
DEFAULT_TERMINAL_GROWTH = 0.03  # 3% perpetual growth
DEFAULT_BETA = 1.0
DEFAULT_PROJECTION_YEARS = 5

# Scenario growth multipliers (applied to base revenue growth)
SCENARIO_MULTIPLIERS = {
    "upside": 1.5,    # 50% better than base
    "base": 1.0,      # Current trajectory
    "downside": 0.5,   # 50% worse than base
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Core DCF Calculation
# ---------------------------------------------------------------------------

def compute_dcf(
    free_cash_flow: float,
    revenue_growth: float,
    wacc: float = DEFAULT_WACC,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    projection_years: int = DEFAULT_PROJECTION_YEARS,
    fcf_margin: Optional[float] = None,
    revenue: Optional[float] = None,
    total_debt: float = 0.0,
    cash: float = 0.0,
    shares_outstanding: Optional[float] = None,
    scenario: str = "base",
) -> Dict[str, Any]:
    """
    Run Discounted Cash Flow valuation for one scenario.

    Uses the two-stage DCF model:
      1. Project FCF for projection_years using revenue_growth
      2. Terminal value = FCF_last * (1 + terminal_growth) / (WACC - terminal_growth)
      3. Discount all cash flows to present value
      4. Enterprise Value → Equity Value → Per-Share Value

    Args:
        free_cash_flow: Current annual free cash flow
        revenue_growth: Annual revenue growth rate (as decimal, e.g., 0.15)
        wacc: Weighted Average Cost of Capital
        terminal_growth: Perpetual growth rate
        projection_years: Number of years to project
        fcf_margin: FCF / Revenue ratio (if None, computed from inputs)
        revenue: Current annual revenue
        total_debt: Total debt
        cash: Cash & equivalents
        shares_outstanding: Diluted shares outstanding
        scenario: Label for this scenario

    Returns:
        Dict with enterprise_value, equity_value, implied_price, and projection detail
    """
    projections = []
    current_fcf = free_cash_flow if free_cash_flow > 0 else 1e6  # Default to safe value

    # Project FCF over projection period
    fcf_series = []
    prev_fcf = current_fcf
    growth = max(revenue_growth, 0.01) if revenue_growth > 0 else 0.01  # Floor growth at 1%

    for year in range(1, projection_years + 1):
        projected_fcf = prev_fcf * (1 + growth)
        fcf_series.append(projected_fcf)
        projections.append({
            "year": year,
            "fcf": round(projected_fcf, 2),
        })
        prev_fcf = projected_fcf

    # Terminal value (Gordon Growth Model)
    last_fcf = fcf_series[-1]
    discount = wacc - terminal_growth
    if discount <= 0:
        discount = 0.03  # Safety floor
    terminal_value = last_fcf * (1 + terminal_growth) / discount

    # Discount to present value
    pv_fcfs = sum(
        fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(fcf_series)
    )
    pv_terminal = terminal_value / ((1 + wacc) ** projection_years)

    enterprise_value = pv_fcfs + pv_terminal
    equity_value = enterprise_value + cash - total_debt

    implied_price = None
    if shares_outstanding and shares_outstanding > 0:
        implied_price = equity_value / shares_outstanding

    return {
        "scenario": scenario,
        "current_fcf": round(current_fcf, 2),
        "growth_rate": round(growth * 100, 1),
        "wacc": round(wacc * 100, 1),
        "terminal_growth": round(terminal_growth * 100, 1),
        "pv_projected_fcfs": round(pv_fcfs, 2),
        "pv_terminal_value": round(pv_terminal, 2),
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "implied_price": round(implied_price, 2) if implied_price else None,
        "projections": projections,
    }


# ---------------------------------------------------------------------------
# Scenario Builder
# ---------------------------------------------------------------------------

async def fetch_financials(symbol: str) -> Dict[str, Any]:
    """
    Fetch financial data needed for DCF from yfinance.

    Returns a dict with: free_cash_flow, revenue, revenue_growth, total_debt, cash,
    shares_outstanding, market_cap, current_price
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"status": "error", "error": "yfinance not installed", "data": None}

    try:
        ticker = yf.Ticker(f"{symbol.removesuffix('.NS')}.NS")
        info = ticker.info or {}

        free_cash_flow = _safe(info.get("freeCashflow"))
        if free_cash_flow <= 0:
            free_cash_flow = _safe(info.get("operatingCashflow")) * 0.7  # Estimate

        data = {
            "symbol": symbol,
            "free_cash_flow": free_cash_flow,
            "revenue": _safe(info.get("totalRevenue")),
            "revenue_growth": _safe(info.get("revenueGrowth"), 0.0),
            "operating_margin": _safe(info.get("operatingMargins"), 0.0) * 100,
            "total_debt": _safe(info.get("totalDebt")),
            "cash": _safe(info.get("totalCash")),
            "shares_outstanding": _safe(info.get("sharesOutstanding")),
            "market_cap": _safe(info.get("marketCap")),
            "current_price": _safe(info.get("currentPrice")) or _safe(info.get("regularMarketPrice")),
            "ebitda": _safe(info.get("ebitda")),
            "enterprise_value": _safe(info.get("enterpriseValue")),
        }
        return {"status": "success", "data": data}
    except Exception as exc:
        logger.error("Failed to fetch financials for %s: %s", symbol, exc)
        return {"status": "error", "error": str(exc), "data": None}


def build_scenarios(fin_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build Base, Upside, and Downside scenarios from financial data.

    Revenue growth is the primary driver. Scenarios apply multipliers to base growth.
    """
    revenue_growth = fin_data.get("revenue_growth", 0.15) or 0.15
    fcf = fin_data.get("free_cash_flow", 0) or 1e6

    scenarios = {}
    for scenario, multiplier in SCENARIO_MULTIPLIERS.items():
        scenario_growth = revenue_growth * multiplier
        scenario_fcf = fcf * multiplier if multiplier != 1.0 else fcf
        scenarios[scenario] = {
            "growth_rate": scenario_growth,
            "fcf": scenario_fcf,
        }
    return scenarios


def run_dcf_valuation(
    fin_data: Dict[str, Any],
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
    wacc: float = DEFAULT_WACC,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
) -> Dict[str, Any]:
    """
    Run DCF for all three scenarios and aggregate results.

    Returns a complete valuation report with:
    - Per-scenario implied prices
    - Risk/reward asymmetry ratio
    - Upside/downside percentages
    - Conviction score
    """
    if scenarios is None:
        scenarios = build_scenarios(fin_data)

    fcf = fin_data.get("free_cash_flow", 0) or 1e6
    total_debt = fin_data.get("total_debt", 0)
    cash = fin_data.get("cash", 0)
    shares = fin_data.get("shares_outstanding")
    current_price = fin_data.get("current_price")

    results = {}
    for scenario_name, params in scenarios.items():
        results[scenario_name] = compute_dcf(
            free_cash_flow=params["fcf"],
            revenue_growth=params["growth_rate"],
            wacc=wacc,
            terminal_growth=terminal_growth,
            total_debt=total_debt,
            cash=cash,
            shares_outstanding=shares,
            scenario=scenario_name,
        )

    # ── Aggregate analysis ────────────────────────────────────────────
    base_price = results.get("base", {}).get("implied_price")
    upside_price = results.get("upside", {}).get("implied_price")
    downside_price = results.get("downside", {}).get("implied_price")

    if not current_price or not base_price or base_price <= 0:
        return {
            "status": "error",
            "error": "Insufficient data: missing current_price or base_price",
            "scenarios": results,
        }

    # Upside / downside percentages
    upside_pct = ((upside_price - current_price) / current_price * 100) if upside_price else None
    downside_pct = ((downside_price - current_price) / current_price * 100) if downside_price else None

    # Asymmetric risk/reward ratio (how many $ upside per $ downside)
    if upside_pct is not None and downside_pct is not None:
        upside_abs = max(upside_pct, 0)
        downside_abs = max(abs(downside_pct), 1)  # Avoid division by 0
        risk_reward_ratio = upside_abs / downside_abs if downside_abs > 0 else 0
    else:
        risk_reward_ratio = None

    # Conviction score (0-100) based on margin of safety and risk/reward
    margin_of_safety = ((base_price - current_price) / current_price * 100)
    conviction = min(100, max(0,
        (margin_of_safety * 1.5 if margin_of_safety > 0 else margin_of_safety)
        + (risk_reward_ratio * 15 if risk_reward_ratio else 0)
        + 50
    ))

    # Recommendation
    if conviction >= 80 and risk_reward_ratio and risk_reward_ratio > 3:
        recommendation = "STRONG BUY"
        conviction_level = "high"
    elif conviction >= 60 and risk_reward_ratio and risk_reward_ratio > 1.5:
        recommendation = "BUY"
        conviction_level = "medium"
    elif conviction >= 40 and margin_of_safety > -10:
        recommendation = "HOLD"
        conviction_level = "low"
    else:
        recommendation = "SELL"
        conviction_level = "low"

    return {
        "status": "success",
        "symbol": fin_data.get("symbol"),
        "current_price": current_price,
        "base_price_target": round(base_price, 2),
        "upside_price_target": round(upside_price, 2) if upside_price else None,
        "downside_price_target": round(downside_price, 2) if downside_price else None,
        "upside_pct": round(upside_pct, 1) if upside_pct is not None else None,
        "downside_pct": round(downside_pct, 1) if downside_pct is not None else None,
        "margin_of_safety_pct": round(margin_of_safety, 1),
        "risk_reward_ratio": round(risk_reward_ratio, 2) if risk_reward_ratio is not None else None,
        "conviction_score": round(conviction, 0),
        "conviction_level": conviction_level,
        "recommendation": recommendation,
        "wacc_used": round(wacc * 100, 1),
        "terminal_growth_used": round(terminal_growth * 100, 1),
        "scenarios": results,
        "calculated_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Valuation Agent (LangGraph node)
# ---------------------------------------------------------------------------

class ValuationAgent:
    """
    DCF Valuation Agent.

    Produces professional-grade valuation reports with:
    - Base/Upside/Downside per-share price targets
    - Asymmetric risk/reward ratio
    - Margin of safety assessment
    - Conviction score and recommendation
    """

    def __init__(self, wacc: float = DEFAULT_WACC, terminal_growth: float = DEFAULT_TERMINAL_GROWTH):
        self.wacc = wacc
        self.terminal_growth = terminal_growth

    async def value(self, symbol: str) -> Dict[str, Any]:
        """
        Run full DCF valuation for a single stock.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")

        Returns:
            Valuation report dict
        """
        result = await fetch_financials(symbol)
        if result.get("status") != "success":
            return result

        fin_data = result["data"]
        if not fin_data or fin_data.get("current_price", 0) <= 0:
            return {"status": "error", "error": f"No price data for {symbol}", "symbol": symbol}

        scenarios = build_scenarios(fin_data)
        return run_dcf_valuation(
            fin_data,
            scenarios=scenarios,
            wacc=self.wacc,
            terminal_growth=self.terminal_growth,
        )

    async def valuation_agent(self, state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """
        LangGraph node: run DCF valuation for all symbols in holdings/positions.

        Reads: state["holdings"], state["positions"], state["fundamental_reports"]
        Writes: state["valuation_reports"] — {symbol: DCF report}

        Args:
            state: Current AgentState dict

        Returns:
            Partial state update with valuation_reports
        """
        errors: list[str] = list(state.get("errors", []) or [])
        valuation_reports: Dict[str, Any] = {}

        holdings: list[dict] = list(state.get("holdings") or [])
        positions: list[dict] = list(state.get("positions") or [])
        fundamental_reports = state.get("fundamental_reports", {})

        # Collect symbols
        symbols: set[str] = set()
        for item in holdings + positions:
            sym = (
                item.get("tradingsymbol")
                or item.get("symbol")
                or item.get("trading_symbol")
                or ""
            ).upper()
            if sym:
                symbols.add(sym)

        # Also include symbols from fundamental reports
        if isinstance(fundamental_reports, dict):
            for sym in fundamental_reports.keys():
                if isinstance(sym, str) and sym not in ("timestamp", "symbols_analyzed", "reports"):
                    symbols.add(sym.upper())

        # Run DCF for each symbol
        for symbol in sorted(symbols)[:15]:  # Cap at 15 to manage rate limits
            try:
                report = await self.value(symbol)
                valuation_reports[symbol] = report
            except Exception as exc:
                msg = f"ValuationAgent failed for {symbol}: {exc}"
                logger.error(msg)
                errors.append(msg)
                valuation_reports[symbol] = {"status": "error", "error": str(exc)}

        # Aggregate conviction
        total_conviction = 0
        count = 0
        recommendations: Dict[str, int] = {}
        for report in valuation_reports.values():
            if report.get("status") == "success":
                total_conviction += report.get("conviction_score", 0)
                count += 1
                rec = report.get("recommendation", "HOLD")
                recommendations[rec] = recommendations.get(rec, 0) + 1

        avg_conviction = round(total_conviction / count, 1) if count > 0 else 0

        return {
            "valuation_reports": {
                "timestamp": datetime.now().isoformat(),
                "symbols_analyzed": count,
                "average_conviction": avg_conviction,
                "recommendation_distribution": recommendations,
                "reports": valuation_reports,
            },
            "errors": errors,
        }


async def valuation_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Convenience function: create and run the ValuationAgent.

    Args:
        state: Current AgentState dict

    Returns:
        Partial state update with valuation_reports
    """
    agent = ValuationAgent()
    return await agent.valuation_agent(state, config)


# ---------------------------------------------------------------------------
# Sector WACC Templates
# ---------------------------------------------------------------------------

SECTOR_WACC = {
    "IT Services": {"wacc": 0.095, "growth_period": 10, "terminal_growth": 0.035},
    "Technology": {"wacc": 0.095, "growth_period": 10, "terminal_growth": 0.035},
    "Banks": {"wacc": 0.105, "growth_period": 8, "terminal_growth": 0.03},
    "Financial Services": {"wacc": 0.105, "growth_period": 8, "terminal_growth": 0.03},
    "Consumer": {"wacc": 0.095, "growth_period": 10, "terminal_growth": 0.035},
    "Consumer Defensive": {"wacc": 0.09, "growth_period": 10, "terminal_growth": 0.035},
    "Pharma": {"wacc": 0.095, "growth_period": 12, "terminal_growth": 0.03},
    "Healthcare": {"wacc": 0.095, "growth_period": 12, "terminal_growth": 0.03},
    "Metals": {"wacc": 0.11, "growth_period": 6, "terminal_growth": 0.025},
    "Basic Materials": {"wacc": 0.11, "growth_period": 6, "terminal_growth": 0.025},
    "Oil & Gas": {"wacc": 0.11, "growth_period": 8, "terminal_growth": 0.025},
    "Energy": {"wacc": 0.11, "growth_period": 8, "terminal_growth": 0.025},
    "Real Estate": {"wacc": 0.12, "growth_period": 8, "terminal_growth": 0.025},
    "Auto": {"wacc": 0.10, "growth_period": 8, "terminal_growth": 0.03},
    "Telecom": {"wacc": 0.10, "growth_period": 8, "terminal_growth": 0.03},
    "Utilities": {"wacc": 0.08, "growth_period": 15, "terminal_growth": 0.025},
    "default": {"wacc": 0.10, "growth_period": 10, "terminal_growth": 0.03},
}


def get_sector_wacc(sector: str) -> dict:
    """Get WACC parameters for a given sector."""
    return SECTOR_WACC.get(sector, SECTOR_WACC["default"])


# ---------------------------------------------------------------------------
# WACC Calculator
# ---------------------------------------------------------------------------

def compute_wacc(
    market_cap: float,
    total_debt: float,
    cost_of_equity: float = 0.12,
    cost_of_debt: float = 0.07,
    tax_rate: float = 0.25,
) -> float:
    """Compute Weighted Average Cost of Capital."""
    total_capital = market_cap + total_debt
    if total_capital <= 0:
        return 0.10
    equity_weight = market_cap / total_capital
    debt_weight = total_debt / total_capital
    return equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)


# ---------------------------------------------------------------------------
# Reverse DCF
# ---------------------------------------------------------------------------

def compute_scenario(fcf: float, growth: float, wacc: float, terminal: float, years: int, shares: float) -> dict:
    """Helper: compute fair value for a single growth scenario."""
    try:
        projected = [fcf * (1 + growth)**i for i in range(1, years + 1)]
        tv = projected[-1] * (1 + terminal) / max(wacc - terminal, 0.001)
        pv = sum(cf / (1 + wacc)**i for i, cf in enumerate(projected, 1))
        pv += tv / (1 + wacc)**years
        fair = pv / max(shares, 1)
        growth_pct = f"{growth*100:.1f}%"
        return {"growth_rate": growth_pct, "fair_value": round(fair, 2)}
    except Exception:
        return {"growth_rate": "error", "fair_value": 0}


async def reverse_dcf(
    current_price: float,
    current_fcf: float,
    shares_outstanding: float,
    wacc: float = 0.10,
    terminal_growth: float = 0.03,
    projection_years: int = 10,
) -> dict:
    """
    Reverse DCF: Back out the revenue/FCF growth rate implied by current stock price.

    Instead of guessing future growth, this tells you what growth rate the MARKET
    is already pricing in. You then decide if that's realistic.

    Args:
        current_price: Current stock price per share
        current_fcf: Current free cash flow
        shares_outstanding: Number of shares outstanding
        wacc: Weighted Average Cost of Capital (default 10%)
        terminal_growth: Long-term growth rate (default 3%, ~GDP)
        projection_years: Years to project (default 10)

    Returns:
        dict with:
        - implied_growth: The CAGR % the market is pricing in
        - fair_value: What the DCF says the stock is worth at current FCF
        - mispricing_pct: How much over/undervalued (positive = undervalued)
        - current_enterprise_value: Market Cap + Debt - Cash (approximate)
        - scenarios: {base, upside, downside} growth rates and fair values
    """
    try:
        enterprise_value = current_price * shares_outstanding

        # Binary search for the growth rate that makes PV = current price
        low, high = -0.10, 0.30
        implied_growth = terminal_growth

        for _ in range(100):
            mid = (low + high) / 2
            projected_fcf = [current_fcf * (1 + mid)**i for i in range(1, projection_years + 1)]
            terminal_value = projected_fcf[-1] * (1 + terminal_growth) / max(wacc - terminal_growth, 0.001)

            pv = sum(fcf / (1 + wacc)**i for i, fcf in enumerate(projected_fcf, 1))
            pv += terminal_value / (1 + wacc)**projection_years

            fair_value_per_share = pv / max(shares_outstanding, 1)

            if fair_value_per_share > current_price:
                low = mid
            else:
                high = mid

            if abs(high - low) < 0.0001:
                break

        implied_growth = (low + high) / 2

        # Compute fair value at implied growth
        projected_fcf = [current_fcf * (1 + implied_growth)**i for i in range(1, projection_years + 1)]
        tv = projected_fcf[-1] * (1 + terminal_growth) / max(wacc - terminal_growth, 0.001)
        pv_total = sum(fcf / (1 + wacc)**i for i, fcf in enumerate(projected_fcf, 1))
        pv_total += tv / (1 + wacc)**projection_years
        fair_value = pv_total / max(shares_outstanding, 1)

        mispricing_pct = (fair_value - current_price) / max(current_price, 0.01)

        # Scenario analysis
        scenarios = {
            "upside": compute_scenario(current_fcf, implied_growth * 1.3, wacc, terminal_growth, projection_years, shares_outstanding),
            "base": compute_scenario(current_fcf, implied_growth, wacc, terminal_growth, projection_years, shares_outstanding),
            "downside": compute_scenario(current_fcf, implied_growth * 0.5, wacc, terminal_growth, projection_years, shares_outstanding),
        }

        return {
            "implied_growth": round(implied_growth, 4),
            "implied_growth_pct": f"{implied_growth*100:.1f}%",
            "fair_value": round(fair_value, 2),
            "current_price": current_price,
            "mispricing_pct": round(mispricing_pct, 4),
            "mispricing_label": "undervalued" if mispricing_pct > 0 else "overvalued",
            "enterprise_value": round(enterprise_value, 0),
            "wacc_used": wacc,
            "scenarios": scenarios,
        }
    except Exception as e:
        logger.warning("Reverse DCF failed: %s", e)
        return {"error": str(e), "implied_growth": 0, "fair_value": current_price}
