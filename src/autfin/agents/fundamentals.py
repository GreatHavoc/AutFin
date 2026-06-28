"""Fundamentals analysis agent for AutFin — agentic with tool-calling loop."""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.tools.fundamental_tools import FundamentalTools
from src.autfin.llm.client import get_llm_client
from src.autfin.agents.tools import FUNDAMENTALS_TOOLS, FUNDAMENTALS_CALLABLES

# Configure logging
logger = logging.getLogger(__name__)

# ================== CACHING ==================

class FundamentalsCache:
    """Caches fundamentals data to avoid redundant API calls."""
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 hour TTL
        """
        Initialize cache with TTL.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        return datetime.now() - self._timestamps[key] > self._ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if valid."""
        if key not in self._cache:
            return None
        if self._is_expired(key):
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp."""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()
# Global cache instance
_cache = FundamentalsCache(ttl_seconds=3600)

# ================== RATIO CALCULATION ==================

def calculate_ratios(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all financial ratios from fundamental data.
    
    Calculates 15+ key financial ratios across 5 categories:
    - Valuation: P/E, P/B, EV/EBITDA, P/S
    - Profitability: ROE, ROA, Net Margin, Operating Margin
    - Growth: Revenue Growth, EPS Growth, Profit Growth
    - Debt: Debt/Equity, Interest Coverage
    - Efficiency: Inventory Turnover, Asset Turnover
    
    Args:
        data: Dictionary with fundamental data
        
    Returns:
        Dictionary with calculated ratios
    """
    if not data:
        return {
            "status": "error",
            "error": "No data provided",
            "ratios": {}
        }
    
    fundamentals = data.get("data", {}) if "data" in data else data
    financials = fundamentals.get("financials", {})
    ratios_input = fundamentals.get("ratios", {})
    
    calculated_ratios = {}
    
    # Valuation Ratios (4)
    calculated_ratios["pe_ratio"] = ratios_input.get("pe_ratio", 0.0)
    calculated_ratios["pb_ratio"] = ratios_input.get("pb_ratio", 0.0)
    calculated_ratios["ev_ebitda"] = ratios_input.get("ev_ebitda", 0.0)
    calculated_ratios["price_to_sales"] = ratios_input.get("price_to_sales", 0.0)
    
    # Profitability Ratios (4)
    calculated_ratios["roe"] = ratios_input.get("roe", 0.0)
    calculated_ratios["roa"] = ratios_input.get("roa", 0.0)
    calculated_ratios["net_margin"] = ratios_input.get("net_margin", 0.0)
    calculated_ratios["operating_margin"] = ratios_input.get("operating_margin", 0.0)
    
    # Growth Ratios (3)
    calculated_ratios["revenue_growth"] = ratios_input.get("revenue_growth", 0.0)
    calculated_ratios["eps_growth"] = ratios_input.get("eps_growth", 0.0)
    calculated_ratios["profit_growth"] = ratios_input.get("profit_growth", 0.0)
    
    # Debt Ratios (2)
    calculated_ratios["debt_to_equity"] = ratios_input.get("debt_to_equity", 0.0)
    calculated_ratios["interest_coverage"] = ratios_input.get("interest_coverage", 0.0)
    
    # Efficiency Ratios (2)
    calculated_ratios["inventory_turnover"] = ratios_input.get("inventory_turnover", 0.0)
    calculated_ratios["asset_turnover"] = ratios_input.get("asset_turnover", 0.0)
    
    # Capital Efficiency Ratios (2) — ROIC and FCF Yield
    calculated_ratios["roic"] = ratios_input.get("roic", 0.0)
    calculated_ratios["fcf_yield"] = ratios_input.get("fcf_yield", 0.0)
    
    # Calculate derived ratios if not provided
    # P/E = Price / EPS
    # P/B = Price / Book Value per Share
    # EV/EBITDA = (Market Cap + Debt - Cash) / EBITDA
    # P/S = Market Cap / Revenue
    
    # ROE = Net Income / Shareholders' Equity
    # ROA = Net Income / Total Assets
    # Net Margin = Net Income / Revenue
    # Operating Margin = Operating Income / Revenue
    
    # Debt/Equity = Total Debt / Shareholders' Equity
    # Interest Coverage = EBIT / Interest Expense
    
    # Inventory Turnover = COGS / Average Inventory
    # Asset Turnover = Revenue / Total Assets
    
    return {
        "status": "success",
        "ratios": calculated_ratios,
        "calculated_at": datetime.now().isoformat()
    }


def compute_health_score(ratios: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute overall financial health score (0-100).
    
    Weights by category:
    - Profitability (30%): ROE, ROA, Net Margin, Operating Margin
    - Growth (20%): Revenue Growth, EPS Growth, Profit Growth
    - Debt (20%): Debt/Equity, Interest Coverage
    - Valuation (15%): P/E, P/B, EV/EBITDA, P/S
    - Efficiency (15%): Inventory Turnover, Asset Turnover, ROIC, FCF Yield
    
    Args:
        ratios: Dictionary with financial ratios
        
    Returns:
        Dictionary with health score (0-100) and breakdown
    """
    if not ratios or "ratios" not in ratios:
        return {
            "status": "error",
            "error": "No ratios provided",
            "health_score": 0.0,
            "breakdown": {}
        }
    
    r = ratios.get("ratios", {})
    
    # Define target ranges for healthy financials
    # Higher is better for profitability/growth/efficiency
    # Lower is better for debt/valuation ratios
    
    # Profitability Score (0-30)
    profitability_factors = []
    
    # ROE: Good > 15%, Fair 10-15%, Poor < 10%
    roe = r.get("roe", 0)
    if roe > 15:
        profitability_factors.append(30)
    elif roe > 10:
        profitability_factors.append(20)
    elif roe > 0:
        profitability_factors.append(max(0, roe))  # Linear scale for negative
    else:
        profitability_factors.append(0)
    
    # ROA: Good > 5%, Fair 3-5%, Poor < 3%
    roa = r.get("roa", 0)
    if roa > 5:
        profitability_factors.append(30)
    elif roa > 3:
        profitability_factors.append(20)
    elif roa > 0:
        profitability_factors.append(max(0, roa * 10))
    else:
        profitability_factors.append(0)
    
    # Net Margin: Good > 15%, Fair 10-15%, Poor < 10%
    net_margin = r.get("net_margin", 0)
    if net_margin > 15:
        profitability_factors.append(30)
    elif net_margin > 10:
        profitability_factors.append(20)
    elif net_margin > 0:
        profitability_factors.append(max(0, net_margin * 2))
    else:
        profitability_factors.append(0)
    
    # Operating Margin: Good > 20%, Fair 15-20%, Poor < 15%
    operating_margin = r.get("operating_margin", 0)
    if operating_margin > 20:
        profitability_factors.append(30)
    elif operating_margin > 15:
        profitability_factors.append(20)
    elif operating_margin > 0:
        profitability_factors.append(max(0, operating_margin * 2))
    else:
        profitability_factors.append(0)
    
    profitability_score = min(30, sum(profitability_factors) / len(profitability_factors))
    
    # Growth Score (0-20)
    growth_factors = []
    
    # Revenue Growth: Good > 15%, Fair 10-15%, Poor < 10%
    rev_growth = r.get("revenue_growth", 0)
    if rev_growth > 15:
        growth_factors.append(20)
    elif rev_growth > 10:
        growth_factors.append(15)
    elif rev_growth > 0:
        growth_factors.append(max(0, rev_growth * 2))
    else:
        growth_factors.append(0)
    
    # EPS Growth: Good > 15%, Fair 10-15%, Poor < 10%
    eps_growth = r.get("eps_growth", 0)
    if eps_growth > 15:
        growth_factors.append(20)
    elif eps_growth > 10:
        growth_factors.append(15)
    elif eps_growth > 0:
        growth_factors.append(max(0, eps_growth * 2))
    else:
        growth_factors.append(0)
    
    # Profit Growth: Good > 15%, Fair 10-15%, Poor < 10%
    profit_growth = r.get("profit_growth", 0)
    if profit_growth > 15:
        growth_factors.append(20)
    elif profit_growth > 10:
        growth_factors.append(15)
    elif profit_growth > 0:
        growth_factors.append(max(0, profit_growth * 2))
    else:
        growth_factors.append(0)
    
    growth_score = min(20, sum(growth_factors) / len(growth_factors))
    
    # Debt Score (0-20)
    # Lower debt is better
    debt_factors = []
    
    # Debt/Equity: Good < 0.5, Fair 0.5-1.0, Poor > 1.0
    debt_equity = r.get("debt_to_equity", 999)
    if debt_equity < 0.5:
        debt_factors.append(20)
    elif debt_equity < 1.0:
        debt_factors.append(15)
    elif debt_equity < 2.0:
        debt_factors.append(10)
    elif debt_equity < 999:
        debt_factors.append(max(0, 20 - debt_equity * 5))
    else:
        debt_factors.append(0)
    
    # Interest Coverage: Good > 5, Fair 3-5, Poor < 3
    interest_coverage = r.get("interest_coverage", 0)
    if interest_coverage > 5:
        debt_factors.append(20)
    elif interest_coverage > 3:
        debt_factors.append(15)
    elif interest_coverage > 0:
        debt_factors.append(min(15, interest_coverage * 5))
    else:
        debt_factors.append(0)
    
    debt_score = min(20, sum(debt_factors) / len(debt_factors))
    
    # Valuation Score (0-15)
    # Lower is better for valuation ratios
    valuation_factors = []
    
    # P/E: Good < 15, Fair 15-25, Poor > 25
    pe_ratio = r.get("pe_ratio", 999)
    if pe_ratio < 15:
        valuation_factors.append(15)
    elif pe_ratio < 25:
        valuation_factors.append(10)
    elif pe_ratio < 999:
        valuation_factors.append(max(0, 25 - pe_ratio) * 0.75)
    else:
        valuation_factors.append(0)
    
    # P/B: Good < 3, Fair 3-5, Poor > 5
    pb_ratio = r.get("pb_ratio", 999)
    if pb_ratio < 3:
        valuation_factors.append(15)
    elif pb_ratio < 5:
        valuation_factors.append(10)
    elif pb_ratio < 999:
        valuation_factors.append(max(0, 5 - pb_ratio) * 3)
    else:
        valuation_factors.append(0)
    
    # EV/EBITDA: Good < 10, Fair 10-15, Poor > 15
    ev_ebitda = r.get("ev_ebitda", 999)
    if ev_ebitda < 10:
        valuation_factors.append(15)
    elif ev_ebitda < 15:
        valuation_factors.append(10)
    elif ev_ebitda < 999:
        valuation_factors.append(max(0, 15 - ev_ebitda) * 3)
    else:
        valuation_factors.append(0)
    
    # P/S: Good < 5, Fair 5-10, Poor > 10
    ps_ratio = r.get("price_to_sales", 999)
    if ps_ratio < 5:
        valuation_factors.append(15)
    elif ps_ratio < 10:
        valuation_factors.append(10)
    elif ps_ratio < 999:
        valuation_factors.append(max(0, 10 - ps_ratio) * 3)
    else:
        valuation_factors.append(0)
    
    valuation_score = min(15, sum(valuation_factors) / len(valuation_factors))
    
    # Efficiency Score (0-15)
    # Higher is better
    efficiency_factors = []
    
    # Inventory Turnover: Good > 5, Fair 3-5, Poor < 3
    inv_turnover = r.get("inventory_turnover", 0)
    if inv_turnover > 5:
        efficiency_factors.append(15)
    elif inv_turnover > 3:
        efficiency_factors.append(10)
    elif inv_turnover > 0:
        efficiency_factors.append(min(10, inv_turnover * 5))
    else:
        efficiency_factors.append(0)
    
    # Asset Turnover: Good > 1, Fair 0.5-1, Poor < 0.5
    asset_turnover = r.get("asset_turnover", 0)
    if asset_turnover > 1:
        efficiency_factors.append(15)
    elif asset_turnover > 0.5:
        efficiency_factors.append(10)
    elif asset_turnover > 0:
        efficiency_factors.append(min(10, asset_turnover * 30))
    else:
        efficiency_factors.append(0)
    
    # ROIC: Good > 15%, Fair 10-15%, Poor < 10%
    roic = r.get("roic", 0)
    if roic > 15:
        efficiency_factors.append(15)
    elif roic > 10:
        efficiency_factors.append(10)
    elif roic > 0:
        efficiency_factors.append(min(10, roic))
    else:
        efficiency_factors.append(0)
    
    # FCF Yield: Good > 5%, Fair 2-5%, Poor < 2%
    fcf_yield = r.get("fcf_yield", 0)
    if fcf_yield > 5:
        efficiency_factors.append(15)
    elif fcf_yield > 2:
        efficiency_factors.append(10)
    elif fcf_yield > 0:
        efficiency_factors.append(min(10, fcf_yield * 5))
    else:
        efficiency_factors.append(0)
    
    efficiency_score = min(15, sum(efficiency_factors) / len(efficiency_factors))
    
    # Calculate total health score (0-100)
    total_score = (
        profitability_score +
        growth_score +
        debt_score +
        valuation_score +
        efficiency_score
    )
    
    return {
        "status": "success",
        "health_score": round(total_score, 2),
        "breakdown": {
            "profitability": {
                "score": round(profitability_score, 2),
                "max": 30,
                "metrics": ["roe", "roa", "net_margin", "operating_margin"]
            },
            "growth": {
                "score": round(growth_score, 2),
                "max": 20,
                "metrics": ["revenue_growth", "eps_growth", "profit_growth"]
            },
            "debt": {
                "score": round(debt_score, 2),
                "max": 20,
                "metrics": ["debt_to_equity", "interest_coverage"]
            },
            "valuation": {
                "score": round(valuation_score, 2),
                "max": 15,
                "metrics": ["pe_ratio", "pb_ratio", "ev_ebitda", "price_to_sales"]
            },
            "efficiency": {
                "score": round(efficiency_score, 2),
                "max": 15,
                "metrics": ["inventory_turnover", "asset_turnover", "roic", "fcf_yield"]
            }
        },
        "calculated_at": datetime.now().isoformat()
    }

# ================== MOAT ASSESSMENT ==================

# FR-033: Evaluate competitive position (moat) based on financial metrics.
# Uses ROE, margins, growth consistency, and debt levels to classify moat width.

def assess_moat(ratios: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess competitive moat width based on financial metrics.

    Moat levels:
    - wide: Consistently high ROE (>20%), strong margins, low debt, growth
    - narrow: Decent ROE (12-20%), acceptable margins, moderate debt
    - none: Low ROE, thin margins, high debt, inconsistent growth

    Args:
        ratios: Dictionary with financial ratios (from calculate_ratios)

    Returns:
        Dict with moat_level, moat_score (0-100), and rationale
    """
    r = ratios.get("ratios", {}) if "ratios" in ratios else ratios

    score = 0
    factors = []

    # --- ROE (most important moat signal) ---
    roe = r.get("roe", 0)
    if roe > 25:
        score += 30
        factors.append(f"Exceptional ROE ({roe:.1f}%)")
    elif roe > 20:
        score += 25
        factors.append(f"Strong ROE ({roe:.1f}%)")
    elif roe > 15:
        score += 18
        factors.append(f"Decent ROE ({roe:.1f}%)")
    elif roe > 10:
        score += 10
        factors.append(f"Weak ROE ({roe:.1f}%)")
    elif roe > 0:
        score += 5
        factors.append(f"Very low ROE ({roe:.1f}%)")
    else:
        factors.append(f"Negative ROE ({roe:.1f}%)")

    # --- Operating margin consistency ---
    op_margin = r.get("operating_margin", 0)
    if op_margin > 25:
        score += 25
        factors.append(f"Excellent operating margin ({op_margin:.1f}%)")
    elif op_margin > 18:
        score += 20
        factors.append(f"Strong operating margin ({op_margin:.1f}%)")
    elif op_margin > 12:
        score += 12
        factors.append(f"Adequate operating margin ({op_margin:.1f}%)")
    elif op_margin > 5:
        score += 5
        factors.append(f"Thin operating margin ({op_margin:.1f}%)")
    else:
        factors.append(f"Very thin operating margin ({op_margin:.1f}%)")

    # --- Net margin ---
    net_margin = r.get("net_margin", 0)
    if net_margin > 18:
        score += 20
        factors.append(f"Excellent net margin ({net_margin:.1f}%)")
    elif net_margin > 12:
        score += 15
        factors.append(f"Good net margin ({net_margin:.1f}%)")
    elif net_margin > 6:
        score += 8
        factors.append(f"Adequate net margin ({net_margin:.1f}%)")
    elif net_margin > 0:
        score += 3
        factors.append(f"Low net margin ({net_margin:.1f}%)")
    else:
        factors.append(f"Negative net margin ({net_margin:.1f}%)")

    # --- Revenue growth ---
    rev_growth = r.get("revenue_growth", 0)
    if rev_growth > 20:
        score += 15
        factors.append(f"High revenue growth ({rev_growth:.1f}%)")
    elif rev_growth > 10:
        score += 10
        factors.append(f"Moderate revenue growth ({rev_growth:.1f}%)")
    elif rev_growth > 0:
        score += 5
        factors.append(f"Slow revenue growth ({rev_growth:.1f}%)")
    else:
        factors.append(f"Negative revenue growth ({rev_growth:.1f}%)")

    # --- Debt penalty (high debt erodes moat) ---
    de = r.get("debt_to_equity", 999)
    if de < 0.3:
        score += 10
        factors.append(f"Very low debt (D/E: {de:.2f})")
    elif de < 0.7:
        score += 5
        factors.append(f"Moderate debt (D/E: {de:.2f})")
    elif de < 1.5:
        score += 0
        factors.append(f"Elevated debt (D/E: {de:.2f})")
    else:
        score -= 10
        factors.append(f"High debt (D/E: {de:.2f}) — erodes moat")

    # Clamp score
    score = max(0, min(100, score))

    # Classify moat
    if score >= 70:
        moat_level = "wide"
    elif score >= 45:
        moat_level = "narrow"
    else:
        moat_level = "none"

    rationale = (
        f"Moat assessment: {moat_level.upper()} (score {score}/100). "
        + "; ".join(factors) + "."
    )

    return {
        "moat_level": moat_level,
        "moat_score": score,
        "factors": factors,
        "rationale": rationale,
        "calculated_at": datetime.now().isoformat(),
    }


# ================== YFINANCE FALLBACK ==================

def _fetch_from_yfinance(symbol: str) -> Dict[str, Any]:
    """
    Fetch fundamental data from yfinance as fallback/primary source.
    
    Uses ticker.info for valuation ratios, financial statements for
    revenue/profit data, and computes derived metrics.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        
    Returns:
        Dictionary with yfinance fundamental data
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"status": "error", "error": "yfinance not installed", "data": None}

    ticker_symbol = f"{symbol.upper().removesuffix('.NS')}.NS"
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}
    except Exception as e:
        return {"status": "error", "error": f"yfinance error: {e}", "data": None}

    if not info or info.get("trailingPe") is None and info.get("forwardPe") is None:
        return {"status": "error", "error": f"No data for {ticker_symbol}", "data": None}

    def _safe(val, default=0.0):
        """Safely convert to float, returning default if None or non-numeric."""
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Extract valuation ratios
    pe_ratio = _safe(info.get("trailingPE")) or _safe(info.get("forwardPE"))
    pb_ratio = _safe(info.get("priceToBook"))
    ps_ratio = _safe(info.get("priceToSalesTrailing12Months"))
    
    # Enterprise value / EBITDA
    ev = _safe(info.get("enterpriseValue"))
    ebitda = _safe(info.get("ebitda"))
    ev_ebitda = ev / ebitda if ebitda and ebitda > 0 else 0.0

    # Profitability ratios
    roe = _safe(info.get("returnOnEquity"), 0.0) * 100  # yfinance returns as decimal
    roa = _safe(info.get("returnOnAssets"), 0.0) * 100
    net_margin = _safe(info.get("profitMargins"), 0.0) * 100
    operating_margin = _safe(info.get("operatingMargins"), 0.0) * 100

    # Debt ratios
    debt_equity = _safe(info.get("debtToEquity"))
    if debt_equity > 100:
        debt_equity = debt_equity / 100  # yfinance sometimes returns as percentage
    interest_coverage = _safe(info.get("interestCoverage"))

    # Financials
    total_revenue = _safe(info.get("totalRevenue"))
    net_income = _safe(info.get("netIncomeToCommon"))
    total_assets = _safe(info.get("totalAssets"))
    total_equity = _safe(info.get("totalStockholderEquity")) or _safe(info.get("bookValue"))
    total_debt = _safe(info.get("totalDebt"))
    cash = _safe(info.get("totalCash"))
    operating_income = _safe(info.get("operatingIncome"))
    free_cash_flow = _safe(info.get("freeCashflow"))
    market_cap = _safe(info.get("marketCap"))
    tax_rate = _safe(info.get("effectiveTaxRate", 0.25))  # Default 25% if not available

    # Growth metrics from yfinance
    revenue_growth = _safe(info.get("revenueGrowth"), 0.0) * 100
    earnings_growth = _safe(info.get("earningsGrowth"), 0.0) * 100

    # ROIC = NOPAT / Invested Capital
    # NOPAT = Operating Income * (1 - Tax Rate)
    # Invested Capital = Total Equity + Total Debt - Cash
    invested_capital = total_equity + total_debt - cash if all(v > 0 for v in [total_equity, total_debt]) else 0.0
    nopat = operating_income * (1 - tax_rate) if operating_income and operating_income > 0 else 0.0
    roic = (nopat / invested_capital * 100) if invested_capital and invested_capital > 0 else 0.0

    # FCF Yield = Free Cash Flow / Market Cap
    fcf_yield = (free_cash_flow / market_cap * 100) if free_cash_flow and market_cap and market_cap > 0 else 0.0

    # Efficiency ratios
    asset_turnover = _safe(info.get("assetTurnover"))
    
    # Try to get financial statements for more accurate growth
    try:
        income = ticker.income_stmt
        if income is not None and not income.empty:
            years = sorted(income.columns)
            if len(years) >= 2:
                latest = income[years[-1]]
                prev = income[years[-2]]
                latest_rev = _safe(latest.get("Total Revenue"))
                prev_rev = _safe(prev.get("Total Revenue"))
                if prev_rev and prev_rev > 0:
                    revenue_growth = ((latest_rev - prev_rev) / prev_rev) * 100
                latest_ni = _safe(latest.get("Net Income"))
                prev_ni = _safe(prev.get("Net Income"))
                if prev_ni and prev_ni != 0:
                    earnings_growth = ((latest_ni - prev_ni) / abs(prev_ni)) * 100
    except Exception:
        pass  # Use yfinance info defaults

    data = {
        "company_info": {
            "symbol": symbol.upper(),
            "name": info.get("shortName", symbol.upper()),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": _safe(info.get("marketCap")),
            "currency": info.get("currency", "INR"),
            "exchange": info.get("exchange", "NSI"),
        },
        "ratios": {
            "pe_ratio": round(pe_ratio, 2),
            "pb_ratio": round(pb_ratio, 2),
            "debt_to_equity": round(debt_equity, 2),
            "roe": round(roe, 2),
            "roa": round(roa, 2),
            "net_margin": round(net_margin, 2),
            "operating_margin": round(operating_margin, 2),
            "ev_ebitda": round(ev_ebitda, 2),
            "price_to_sales": round(ps_ratio, 2),
            "interest_coverage": round(interest_coverage, 2) if interest_coverage else 0.0,
            "inventory_turnover": 0.0,  # Not available from yfinance info
            "asset_turnover": round(asset_turnover, 2),
            "revenue_growth": round(revenue_growth, 2),
            "eps_growth": round(earnings_growth, 2),
            "profit_growth": round(earnings_growth, 2),  # Same as earnings growth
            "roic": round(roic, 2),
            "fcf_yield": round(fcf_yield, 2),
        },
        "financials": {
            "revenue": total_revenue,
            "net_income": net_income,
            "ebitda": ebitda,
            "total_assets": total_assets,
            "total_equity": total_equity,
            "total_debt": total_debt,
        },
        "growth_metrics": {
            "revenue_cagr_3yr": round(revenue_growth, 2),
            "eps_cagr_3yr": round(earnings_growth, 2),
            "profit_cagr_3yr": round(earnings_growth, 2),
        },
    }

    return {"status": "success", "symbol": symbol.upper(), "data": data}


# ================== MAIN AGENT ==================

class FundamentalsAgent:
    """Analyzes company fundamentals and financial health."""
    
    def __init__(self):
        """Initialize the Fundamentals Agent."""
        self._tools: Optional[FundamentalTools] = None

    async def close(self):
        """Clean up resources."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def scrape_screener(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch company fundamentals — yfinance primary, Screener.in fallback.
        
        Uses caching to avoid redundant API calls.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            
        Returns:
            Dictionary with fundamental data
        """
        # Check cache first
        cache_key = f"fundamentals_{symbol.upper()}"
        cached_data = _cache.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"Cache hit for {symbol}")
            return cached_data
        
        # yfinance is the primary source — reliable, comprehensive, no scraping
        result = _fetch_from_yfinance(symbol)
        
        if result.get("status") == "success":
            _cache.set(cache_key, result)
        
        return result
    
    async def calculate_ratios(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate financial ratios for a stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with calculated ratios
        """
        # Scrape fundamentals
        scraped = await self.scrape_screener(symbol)
        
        if scraped.get("status") != "success":
            return scraped
        
        # Calculate ratios
        ratios = calculate_ratios(scraped)
        
        return ratios
    
    async def compute_health_score(self, symbol: str) -> Dict[str, Any]:
        """
        Compute financial health score for a stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with health score and breakdown
        """
        # Calculate ratios
        ratios = await self.calculate_ratios(symbol)
        
        if ratios.get("status") != "success":
            return ratios
        
        # Compute health score
        health = compute_health_score(ratios)
        
        return health
    
    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Comprehensive fundamental analysis for a stock.
        
        Combines scraping, ratio calculation, and health scoring.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Complete analysis report
        """
        # Scrape fundamentals
        scraped = await self.scrape_screener(symbol)
        
        if scraped.get("status") != "success":
            return scraped
        
        # Calculate ratios
        ratios = calculate_ratios(scraped)
        
        # Compute health score
        health = compute_health_score(ratios)
        
        # Assess competitive moat (FR-033)
        moat = assess_moat(ratios)
        
        # Build comprehensive report
        report = {
            "status": "success",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "scraped_data": scraped.get("data", {}),
            "ratios": ratios.get("ratios", {}),
            "health_score": health.get("health_score", 0),
            "health_breakdown": health.get("breakdown", {}),
            "health_category": self._get_health_category(health.get("health_score", 0)),
            "moat_level": moat.get("moat_level", "none"),
            "moat_score": moat.get("moat_score", 0),
            "moat_rationale": moat.get("rationale", ""),
        }

        # Generate LLM-powered narrative (optional enhancement)
        llm_narrative = await self.generate_llm_narrative(symbol, report)
        if llm_narrative:
            report["llm_narrative"] = llm_narrative

        return report
    
    def _get_health_category(self, score: float) -> str:
        """
        Get health category from score.
        
        Categories:
        - Strong Buy: > 80
        - Buy: 60-80
        - Hold: 40-60
        - Sell: 20-40
        - Strong Sell: < 20
        """
        if score > 80:
            return "strong_buy"
        elif score > 60:
            return "buy"
        elif score > 40:
            return "hold"
        elif score > 20:
            return "sell"
        else:
            return "strong_sell"

    async def generate_llm_narrative(
        self, symbol: str, report: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate LLM-powered fundamental analysis narrative.

        Falls back to None if LLM unavailable or fails.

        Args:
            symbol: Stock symbol
            report: Full analysis report from analyze()

        Returns:
            Narrative string or None
        """
        llm = get_llm_client()
        if not llm.is_available:
            return None

        ratios = report.get("ratios", {})
        health = report.get("health_score", 0)
        category = report.get("health_category", "hold")

        prompt = (
            f"You are a fundamental analyst covering Indian equities.\n"
            f"Write a 2-paragraph fundamental analysis for {symbol}.\n\n"
            f"## Financial Data\n"
            f"- Health Score: {health:.0f}/100 ({category})\n"
            f"- P/E: {ratios.get('pe_ratio', 'N/A')}\n"
            f"- P/B: {ratios.get('pb_ratio', 'N/A')}\n"
            f"- ROE: {ratios.get('roe', 'N/A')}%\n"
            f"- Debt/Equity: {ratios.get('debt_to_equity', 'N/A')}\n"
            f"- Revenue Growth: {ratios.get('revenue_growth', 'N/A')}%\n"
            f"- Profit Growth: {ratios.get('profit_growth', 'N/A')}%\n"
            f"- Net Margin: {ratios.get('net_margin', 'N/A')}%\n"
            f"- Operating Margin: {ratios.get('operating_margin', 'N/A')}%\n"
            f"- EV/EBITDA: {ratios.get('ev_ebitda', 'N/A')}\n"
            f"- Interest Coverage: {ratios.get('interest_coverage', 'N/A')}x\n\n"
            f"## Instructions\n"
            "1. Assess valuation (cheap/fair/expensive vs sector)\n"
            "2. Assess financial health (profitability, debt, growth)\n"
            "3. Give a clear fundamental verdict\n"
            "Write in plain text, no markdown."
        )

        system = (
            "You are a fundamental research analyst at an Indian brokerage. "
            "Be specific with numbers. Write plain text, no markdown formatting."
        )

        return await llm.generate(
            prompt=prompt,
            system_prompt=system,
            max_tokens=400,
        )

    async def compare_peers(self, symbol: str) -> Dict[str, Any]:
        """
        Compare a stock with its peers.
        
        Args:
            symbol: Stock symbol to compare
            
        Returns:
            Dictionary with peer comparison data
        """
        # Get the stock's fundamentals
        stock_report = await self.analyze(symbol)
        
        if stock_report.get("status") != "success":
            return stock_report
        
        # For now, return placeholder peer comparison
        # In production, you would:
        # 1. Identify peer companies in the same sector/industry
        # 2. Fetch fundamentals for each peer
        # 3. Compare ratios and metrics

        return {
            "status": "success",
            "symbol": symbol,
            "peer_comparisons": [],
            "rank_in_sector": None,
            "percentile_rank": None,
            "analyst_recommendations": [],
        }

    async def _agentic_analyze(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Tool-calling agentic analysis: LLM decides what data to fetch and how to analyze."""
        llm = get_llm_client()
        if not llm.is_available:
            return None

        system_prompt = (
            f"You are an expert fundamental analyst. Analyze {symbol} on the Indian NSE exchange.\n"
            "Use your tools to gather data, then produce a COMPLETE analysis as a single JSON object.\n"
            "Include: pe_ratio, pb_ratio, ps_ratio, ev_ebitda, roe, roa, net_margin, operating_margin,\n"
            "debt_equity, interest_coverage, revenue_growth, earnings_growth, current_ratio,\n"
            "fcf_yield, roic, dividend_yield, sector, market_cap, revenue, net_income, cash,\n"
            "moat_level (wide/narrow/none), moat_score (0-100), health_score (0-100),\n"
            "recommendation (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL),\n"
            "strength (-5 to +5), conviction (0.0-1.0), risks (list), catalysts (list),\n"
            "dcf_target_price, dcf_upside_pct, dcf_downside_pct.\n"
            "Output ONLY the JSON object — no markdown, no explanation."
        )

        raw = await llm.generate_with_tools(
            system_prompt=system_prompt,
            tools=FUNDAMENTALS_TOOLS,
            tool_callables=FUNDAMENTALS_CALLABLES,
            model=llm.deep_model,
            max_tokens=2048,
        )
        if not raw:
            return None

        from src.autfin.utils import extract_json
        result = extract_json(raw)
        if result:
            return result
        logger.warning(f"Fundamentals agentic output not JSON for {symbol}")
        return None

    async def fundamentals_agent(self, state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """
        Orchestrate fundamentals analysis workflow.
        
        Runs fundamental analysis on stocks in state,
        updates state with fundamental signals, and handles errors gracefully.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with fundamental signals
        """
        errors = state.get("errors", [])
        fundamental_reports = state.get("fundamental_reports", {})
        
        try:
            # Get symbols from holdings or default to major indices
            symbols = []
            
            # Try to get symbols from holdings
            holdings = state.get("holdings", [])
            for holding in holdings:
                if "tradingsymbol" in holding:
                    symbols.append(holding["tradingsymbol"])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_symbols = []
            for symbol in symbols:
                if symbol not in seen:
                    seen.add(symbol)
                    unique_symbols.append(symbol)
            
            # If no symbols, default to major indices
            if not unique_symbols:
                unique_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
            
            # Analyze each symbol — try agentic path first, fall back to deterministic
            for symbol in unique_symbols[:10]:  # Limit to 10 symbols
                report = await self._agentic_analyze(symbol)
                if report:
                    report["status"] = "success"
                    fundamental_reports[symbol] = report
                else:
                    report = await self.analyze(symbol)
                    if report.get("status") == "success":
                        fundamental_reports[symbol] = report
                    else:
                        errors.append(
                            f"Failed to analyze {symbol}: {report.get('error', 'Unknown error')}"
                        )
            
            # Build final state update
            updated_state = {
                "fundamental_reports": {
                    "timestamp": datetime.now().isoformat(),
                    "symbols_analyzed": len(fundamental_reports),
                    "reports": fundamental_reports,
                },
                "agent": "fundamentals",
                "next": None,  # Will be set by supervisor
                "errors": errors,
            }
            
            return updated_state
            
        except Exception as e:
            errors.append(f"Fundamentals analysis failed: {str(e)}")
            
            return {
                "fundamental_reports": {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                },
                "agent": "fundamentals",
                "next": None,
                "errors": errors,
            }


# Global fundamentals agent instance for convenience
_fundamentals_agent: Optional[FundamentalsAgent] = None


def create_fundamentals_agent() -> FundamentalsAgent:
    """
    Create or get global fundamentals agent instance.
    
    Returns:
        FundamentalsAgent instance
    """
    global _fundamentals_agent
    
    if _fundamentals_agent is None:
        _fundamentals_agent = FundamentalsAgent()
    
    return _fundamentals_agent


async def fundamentals_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Convenience function to run fundamentals agent.

    Args:
        state: Current agent state
        config: RunnableConfig for tracing

    Returns:
        Updated state with fundamental signals
    """
    agent = create_fundamentals_agent()
    return await agent.fundamentals_agent(state, config)