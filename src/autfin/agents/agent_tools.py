"""
Tool wrappers for v5.4 — deterministic-first pattern.

v5.3: Each tool wrapped an agent with its own LLM call.
v5.4: Each tool calls the underlying function DIRECTLY.
      LLM is only used by research_synthesis (Bull/Bear debate).

These are TOOLS, not agents — they fetch/compute data deterministically.
The only LLM-driven tool is research_synthesis (Bull/Bear debate).

This gives 3x speed improvement: 20-40s vs 60-120s.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# LangSmith @traceable — creates separate traced runs for each agent tool
try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(name: str = "", run_type: str = "chain", **kw):  # type: ignore[misc]
        """Fallback no-op decorator when langsmith is not installed."""
        def deco(fn):
            return fn
        return deco


# ---------------------------------------------------------------------------
# process_inputs helpers — strip large payloads from LangSmith traces.
# ---------------------------------------------------------------------------

def _filter_symbol_inputs(inputs: dict) -> dict:
    """Keep only 'symbol' and 'depth' from agent tool inputs.

    LangSmith passes inputs as a flat kwargs dict; tolerate missing/extra
    keys and non-dict shapes (defensive).
    """
    if not isinstance(inputs, dict):
        return {}
    return {k: v for k, v in inputs.items() if k in ("symbol", "depth")}

def _filter_no_args_inputs(inputs: dict) -> dict:
    """No-argument tools: macro_analysis, portfolio_allocation — return empty dict."""
    return {}

def _filter_symbols_inputs(inputs: dict) -> dict:
    """Keep only 'symbols' and 'depth' from research_tool inputs."""
    if not isinstance(inputs, dict):
        return {}
    return {k: v for k, v in inputs.items() if k in ("symbols", "depth")}

def _filter_trade_inputs(inputs: dict) -> dict:
    """Keep only 'trade_json' from risk_tool inputs."""
    if not isinstance(inputs, dict):
        return {}
    return {k: v for k, v in inputs.items() if k == "trade_json"}


def _filter_agent_outputs(outputs: Any) -> dict:
    """Strip large payloads from agent tool outputs for LangSmith.

    LangSmith calls process_outputs(return_value). Depending on the tool,
    the return value can be:
      - a dict like {'output': ..., 'symbol': ..., 'status': ...}
      - a plain dict with the same keys at the top level
      - a JSON string (from json.dumps)
      - a non-serializable object (we tag it with a type marker)
    Normalise to a dict before slicing so this handler never crashes
    with AttributeError("'str' object has no attribute 'get'").
    """
    raw: Any = outputs

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                _data = parsed
            else:
                return {"_filtered": True, "preview": str(parsed)[:300]}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"_filtered": True, "preview": raw[:500] + "...[truncated]" if len(raw) > 500 else raw}
    elif isinstance(raw, dict):
        _data = raw
    elif raw is None:
        return {"_filtered": True, "preview": None}
    else:
        return {"_filtered": True, "preview": repr(raw)[:300]}

    if _data.get("output") is not None and isinstance(_data.get("output"), (dict, str, list)):
        candidate = _data["output"]
        if isinstance(candidate, dict):
            _data = candidate
        elif isinstance(candidate, str):
            try:
                parsed_inner = json.loads(candidate)
                if isinstance(parsed_inner, dict):
                    _data = parsed_inner
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    result: dict = {}
    for key in ("status", "symbol", "symbols", "error"):
        if key in _data and not isinstance(_data[key], (dict, list)):
            result[key] = _data[key]
    if "raw_analysis" in _data:
        v = _data["raw_analysis"]
        result["raw_analysis"] = v[:500] + "...[truncated]" if isinstance(v, str) and len(v) > 500 else v
    if "summary" in _data:
        result["summary"] = str(_data["summary"])[:300]
    if not result:
        result["_filtered"] = True
    return result


# ---------------------------------------------------------------------------
# OpenAI-format tool definition helper
# ---------------------------------------------------------------------------

def _tool_def(name: str, description: str, parameters: Dict) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys()),
            },
        },
    }


# ---------------------------------------------------------------------------
# DETERMINISTIC TOOLS — no LLM, direct function calls
# ---------------------------------------------------------------------------


@_traceable(name="fundamentals_analysis", run_type="chain", process_inputs=_filter_symbol_inputs, process_outputs=_filter_agent_outputs)
async def fundamentals_tool(symbol: str, depth: str = "standard") -> str:
    """Run fundamental analysis on a stock. Deterministic — no LLM.

    Uses FundamentalTools directly: yfinance scrape → ratios
    → health score → peer comparison.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick/standard/deep (controls which sub-analyses run)
    """
    from src.autfin.tools.fundamental_tools import FundamentalTools

    try:
        tools = FundamentalTools()
        financials = await tools.get_financials(symbol)
        health = await tools.compute_health_score(symbol)

        report = {
            "symbol": symbol,
            "financials": financials,
            "health_score": health.get("health_score", 0),
            "health_rating": health.get("rating", "UNKNOWN"),
            "health_details": health.get("details", {}),
        }

        if depth == "deep":
            # Default peer set for Indian large caps
            peers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
            peer_data = await tools.compare_peers(symbol, peers)
            if peer_data and peer_data.get("peers"):
                report["peer_comparison"] = peer_data

        report["depth"] = depth
        return json.dumps(report, default=str)
    except Exception as e:
        logger.warning("fundamentals_analysis failed for %s: %s", symbol, e)
        return json.dumps({"status": "error", "symbol": symbol, "message": str(e)})


@_traceable(name="technical_analysis", run_type="chain", process_inputs=_filter_symbol_inputs, process_outputs=_filter_agent_outputs)
async def technical_tool(symbol: str, depth: str = "standard", as_of: str | None = None) -> str:
    """Run technical analysis on a stock. Deterministic — no LLM.

    Fetches OHLCV via yfinance and computes indicators via technical_tools.
    Computes: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Stochastic.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (3mo), standard (6mo), deep (1y)
        as_of: ISO datetime for point-in-time backtests (None = live)
    """
    from src.autfin.tools.technical_tools import fetch_technical_indicators, detect_patterns

    try:
        indicators_json = await fetch_technical_indicators(symbol)
        indicators = json.loads(indicators_json)

        if "error" in indicators:
            return json.dumps({
                "status": "error", "symbol": symbol,
                "message": indicators["error"],
            })

        # Add pattern detection for deeper analysis
        if depth in ("standard", "deep"):
            patterns_json = await detect_patterns(symbol)
            patterns = json.loads(patterns_json)
            if "error" not in patterns:
                indicators["patterns"] = patterns

        indicators["depth"] = depth
        return json.dumps(indicators, default=str)
    except Exception as e:
        logger.warning("technical_analysis failed for %s: %s", symbol, e)
        return json.dumps({"status": "error", "symbol": symbol, "message": str(e)})


@_traceable(name="sentiment_analysis", run_type="chain", process_inputs=_filter_symbol_inputs, process_outputs=_filter_agent_outputs)
async def sentiment_tool(symbol: str) -> str:
    """Run sentiment analysis on a stock. Deterministic — no LLM.

    Uses FinBERT + VADER on Marketaux news + Reddit (Indian finance subs).
    No LLM call — sentiment scoring is fully algorithmic.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
    """
    from src.autfin.tools.sentiment_tools import SentimentTools

    try:
        tools = SentimentTools()
        result = await tools.get_multi_source_sentiment(symbol)
        await tools.close()
        return json.dumps(result, default=str)
    except Exception as e:
        logger.warning("sentiment_analysis failed for %s: %s", symbol, e)
        return json.dumps({"status": "error", "symbol": symbol, "message": str(e)})


@_traceable(name="macro_analysis", run_type="chain", process_inputs=_filter_no_args_inputs, process_outputs=_filter_agent_outputs)
async def macro_tool() -> str:
    """Run global macro analysis. Deterministic — no LLM.

    Fetches global indices, rates, commodities, computes India impact.
    Returns {indices, rates, india_impact, narrative, sector_weights}.
    """
    from src.autfin.tools.supervisor_tools import analyze_macro_context

    try:
        result = await analyze_macro_context()
        return json.dumps(result, default=str)
    except Exception as e:
        logger.warning("macro_analysis failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@_traceable(name="portfolio_allocation", run_type="chain", process_inputs=_filter_no_args_inputs, process_outputs=_filter_agent_outputs)
async def portfolio_tool() -> str:
    """Get current portfolio state. Deterministic — no LLM.

    Returns holdings, cash, margin, positions from Kite SDK.
    No arguments — portfolio is always global state.
    """
    from src.autfin.tools.supervisor_tools import get_portfolio_state

    try:
        portfolio_data = await get_portfolio_state()
        return json.dumps(portfolio_data, default=str)
    except Exception as e:
        logger.warning("portfolio_allocation failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@_traceable(name="market_status", run_type="chain", process_inputs=_filter_no_args_inputs, process_outputs=_filter_agent_outputs)
async def market_status_tool() -> str:
    """Check market status: open/closed, Nifty, Bank Nifty, global indices, sector performance.
    Deterministic — no LLM. No arguments.
    Returns {is_open, nifty_level, nifty_change_pct, bank_nifty, global_indices, sectors}.
    """
    from src.autfin.tools.supervisor_tools import get_market_status

    try:
        status = await get_market_status()
        return json.dumps(status, default=str)
    except Exception as e:
        logger.warning("market_status failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@_traceable(name="market_quotes", run_type="chain", process_inputs=_filter_symbols_inputs, process_outputs=_filter_agent_outputs)
async def quotes_tool(symbols: str = "") -> str:
    """Get live price quotes for symbols. Deterministic — no LLM.
    Uses Kite SDK with yfinance fallback.
    Args: symbols (comma-separated, e.g. 'RELIANCE,TCS,INFY')
    """
    from src.autfin.tools.supervisor_tools import get_market_quotes

    try:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else []
        quotes = await get_market_quotes(sym_list)
        return json.dumps(quotes, default=str)
    except Exception as e:
        logger.warning("market_quotes failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@_traceable(name="risk_assessment", run_type="chain", process_inputs=_filter_trade_inputs, process_outputs=_filter_agent_outputs)
async def risk_tool(trade_json: str) -> str:
    """Run risk assessment on a proposed trade. Deterministic — no LLM.

    Runs 14 risk checks (position sizing, sector concentration, drawdown,
    liquidity, correlation, etc.) via RiskManagerAgent.

    Args:
        trade_json: JSON string: {"symbol": "X", "action": "BUY"/"SELL", "quantity": N, "price": P}
    """
    from src.autfin.tools.supervisor_tools import run_risk_checks

    try:
        trade = json.loads(trade_json) if isinstance(trade_json, str) else trade_json
        result = await run_risk_checks(trade)
        return json.dumps(result, default=str)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"status": "error", "message": "Invalid trade JSON"})
    except Exception as e:
        logger.warning("risk_assessment failed: %s", e)
        return json.dumps({"status": "error", "message": str(e)})


@_traceable(name="distortion_audit", run_type="chain", process_inputs=_filter_symbol_inputs, process_outputs=_filter_agent_outputs)
async def distortion_tool(symbol: str, depth: str = "standard") -> str:
    """Run price-distortion detection. Deterministic — no LLM.

    Detects fake breakouts, pump patterns, distribution signals,
    circuit suspicion, and sector relative weakness.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (fewer checks), standard (all), deep (all + extended)
    """
    from src.autfin.agents.distortion import distortion_audit

    try:
        result = await distortion_audit(symbol, depth)
        return json.dumps(result, default=str)
    except Exception as e:
        logger.warning("distortion_audit failed for %s: %s", symbol, e)
        return json.dumps({"status": "error", "symbol": symbol, "message": str(e)})


@_traceable(name="regulatory_analysis", run_type="chain", process_inputs=_filter_symbol_inputs, process_outputs=_filter_agent_outputs)
async def regulatory_tool(symbol: str, depth: str = "standard") -> str:
    """Run regulatory analysis on a stock. Deterministic — no LLM.

    Scans NSE corporate announcements, SEBI orders, RBI circulars,
    and DPCO notifications. Classifies impact on sectors.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (3 days), standard (7 days), deep (30 days)
    """
    from src.autfin.agents.regulatory import regulatory_audit

    try:
        result = await regulatory_audit(symbol, depth)
        return json.dumps(result, default=str)
    except Exception as e:
        logger.warning("regulatory_analysis failed for %s: %s", symbol, e)
        return json.dumps({"status": "error", "symbol": symbol, "message": str(e)})


@_traceable(name="valuation_analysis", run_type="chain", process_inputs=_filter_symbol_inputs, process_outputs=_filter_agent_outputs)
async def valuation_tool(symbol: str, depth: str = "standard") -> str:
    """Run DCF valuation on a stock. Deterministic — no LLM.

    Fetches financials via yfinance (via FundamentalTools), then runs DCF scenarios
    (base/upside/downside) and reverse DCF.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (simple DCF), standard (3 scenarios), deep (full + reverse DCF)
    """
    from src.autfin.tools.fundamental_tools import FundamentalTools
    from datetime import datetime

    def _compute_dcf(
        free_cash_flow: float,
        revenue_growth: float,
        shares_outstanding: float,
        wacc: float = 0.12,
        terminal_growth: float = 0.03,
        forecast_years: int = 5,
    ) -> Dict[str, Any]:
        """Compute DCF valuation with standard assumptions."""
        if free_cash_flow <= 0 or shares_outstanding <= 0:
            return {"fair_value": 0, "error": "Invalid inputs"}

        # Project FCF
        fcf_projections = []
        for year in range(1, forecast_years + 1):
            fcf_year = free_cash_flow * (1 + revenue_growth) ** year
            fcf_projections.append(fcf_year)

        # Discount to present value
        pv_sum = 0
        for i, fcf in enumerate(fcf_projections, 1):
            pv = fcf / (1 + wacc) ** i
            pv_sum += pv

        # Terminal value
        terminal_fcf = fcf_projections[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        terminal_pv = terminal_value / (1 + wacc) ** forecast_years

        enterprise_value = pv_sum + terminal_pv
        equity_value = enterprise_value  # simplified: no net debt
        fair_value = equity_value / shares_outstanding

        return {
            "fair_value": round(fair_value, 2),
            "enterprise_value": round(enterprise_value, 2),
            "pv_fcf": round(pv_sum, 2),
            "terminal_value": round(terminal_pv, 2),
            "assumptions": {
                "wacc": wacc,
                "terminal_growth": terminal_growth,
                "forecast_years": forecast_years,
                "revenue_growth": revenue_growth,
            },
        }

    def _build_scenarios(fin_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Build DCF scenarios from financial data."""
        base_growth = fin_data.get("revenue_growth_3y", 0.15) or 0.15
        return {
            "base": {"growth": base_growth},
            "upside": {"growth": min(base_growth * 1.5, 0.40)},
            "downside": {"growth": max(base_growth * 0.5, 0.02)},
        }

    def _reverse_dcf(
        current_price: float,
        current_fcf: float,
        shares_outstanding: float,
        wacc: float = 0.12,
        terminal_growth: float = 0.03,
        forecast_years: int = 5,
    ) -> Dict[str, Any]:
        """Compute implied growth rate from current price."""
        if current_price <= 0 or shares_outstanding <= 0:
            return {"implied_growth": 0, "error": "Invalid inputs"}

        target_ev = current_price * shares_outstanding
        # Simple binary search for implied growth
        low, high = 0.0, 0.50
        for _ in range(30):
            mid = (low + high) / 2
            dcf = _compute_dcf(current_fcf, mid, shares_outstanding, wacc, terminal_growth, forecast_years)
            if dcf["enterprise_value"] > target_ev:
                high = mid
            else:
                low = mid
        implied = (low + high) / 2
        return {"implied_growth": round(implied, 4), "current_price": current_price}

    async def _fetch_financials(symbol: str) -> Dict[str, Any]:
        tools = FundamentalTools()
        data = await tools.get_financials(symbol)
        if "error" in data:
            return {"status": "error", "message": data["error"]}
        # Extract needed fields
        data["free_cash_flow"] = data.get("fcf_yield", 0) * data.get("market_cap", 0) if data.get("market_cap") else 1e6
        data["shares_outstanding"] = data.get("market_cap", 0) / data.get("pe", 1) / data.get("eps", 1) if data.get("pe") and data.get("eps") else 1e9
        data["current_price"] = data.get("market_cap", 0) / data.get("shares_outstanding", 1) if data.get("market_cap") and data.get("shares_outstanding") else 0
        data["revenue_growth"] = data.get("revenue_growth_3y", 0.15)
        data["status"] = "success"
        return data

    try:
        fin_data = await _fetch_financials(symbol)

        if fin_data.get("status") == "error":
            return json.dumps(fin_data, default=str)

        fcf = fin_data.get("free_cash_flow", 0) or 1e6
        growth = fin_data.get("revenue_growth", 0.15) or 0.15
        shares = fin_data.get("shares_outstanding", 1e9) or 1e9
        price = fin_data.get("current_price", 0) or 0

        # Build scenarios
        scenarios = _build_scenarios(fin_data)

        # Run DCF for each scenario
        dcf_results = {}
        for scenario_name, scenario_data in scenarios.items():
            dcf_results[scenario_name] = _compute_dcf(
                free_cash_flow=fcf,
                revenue_growth=scenario_data.get("growth", growth),
                shares_outstanding=shares,
            )

        result = {
            "status": "success",
            "symbol": symbol,
            "dcf_scenarios": dcf_results,
            "financials": {
                "fcf": fcf,
                "revenue_growth": growth,
                "shares_outstanding": shares,
                "current_price": price,
            },
            "depth": depth,
            "timestamp": datetime.now().isoformat(),
        }

        # Add reverse DCF for deep analysis
        if depth == "deep" and price > 0:
            try:
                rev_dcf = _reverse_dcf(
                    current_price=price,
                    current_fcf=fcf,
                    shares_outstanding=shares,
                )
                result["reverse_dcf"] = rev_dcf
            except Exception:
                pass

        return json.dumps(result, default=str)
    except Exception as e:
        logger.warning("valuation_analysis failed for %s: %s", symbol, e)
        return json.dumps({"status": "error", "symbol": symbol, "message": str(e)})


# ---------------------------------------------------------------------------
# LLM-DRIVEN TOOLS — genuinely need reasoning
# ---------------------------------------------------------------------------


@_traceable(name="research_synthesis", run_type="chain", process_inputs=_filter_symbols_inputs, process_outputs=_filter_agent_outputs)
async def research_tool(symbols: str, depth: str = "standard") -> str:
    """Run Bull/Bear debate on one or more stocks. LLM-DRIVEN.

    This is the one agent tool that genuinely needs LLM reasoning:
    adversarial multi-round debate between bull and bear researchers.

    Args:
        symbols: Comma-separated stock symbols (e.g. 'RELIANCE,TCS')
        depth: quick, standard, or deep
    """
    from src.autfin.llm.client import get_llm_client
    from src.autfin.agents.tools import FUNDAMENTALS_TOOLS, FUNDAMENTALS_CALLABLES

    llm = get_llm_client()
    if not llm.is_available:
        return json.dumps({"status": "error", "message": "LLM unavailable"})

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return json.dumps({"status": "error", "message": "No valid symbols provided"})

    system_prompt = (
        f"You are a senior research analyst at an Indian investment firm.\n"
        f"Run a comprehensive Bull/Bear debate for: {', '.join(symbol_list)}\n\n"
        "For each stock, gather fundamentals, technicals, and sentiment data,\n"
        "then synthesize into a clear investment thesis.\n\n"
        "Use your tools to fetch financial data, then produce the synthesis.\n\n"
        "Output a SINGLE JSON:\n"
        '{"stocks": {"SYMBOL": {"bull_case": str, "bear_case": str, '
        '"rating": "STRONG_BUY"/"BUY"/"HOLD"/"SELL"/"STRONG_SELL", '
        '"conviction": float(0.0-1.0), "target_price": float, "summary": str}}, '
        '"portfolio_impact": str, "key_risks": list, "top_picks": list}\n\n'
        "Output ONLY the raw JSON."
    )

    raw = await llm.generate_with_tools(
        system_prompt=system_prompt,
        tools=FUNDAMENTALS_TOOLS,
        tool_callables=FUNDAMENTALS_CALLABLES,
        model=llm.deep_model,
        max_tokens=4096,
        max_iterations=20,
    )

    if not raw:
        return json.dumps({"status": "error", "message": "Research synthesis failed"})

    from src.autfin.utils import extract_json
    result = extract_json(raw)
    if result:
        return json.dumps(result)
    return json.dumps({"status": "success", "symbols": symbol_list, "raw_analysis": raw[:2000]})


# ---------------------------------------------------------------------------
# Tool definitions — OpenAI format for supervisor's tool-calling loop
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict] = [
    _tool_def(
        "fundamentals_analysis",
        "Run fundamental analysis on a stock (Piotroski, Magic Formula, DCF, moat). "
        "Deterministic — returns ratios, health score, and recommendation.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
        },
    ),
    _tool_def(
        "technical_analysis",
        "Run technical analysis on a stock (RSI, MACD, Bollinger, patterns). "
        "Deterministic — returns indicators, signal, and conviction.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
            "as_of": {"type": "string", "description": "ISO datetime for point-in-time backtests (omit for live)"},
        },
    ),
    _tool_def(
        "sentiment_analysis",
        "Run sentiment analysis on a stock (news, social media, FinBERT scoring). "
        "Deterministic — returns sentiment score and catalysts.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
        },
    ),
    _tool_def(
        "macro_analysis",
        "Run global macro analysis (Fed, oil, DXY, VIX, FII flows, RBI policy). "
        "Deterministic — returns impact signal and sector matrix.",
        {},
    ),
    _tool_def(
        "research_synthesis",
        "Run Bull/Bear debate on one or more stocks. LLM-DRIVEN — most expensive tool. "
        "Only use for final investment decisions.",
        {
            "symbols": {"type": "string", "description": "Comma-separated symbols (e.g. RELIANCE,TCS)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
        },
    ),
    _tool_def(
        "risk_assessment",
        "Run 14 deterministic risk checks on a proposed trade. "
        "Returns verdict, approved size, and stop-loss.",
        {
            "trade_json": {"type": "string", "description": "JSON trade: {symbol, action, quantity, price}"},
        },
    ),
    _tool_def(
        "distortion_audit",
        "Run price-distortion detection (fake breakouts, pump patterns, distribution). "
        "Deterministic — returns flags and conviction adjustment.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
        },
    ),
    _tool_def(
        "regulatory_analysis",
        "Scan NSE/SEBI/RBI regulatory announcements. "
        "Deterministic — returns impact level, sector, and direction.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "depth": {"type": "string", "description": "quick (3d)/standard (7d)/deep (30d)"},
        },
    ),
    _tool_def(
        "valuation_analysis",
        "Run DCF valuation on a stock (base/upside/downside scenarios + reverse DCF). "
        "Deterministic — returns intrinsic value and implied growth.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
        },
    ),
    _tool_def(
        "portfolio_allocation",
        "Get current portfolio state with live holdings, cash, and sector exposure. "
        "Deterministic — returns total value, P&L, and holdings.",
        {},
    ),
]

# Callable mapping — name → async function
TOOL_CALLABLES: Dict[str, Any] = {
    "fundamentals_analysis": fundamentals_tool,
    "technical_analysis": technical_tool,
    "sentiment_analysis": sentiment_tool,
    "macro_analysis": macro_tool,
    "research_synthesis": research_tool,
    "risk_assessment": risk_tool,
    "distortion_audit": distortion_tool,
    "regulatory_analysis": regulatory_tool,
    "valuation_analysis": valuation_tool,
    "portfolio_allocation": portfolio_tool,
}
