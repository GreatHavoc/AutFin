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
    """Keep only 'symbol' and 'depth' from agent tool inputs."""
    return {k: v for k, v in inputs.items() if k in ("symbol", "depth")}

def _filter_no_args_inputs(inputs: dict) -> dict:
    """No-argument tools: macro_analysis, portfolio_allocation — return empty dict."""
    return {}

def _filter_symbols_inputs(inputs: dict) -> dict:
    """Keep only 'symbols' and 'depth' from research_tool inputs."""
    return {k: v for k, v in inputs.items() if k in ("symbols", "depth")}

def _filter_trade_inputs(inputs: dict) -> dict:
    """Keep only 'trade_json' from risk_tool inputs."""
    return {k: v for k, v in inputs.items() if k == "trade_json"}


def _filter_agent_outputs(outputs: dict) -> dict:
    """Strip large payloads from agent tool outputs for LangSmith."""
    raw = outputs.get("output", outputs)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = {"raw": raw[:500] + "...[truncated]" if len(raw) > 500 else raw}
    elif isinstance(raw, dict):
        parsed = raw
    else:
        return {"output_type": type(raw).__name__}

    result: dict = {}
    for key in ("status", "symbol", "symbols"):
        if key in parsed:
            result[key] = parsed[key]
    if "raw_analysis" in parsed:
        v = parsed["raw_analysis"]
        result["raw_analysis"] = v[:500] + "...[truncated]" if isinstance(v, str) and len(v) > 500 else v
    if "summary" in parsed:
        result["summary"] = str(parsed["summary"])[:300]
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

    Calls FundamentalsAgent.analyze() directly: yfinance scrape → ratios
    → Piotroski F-score → moat assessment.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick/standard/deep (controls which sub-analyses run)
    """
    from src.autfin.agents.fundamentals import FundamentalsAgent

    try:
        agent = FundamentalsAgent()
        report = await agent.analyze(symbol)

        if depth == "deep":
            peers = await agent.compare_peers(symbol)
            if peers and peers.get("status") == "success":
                report["peer_comparison"] = peers

        report["depth"] = depth
        return json.dumps(report, default=str)
    except Exception as e:
        logger.warning("fundamentals_analysis failed for %s: %s", symbol, e)
        return json.dumps({"status": "error", "symbol": symbol, "message": str(e)})


@_traceable(name="technical_analysis", run_type="chain", process_inputs=_filter_symbol_inputs, process_outputs=_filter_agent_outputs)
async def technical_tool(symbol: str, depth: str = "standard") -> str:
    """Run technical analysis on a stock. Deterministic — no LLM.

    Fetches OHLCV via yfinance, then calls TechnicalAgent.analyze()
    which runs pandas_ta_classic indicators (RSI, MACD, BB, ATR, etc.)
    and generates a structured signal.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (3mo), standard (6mo), deep (1y)
    """
    from src.autfin.agents.technical import TechnicalAgent
    import pandas as pd

    try:
        import yfinance as yf

        period_map = {"quick": "3mo", "standard": "6mo", "deep": "1y"}
        period = period_map.get(depth, "6mo")

        ticker = yf.Ticker(f"{symbol.removesuffix('.NS')}.NS")
        hist = ticker.history(period=period, interval="1d")

        if hist is None or hist.empty:
            return json.dumps({
                "status": "error", "symbol": symbol,
                "message": f"OHLCV data unavailable for {symbol}",
            })

        price_data = hist.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        agent = TechnicalAgent()
        report = await agent.analyze(symbol, data=price_data)
        report["depth"] = depth
        return json.dumps(report, default=str)
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

    Fetches macro indicators (VIX, oil, USD/INR, Fed rate, etc.) via
    yfinance, then runs deterministic India-impact scoring.

    No arguments — macro analysis is always portfolio-wide.
    """
    from src.autfin.agents.global_macro import GlobalMacroAgent

    try:
        agent = GlobalMacroAgent()
        indicators = await agent.fetch_indicators()
        impact = await agent.analyze_impact(indicators)
        score = await agent.calculate_impact_score(impact)

        result = {
            "status": "success",
            "indicators": indicators,
            "impact": impact,
            "impact_score": score,
        }
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
    "portfolio_allocation": portfolio_tool,
}
