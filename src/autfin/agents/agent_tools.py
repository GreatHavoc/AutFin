"""
Agent tool wrappers for v5.3 Subagents pattern.

Each tool wraps an entire agent (with its own LLM call) into a callable
tool the supervisor can invoke.  This gives context isolation — each
agent gets a clean context window — and enables parallel execution.

These are DIFFERENT from the low-level data tools in ``tools.py``:
- ``tools.py``: fetch_financial_info, compute_rsi, etc. (data fetching)
- ``agent_tools.py``: fundamentals_tool, technical_tool, etc. (full agent runs)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-format tool definition helper (same as tools.py)
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
# Agent tools — each makes its own LLM call
# ---------------------------------------------------------------------------


async def fundamentals_tool(symbol: str, depth: str = "standard") -> str:
    """Run fundamental analysis on a stock (Piotroski, Magic Formula, DCF, moat).

    Each call makes its own LLM call with agent-specific context.
    Returns structured analysis with ratios, health score, and recommendation.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (skip DCF/peers), standard (all but DCF), deep (everything)
    """
    from src.autfin.llm.client import get_llm_client
    from src.autfin.agents.fundamentals import FundamentalsAgent
    from src.autfin.agents.tools import FUNDAMENTALS_TOOLS, FUNDAMENTALS_CALLABLES

    llm = get_llm_client()
    if not llm.is_available:
        return json.dumps({"status": "error", "message": "LLM unavailable"})

    # Build agent-specific system prompt
    system_prompt = (
        f"You are an expert fundamental analyst covering Indian equities on NSE.\n"
        f"Analyze {symbol} at depth={depth}.\n\n"
        "Use your tools to gather financial data, then produce a COMPLETE analysis.\n\n"
        "Available tools:\n"
        "- fetch_financial_info(symbol): All ratios and metrics\n"
        "- fetch_balance_sheet(symbol): Balance sheet\n"
        "- fetch_income_statement(symbol): Income statement\n"
        "- fetch_cashflow(symbol): Cash flow\n"
        "- fetch_company_info(symbol): Company profile\n"
        "- compare_peers(symbol, metric): Peer comparison\n"
        "- compute_dcf_valuation(fcf, growth, shares): DCF valuation\n\n"
        "STEP 1: Call fetch_financial_info to get all ratios.\n"
        "STEP 2: Call fetch_company_info for sector context.\n"
    )

    if depth == "deep":
        system_prompt += (
            "STEP 3: Call fetch_balance_sheet, fetch_income_statement, fetch_cashflow.\n"
            "STEP 4: Call compute_dcf_valuation (needs FCF, growth rate, shares outstanding).\n"
            "STEP 5: Call compare_peers for sector comparison.\n"
        )
    elif depth == "standard":
        system_prompt += (
            "STEP 3: Call fetch_balance_sheet for debt analysis.\n"
        )
    # quick: just fetch_financial_info + fetch_company_info

    system_prompt += (
        "\nAfter ALL tool calls complete, output a SINGLE JSON object:\n"
        '{"pe_ratio": float, "pb_ratio": float, "ev_ebitda": float, '
        '"roe": float, "roa": float, "net_margin": float, "operating_margin": float, '
        '"debt_equity": float, "revenue_growth": float, "eps_growth": float, '
        '"sector": str, "market_cap": float, '
        '"health_score": int(0-100), "moat_level": str, '
        '"recommendation": str, "conviction": float(0.0-1.0), "strength": int(-5 to +5), '
        '"risks": list, "catalysts": list, "summary": str}\n\n'
        "Output ONLY the raw JSON — no markdown, no explanation."
    )

    raw = await llm.generate_with_tools(
        system_prompt=system_prompt,
        tools=FUNDAMENTALS_TOOLS,
        tool_callables=FUNDAMENTALS_CALLABLES,
        model=llm.deep_model,
        max_tokens=4096,
        max_iterations=15,
    )

    if not raw:
        # Deterministic fallback
        try:
            agent = FundamentalsAgent()
            report = await agent.analyze(symbol)
            return json.dumps(report, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    from src.autfin.utils import extract_json
    result = extract_json(raw)
    if result:
        return json.dumps(result)
    return json.dumps({"status": "success", "symbol": symbol, "raw_analysis": raw[:2000]})


async def technical_tool(symbol: str, depth: str = "standard") -> str:
    """Run technical analysis on a stock (RSI, MACD, support/resistance, patterns).

    Each call makes its own LLM call with agent-specific context.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (RSI/MACD only), standard (all indicators), deep (patterns + multi-timeframe)
    """
    from src.autfin.llm.client import get_llm_client
    from src.autfin.agents.tools import TECHNICAL_TOOLS, TECHNICAL_CALLABLES

    llm = get_llm_client()
    if not llm.is_available:
        return json.dumps({"status": "error", "message": "LLM unavailable"})

    system_prompt = (
        f"You are an expert technical analyst. Analyze {symbol} on NSE.\n"
        "Use your tools to fetch price data, compute indicators, and detect patterns.\n\n"
        "Available tools:\n"
        "- fetch_historical_ohlcv(symbol, period, interval): Price history\n"
        "- compute_sma(prices_json, periods): Simple Moving Average\n"
        "- compute_rsi(prices_json, period): RSI\n"
        "- compute_macd(prices_json): MACD\n"
        "- compute_bollinger_bands(prices_json): Bollinger Bands\n"
        "- compute_atr(high_json, low_json, close_json): ATR\n"
        "- detect_candlestick_patterns(high_json, low_json, open_json, close_json): Patterns\n"
        "- compute_volume_analysis(volume_json, price_json): Volume analysis\n\n"
    )

    if depth == "deep":
        system_prompt += (
            "STEP 1: fetch_historical_ohlcv(symbol, '1y', '1d') for long-term data.\n"
            "STEP 2: Compute SMA(20,50,200), RSI, MACD, Bollinger, ATR.\n"
            "STEP 3: detect_candlestick_patterns for recent candle signals.\n"
            "STEP 4: compute_volume_analysis for volume confirmation.\n"
        )
    elif depth == "standard":
        system_prompt += (
            "STEP 1: fetch_historical_ohlcv(symbol, '6mo', '1d').\n"
            "STEP 2: Compute SMA(20,50,200), RSI, MACD.\n"
        )
    else:  # quick
        system_prompt += (
            "STEP 1: fetch_historical_ohlcv(symbol, '3mo', '1d').\n"
            "STEP 2: Compute RSI, MACD only.\n"
        )

    system_prompt += (
        "\nAfter ALL tool calls, output a SINGLE JSON:\n"
        '{"sma_20": float, "sma_50": float, "sma_200": float, '
        '"rsi": float, "macd": float, "macd_signal": float, "macd_histogram": float, '
        '"bollinger_upper": float, "bollinger_middle": float, "bollinger_lower": float, '
        '"atr": float, "volume_trend": str, '
        '"candlestick_patterns": list, '
        '"signal": "STRONG_BUY"/"BUY"/"HOLD"/"SELL"/"STRONG_SELL", '
        '"strength": int(-5 to +5), "conviction": float(0.0-1.0), "summary": str}\n\n'
        "Output ONLY the raw JSON."
    )

    raw = await llm.generate_with_tools(
        system_prompt=system_prompt,
        tools=TECHNICAL_TOOLS,
        tool_callables=TECHNICAL_CALLABLES,
        model=llm.deep_model,
        max_tokens=4096,
        max_iterations=15,
    )

    if not raw:
        return json.dumps({"status": "error", "message": "Technical analysis failed"})

    from src.autfin.utils import extract_json
    result = extract_json(raw)
    if result:
        return json.dumps(result)
    return json.dumps({"status": "success", "symbol": symbol, "raw_analysis": raw[:2000]})


async def sentiment_tool(symbol: str) -> str:
    """Run sentiment analysis on a stock (news, social media, analyst views).

    Each call makes its own LLM call with agent-specific context.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
    """
    from src.autfin.llm.client import get_llm_client
    from src.autfin.agents.tools import SENTIMENT_TOOLS, SENTIMENT_CALLABLES

    llm = get_llm_client()
    if not llm.is_available:
        return json.dumps({"status": "error", "message": "LLM unavailable"})

    system_prompt = (
        f"You are an expert sentiment analyst covering Indian equities.\n"
        f"Analyze sentiment for {symbol}.\n\n"
        "Available tools:\n"
        "- fetch_marketaux_news(symbol, limit): Recent news with sentiment scores\n"
        "- fetch_nse_announcements(symbol): NSE corporate announcements\n"
        "- fetch_company_info(symbol): Company context\n\n"
        "STEP 1: fetch_marketaux_news for recent news sentiment.\n"
        "STEP 2: fetch_nse_announcements for corporate filings.\n"
        "STEP 3: Analyze sentiment from news + announcements.\n\n"
        "Output a SINGLE JSON:\n"
        '{"sentiment_score": float(-1.0 to 1.0), "sentiment_label": "bullish"/"bearish"/"neutral", '
        '"news_sentiment": float, "announcement_sentiment": float, '
        '"catalysts": list, "risks": list, "summary": str}\n\n'
        "Output ONLY the raw JSON."
    )

    raw = await llm.generate_with_tools(
        system_prompt=system_prompt,
        tools=SENTIMENT_TOOLS,
        tool_callables=SENTIMENT_CALLABLES,
        model=llm.deep_model,
        max_tokens=4096,
        max_iterations=10,
    )

    if not raw:
        return json.dumps({"status": "error", "message": "Sentiment analysis failed"})

    from src.autfin.utils import extract_json
    result = extract_json(raw)
    if result:
        return json.dumps(result)
    return json.dumps({"status": "success", "symbol": symbol, "raw_analysis": raw[:2000]})


async def macro_tool() -> str:
    """Run global macro analysis (Fed, oil, DXY, VIX, FII flows, RBI policy).

    Each call makes its own LLM call with agent-specific context.
    No arguments — macro analysis is always portfolio-wide.
    """
    from src.autfin.llm.client import get_llm_client
    from src.autfin.agents.tools import MACRO_TOOLS, MACRO_CALLABLES

    llm = get_llm_client()
    if not llm.is_available:
        return json.dumps({"status": "error", "message": "LLM unavailable"})

    system_prompt = (
        "You are an expert global macro analyst covering India-specific impacts.\n"
        "Analyze current macro conditions affecting Indian equities.\n\n"
        "Available tools:\n"
        "- fetch_macro_indicator(indicator): Yahoo Finance macro data (^GSPC, ^VIX, CL=F, GC=F, ^TNX, USDINR=X)\n"
        "- fetch_fred_indicator(series_id): US economic data (FEDFUNDS, DGS10, UNRATE, CPIAUCSL)\n\n"
        "Fetch: ^VIX (fear), CL=F (oil), USDINR=X (rupee), FEDFUNDS (Fed rate), DGS10 (10Y yield).\n\n"
        "Output a SINGLE JSON:\n"
        '{"impact_signal": "risk_on"/"risk_off"/"neutral", '
        '"vix_level": float, "oil_price": float, "dxy_trend": str, "inr_usd": float, '
        '"fed_stance": str, "sector_impacts": dict, '
        '"india_specific": dict, "narrative": str, "conviction": float(0.0-1.0)}\n\n'
        "Output ONLY the raw JSON."
    )

    raw = await llm.generate_with_tools(
        system_prompt=system_prompt,
        tools=MACRO_TOOLS,
        tool_callables=MACRO_CALLABLES,
        model=llm.deep_model,
        max_tokens=4096,
        max_iterations=10,
    )

    if not raw:
        return json.dumps({"status": "error", "message": "Macro analysis failed"})

    from src.autfin.utils import extract_json
    result = extract_json(raw)
    if result:
        return json.dumps(result)
    return json.dumps({"status": "success", "raw_analysis": raw[:2000]})


async def research_tool(symbols: str, depth: str = "standard") -> str:
    """Run multi-agent research synthesis (Bull/Bear debate) on one or more stocks.

    Calls fundamentals, technical, and sentiment agents, then synthesizes.
    This is the most expensive agent tool — use for final investment decisions.

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


async def risk_tool(trade_json: str) -> str:
    """Run risk assessment on a proposed trade (14 risk checks with veto power).

    Each call makes its own LLM call with risk-specific context.

    Args:
        trade_json: JSON string with trade details: {"symbol": "X", "action": "BUY"/"SELL", "quantity": N, "price": P}
    """
    from src.autfin.llm.client import get_llm_client

    llm = get_llm_client()
    if not llm.is_available:
        return json.dumps({"status": "error", "message": "LLM unavailable"})

    try:
        trade = json.loads(trade_json) if isinstance(trade_json, str) else trade_json
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"status": "error", "message": "Invalid trade JSON"})

    system_prompt = (
        "You are a risk manager at an Indian investment firm.\n"
        f"Evaluate this trade: {json.dumps(trade)}\n\n"
        "Run these risk checks:\n"
        "1. Position sizing (max 5% per position)\n"
        "2. Sector concentration (max 30%)\n"
        "3. Cash reserve (min 10%)\n"
        "4. Risk/reward ratio (min 2:1)\n"
        "5. Stop-loss level (max 8% downside)\n"
        "6. Market hours check\n"
        "7. Liquidity check\n"
        "8. Volatility check\n"
        "9. Correlation check\n"
        "10. Reasonability gate (no penny stocks, no 10x leverage)\n"
        "11. Regime check (market regime impact)\n"
        "12. Tax efficiency (STCG vs LTCG)\n"
        "13. Correlation with existing portfolio\n"
        "14. Kelly criterion sizing\n\n"
        "Output a SINGLE JSON:\n"
        '{"verdict": "approved"/"rejected"/"conditional", '
        '"checks": list, "approved_size": int, "risk_reward": float, '
        '"stop_loss_pct": float, "rationale": str}\n\n'
        "Output ONLY the raw JSON."
    )

    raw = await llm.generate(
        prompt="Evaluate this trade for risk.",
        system_prompt=system_prompt,
        model=llm.deep_model,
        max_tokens=2048,
    )

    if not raw:
        return json.dumps({"status": "error", "message": "Risk assessment failed"})

    from src.autfin.utils import extract_json
    result = extract_json(raw)
    if result:
        return json.dumps(result)
    return json.dumps({"status": "success", "raw_analysis": raw[:2000]})


async def portfolio_tool() -> str:
    """Get current portfolio state with live holdings, cash, and sector exposure.

    Each call makes its own LLM call to synthesize portfolio data.
    No arguments — portfolio is always global state.
    """
    from src.autfin.llm.client import get_llm_client
    from src.autfin.tools.supervisor_tools import get_portfolio_state

    llm = get_llm_client()

    # First get raw portfolio data
    try:
        portfolio_data = await get_portfolio_state()
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to fetch portfolio: {e}"})

    if not llm.is_available:
        return json.dumps(portfolio_data) if isinstance(portfolio_data, dict) else str(portfolio_data)

    system_prompt = (
        "You are a portfolio manager at an Indian investment firm.\n"
        "Analyze the current portfolio state and provide:\n"
        "1. Total portfolio value and P&L\n"
        "2. Sector allocation breakdown\n"
        "3. Cash position and deployment capacity\n"
        "4. Top winners and losers\n"
        "5. Rebalancing suggestions\n\n"
        "Output a SINGLE JSON:\n"
        '{"total_value": float, "total_pnl": float, "cash_available": float, '
        '"sector_allocation": dict, "holdings": list, '
        '"rebalancing_suggestions": list, "summary": str}\n\n'
        "Output ONLY the raw JSON."
    )

    raw = await llm.generate(
        prompt=f"Analyze this portfolio:\n{json.dumps(portfolio_data, default=str) if isinstance(portfolio_data, dict) else str(portfolio_data)}",
        system_prompt=system_prompt,
        model=llm.fast_model,
        max_tokens=2048,
    )

    if raw:
        from src.autfin.utils import extract_json
        result = extract_json(raw)
        if result:
            return json.dumps(result)
    return json.dumps(portfolio_data) if isinstance(portfolio_data, dict) else str(portfolio_data)


# ---------------------------------------------------------------------------
# Tool definitions — OpenAI format for supervisor's tool-calling loop
# ---------------------------------------------------------------------------

AGENT_TOOL_DEFINITIONS: List[Dict] = [
    _tool_def(
        "fundamentals_analysis",
        "Run fundamental analysis on a stock (Piotroski, Magic Formula, DCF, moat). "
        "Returns ratios, health score, and recommendation.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
        },
    ),
    _tool_def(
        "technical_analysis",
        "Run technical analysis on a stock (RSI, MACD, Bollinger, patterns). "
        "Returns indicators, signal, and conviction.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
        },
    ),
    _tool_def(
        "sentiment_analysis",
        "Run sentiment analysis on a stock (news, social media, analyst views). "
        "Returns sentiment score and catalysts.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
        },
    ),
    _tool_def(
        "macro_analysis",
        "Run global macro analysis (Fed, oil, DXY, VIX, FII flows, RBI policy). "
        "Returns impact signal and sector matrix.",
        {},
    ),
    _tool_def(
        "research_synthesis",
        "Run multi-agent research synthesis (Bull/Bear debate) on one or more stocks. "
        "Calls fundamentals, technical, and sentiment internally. Most expensive tool.",
        {
            "symbols": {"type": "string", "description": "Comma-separated symbols (e.g. RELIANCE,TCS)"},
            "depth": {"type": "string", "description": "quick/standard/deep (default: standard)"},
        },
    ),
    _tool_def(
        "risk_assessment",
        "Run risk assessment on a proposed trade (14 risk checks with veto power). "
        "Returns verdict, approved size, and stop-loss.",
        {
            "trade_json": {"type": "string", "description": "JSON trade: {symbol, action, quantity, price}"},
        },
    ),
    _tool_def(
        "portfolio_allocation",
        "Get current portfolio state with live holdings, cash, and sector exposure. "
        "Returns total value, P&L, and rebalancing suggestions.",
        {},
    ),
]

# Callable mapping — name → async function
AGENT_TOOL_CALLABLES: Dict[str, Any] = {
    "fundamentals_analysis": fundamentals_tool,
    "technical_analysis": technical_tool,
    "sentiment_analysis": sentiment_tool,
    "macro_analysis": macro_tool,
    "research_synthesis": research_tool,
    "risk_assessment": risk_tool,
    "portfolio_allocation": portfolio_tool,
}
