"""
Sub-agents for v5.5 — LangGraph subagents pattern.

Architecture:
  Supervisor sees 5 high-level tools:
    1. get_market_data     — MarketDataAgent
    2. analyze_stock       — AnalysisAgent
    3. research_stock      — ResearchAgent
    4. execute_trade       — RiskAgent
    5. manage_memory       — MemoryAgent

Each sub-agent:
  - Has its own system prompt and tools
  - Is stateless (no memory between invocations)
  - Returns only the final result (context isolation)
  - Is invoked via @tool wrapper that the supervisor calls

Benefits:
  - 30 tools → 5 tools on supervisor (cleaner routing)
  - Context isolation (each sub-agent gets clean context)
  - Parallel execution (supervisor can call multiple in one response)
  - Domain-specific prompts (better tool selection within each domain)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: OpenAI-format tool definition
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
# Sub-agent tool definitions — the 5 high-level tools the supervisor sees
# ---------------------------------------------------------------------------


SUBAGENT_TOOLS: List[Dict] = [
    _tool_def(
        "get_market_data",
        "Get portfolio state, market quotes, and market status. "
        "Use for: checking holdings, cash, positions, current prices, market hours. "
        "Returns JSON with available_cash, holdings, positions, market_status, quotes.",
        {
            "data_type": {
                "type": "string",
                "enum": ["portfolio", "quotes", "status", "all"],
                "description": "Type of data to fetch: portfolio (holdings/cash), quotes (current prices), status (market hours), all (everything)",
            },
            "symbols": {
                "type": "string",
                "description": "Comma-separated symbols for quotes (e.g. 'RELIANCE,TCS,INFY'). Ignored for portfolio/status.",
            },
        },
    ),
    _tool_def(
        "analyze_stock",
        "Run comprehensive stock analysis: fundamentals, technicals, sentiment, macro, valuation, distortion, regulatory. "
        "All analysis is deterministic (no LLM). Returns structured analysis results. "
        "Use for: any stock analysis request. Can analyze multiple aspects in one call.",
        {
            "symbol": {
                "type": "string",
                "description": "Stock symbol to analyze (e.g. 'RELIANCE')",
            },
            "analysis_type": {
                "type": "string",
                "enum": ["fundamentals", "technical", "sentiment", "macro", "valuation", "distortion", "regulatory", "full"],
                "description": "Type of analysis: full runs all 7 types, or pick specific ones",
            },
            "depth": {
                "type": "string",
                "enum": ["quick", "standard", "deep"],
                "description": "Analysis depth: quick (skip DCF/peers), standard (all but DCF), deep (everything)",
            },
        },
    ),
    _tool_def(
        "research_stock",
        "Research stocks: screen candidates, deep-dive research, build investment pitches, synthesize bull/bear cases. "
        "research_synthesis is LLM-driven (expensive). Use for: final investment decisions, screening, pitch decks.",
        {
            "action": {
                "type": "string",
                "enum": ["screen", "research", "research_batch", "pitch", "synthesize"],
                "description": "Action: screen (find candidates), research (deep-dive one stock), research_batch (multiple stocks), pitch (investment thesis), synthesize (bull/bear debate)",
            },
            "symbol": {
                "type": "string",
                "description": "Stock symbol for research/pitch actions",
            },
            "symbols": {
                "type": "string",
                "description": "Comma-separated symbols for research_batch",
            },
            "depth": {
                "type": "string",
                "enum": ["quick", "standard", "deep"],
                "description": "Research depth",
            },
            "criteria": {
                "type": "string",
                "description": "Screening criteria for 'screen' action (e.g. 'ROE > 15%, debt < 2x')",
            },
        },
    ),
    _tool_def(
        "execute_trade",
        "Manage risk and execute trades: risk assessment, capital allocation, order execution, thesis validation. "
        "Includes 14 risk checks with veto power. Use for: validating trades, allocating capital, placing orders.",
        {
            "action": {
                "type": "string",
                "enum": ["risk_check", "allocate", "execute", "validate_thesis"],
                "description": "Action: risk_check (validate trade risks), allocate (position sizing), execute (place order), validate_thesis (check thesis integrity)",
            },
            "trade_json": {
                "type": "string",
                "description": "JSON string with trade details: {symbol, action (BUY/SELL), quantity, price, order_type (MARKET/LIMIT)}",
            },
            "thesis": {
                "type": "string",
                "description": "Investment thesis for validate_thesis action",
            },
            "conviction": {
                "type": "number",
                "description": "Conviction score (0-1) for capital allocation",
            },
        },
    ),
    _tool_def(
        "manage_memory",
        "Manage trading memory: retrieve past learnings, log decisions, record outcomes, calibrate conviction, detect patterns. "
        "Use for: accessing historical context, improving future decisions, self-learning.",
        {
            "action": {
                "type": "string",
                "enum": ["retrieve", "log", "record", "calibrate", "detect", "guardian", "approve", "finalize"],
                "description": "Action: retrieve (get past learnings), log (save decision), record (save outcome), calibrate (update conviction), detect (find patterns), guardian (check risk limits), approve (request HITL), finalize (end session)",
            },
            "symbol": {
                "type": "string",
                "description": "Stock symbol for log/record/calibrate actions",
            },
            "decision": {
                "type": "string",
                "description": "Decision details for log action",
            },
            "outcome": {
                "type": "string",
                "description": "Outcome for record action (e.g. 'profit_5%')",
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Sub-agent tool callables — functions the LLM can call
# ---------------------------------------------------------------------------


async def _call_market_data(data_type: str = "all", symbols: str = "") -> str:
    """MarketDataAgent: fetch portfolio, quotes, market status."""
    try:
        from src.autfin.tools.supervisor_tools import (
            get_portfolio_state, get_market_quotes, get_market_status,
        )

        results: Dict[str, Any] = {}

        if data_type in ("portfolio", "all"):
            try:
                portfolio = await get_portfolio_state()
                results["portfolio"] = portfolio
            except Exception as e:
                results["portfolio"] = {"status": "error", "error": str(e)}

        if data_type in ("quotes", "all") and symbols:
            try:
                symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
                quotes = await get_market_quotes(symbol_list)
                results["quotes"] = quotes
            except Exception as e:
                results["quotes"] = {"status": "error", "error": str(e)}

        if data_type in ("status", "all"):
            try:
                status = await get_market_status()
                results["market_status"] = status
            except Exception as e:
                results["market_status"] = {"status": "error", "error": str(e)}

        return json.dumps(results, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _call_analysis(
    symbol: str,
    analysis_type: str = "full",
    depth: str = "standard",
) -> str:
    """AnalysisAgent: run stock analysis (deterministic, no LLM)."""
    try:
        from src.autfin.agents.agent_tools import (
            fundamentals_tool,
            technical_tool,
            sentiment_tool,
            macro_tool,
            valuation_tool,
            distortion_tool,
            regulatory_tool,
        )

        results: Dict[str, Any] = {"symbol": symbol, "analyses": {}}

        analysis_map = {
            "fundamentals": [fundamentals_tool],
            "technical": [technical_tool],
            "sentiment": [sentiment_tool],
            "macro": [macro_tool],
            "valuation": [valuation_tool],
            "distortion": [distortion_tool],
            "regulatory": [regulatory_tool],
            "full": [
                fundamentals_tool,
                technical_tool,
                sentiment_tool,
                distortion_tool,
                regulatory_tool,
                valuation_tool,
                macro_tool,
            ],
        }

        tools_to_run = analysis_map.get(analysis_type, analysis_map["full"])

        import asyncio
        tasks = []
        for tool_fn in tools_to_run:
            fn_name = tool_fn.__name__
            label = fn_name.replace("_tool", "")
            if fn_name == "macro_tool":
                tasks.append((label, tool_fn()))
            else:
                tasks.append((label, tool_fn(symbol, depth)))

        gathered = await asyncio.gather(
            *[t[1] for t in tasks],
            return_exceptions=True,
        )

        for (name, _), result in zip(tasks, gathered):
            if isinstance(result, Exception):
                results["analyses"][name] = {"status": "error", "error": str(result)}
            else:
                results["analyses"][name] = result

        return json.dumps(results, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _call_research(
    action: str,
    symbol: str = "",
    symbols: str = "",
    depth: str = "standard",
    criteria: str = "",
) -> str:
    """ResearchAgent: screen, research, pitch, synthesize."""
    try:
        from src.autfin.tools.supervisor_tools import (
            screen_candidates,
            research_stock,
            research_batch,
            build_stock_pitch,
        )
        from src.autfin.agents.agent_tools import research_tool as research_synthesis

        if action == "screen":
            result = await screen_candidates(criteria or "ROE > 15%")
            return json.dumps({"action": "screen", "result": result}, default=str)

        elif action == "research" and symbol:
            result = await research_stock(symbol, depth)
            return json.dumps({"action": "research", "symbol": symbol, "result": result}, default=str)

        elif action == "research_batch" and symbols:
            symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
            result = await research_batch(symbol_list, depth)
            return json.dumps({"action": "research_batch", "result": result}, default=str)

        elif action == "pitch" and symbol:
            result = await build_stock_pitch(symbol, depth)
            return json.dumps({"action": "pitch", "symbol": symbol, "result": result}, default=str)

        elif action == "synthesize" and symbol:
            # research_tool takes a comma-separated symbols string
            result = await research_synthesis(symbol, depth)
            return json.dumps({"action": "synthesize", "symbol": symbol, "result": result}, default=str)

        else:
            return json.dumps({"status": "error", "error": f"Invalid action '{action}' or missing required params"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _call_risk(
    action: str,
    trade_json: str = "",
    thesis: str = "",
    conviction: float = 0.5,
) -> str:
    """RiskAgent: risk checks, allocation, execution, thesis validation."""
    try:
        from src.autfin.tools.supervisor_tools import (
            run_risk_checks,
            allocate_capital,
            execute_order,
            validate_thesis,
        )

        if action == "risk_check" and trade_json:
            trade = json.loads(trade_json) if isinstance(trade_json, str) else trade_json
            result = await run_risk_checks(trade)
            return json.dumps({"action": "risk_check", "result": result}, default=str)

        elif action == "allocate":
            result = await allocate_capital(conviction)
            return json.dumps({"action": "allocate", "conviction": conviction, "result": result}, default=str)

        elif action == "execute" and trade_json:
            trade = json.loads(trade_json) if isinstance(trade_json, str) else trade_json
            result = await execute_order(trade)
            return json.dumps({"action": "execute", "result": result}, default=str)

        elif action == "validate_thesis" and thesis:
            result = await validate_thesis(thesis)
            return json.dumps({"action": "validate_thesis", "result": result}, default=str)

        else:
            return json.dumps({"status": "error", "error": f"Invalid action '{action}' or missing required params"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _call_memory(
    action: str,
    symbol: str = "",
    decision: str = "",
    outcome: str = "",
) -> str:
    """MemoryAgent: learnings, logging, calibration, patterns, guardian, HITL, session."""
    try:
        from src.autfin.tools.supervisor_tools import (
            retrieve_learnings,
            log_decision,
            record_learning_tool,
            calibrate_conviction_tool,
            detect_patterns_tool,
            check_guardian,
            request_human_approval_tool,
            finalize_session,
        )

        if action == "retrieve":
            result = await retrieve_learnings()
            return json.dumps({"action": "retrieve", "result": result}, default=str)

        elif action == "log" and decision:
            result = await log_decision(symbol, decision)
            return json.dumps({"action": "log", "symbol": symbol, "result": result}, default=str)

        elif action == "record" and symbol and outcome:
            # record_learning_tool(symbol, action, conviction, pnl_pct, thesis_summary, outcome)
            # Default conviction=0.5, action from outcome prefix or "BUY"
            action_side = "BUY" if outcome.lower().startswith("buy") else (
                "SELL" if outcome.lower().startswith("sell") else "BUY"
            )
            conv = float(decision) if decision else 0.5
            result = await record_learning_tool(
                symbol=symbol,
                action=action_side,
                conviction=conv,
                pnl_pct=0.0,
                thesis_summary=decision or f"Recorded outcome {outcome} for {symbol}",
                outcome=outcome,
            )
            return json.dumps({"action": "record", "symbol": symbol, "result": result}, default=str)

        elif action == "calibrate" and symbol:
            result = await calibrate_conviction_tool(symbol)
            return json.dumps({"action": "calibrate", "symbol": symbol, "result": result}, default=str)

        elif action == "detect":
            result = await detect_patterns_tool()
            return json.dumps({"action": "detect", "result": result}, default=str)

        elif action == "guardian":
            result = await check_guardian()
            return json.dumps({"action": "guardian", "result": result}, default=str)

        elif action == "approve":
            result = await request_human_approval_tool(
                question="Trade requires human approval",
                trade_details={"symbol": symbol, "notes": decision},
            )
            return json.dumps({"action": "approve", "result": result, "symbol": symbol}, default=str)

        elif action == "finalize":
            result = await finalize_session()
            return json.dumps({"action": "finalize", "result": result}, default=str)

        else:
            return json.dumps({"status": "error", "error": f"Invalid action '{action}' or missing required params"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# Sub-agent callable map — name → async function
# ---------------------------------------------------------------------------


SUBAGENT_CALLABLES: Dict[str, Callable] = {
    "get_market_data": _call_market_data,
    "analyze_stock": _call_analysis,
    "research_stock": _call_research,
    "execute_trade": _call_risk,
    "manage_memory": _call_memory,
}
