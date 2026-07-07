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
            research_data = {"depth": depth, "symbol": symbol, "source": "supervisor"}
            result = await build_stock_pitch(symbol, research_data)
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
            trades = json.loads(trade_json) if trade_json else [{"conviction": conviction}]
            available_cash = float(conviction) * 1_000_000  # scale by conviction
            result = await allocate_capital(approved_trades=trades, available_cash=available_cash)
            return json.dumps({"action": "allocate", "conviction": conviction, "result": result}, default=str)

        elif action == "execute" and trade_json:
            trade = json.loads(trade_json) if isinstance(trade_json, str) else trade_json
            result = await execute_order(trade)
            return json.dumps({"action": "execute", "result": result}, default=str)

        elif action == "validate_thesis" and thesis:
            sym = ""
            if trade_json:
                try:
                    trade = json.loads(trade_json) if isinstance(trade_json, str) else trade_json
                    sym = trade.get("symbol", "")
                except (json.JSONDecodeError, TypeError):
                    pass
            result = await validate_thesis(
                symbol=sym,
                holding_data={"thesis": thesis, "symbol": sym},
            )
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
            request_human_approval,
            finalize_session,
        )

        if action == "retrieve":
            query = symbol or decision or "general learnings"
            result = await retrieve_learnings(query)
            return json.dumps({"action": "retrieve", "result": result}, default=str)

        elif action == "log" and decision:
            result = await log_decision(
                decision_type=symbol or "trade",
                details={"summary": decision, "symbol": symbol},
            )
            return json.dumps({"action": "log", "symbol": symbol, "result": result}, default=str)

        elif action == "record" and symbol and outcome:
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
            raw_conv = float(decision) if decision else 0.5
            result = await calibrate_conviction_tool(raw_conviction=raw_conv)
            return json.dumps({"action": "calibrate", "symbol": symbol, "result": result}, default=str)

        elif action == "detect":
            result = await detect_patterns_tool()
            return json.dumps({"action": "detect", "result": result}, default=str)

        elif action == "guardian":
            result = await check_guardian()
            return json.dumps({"action": "guardian", "result": result}, default=str)

        elif action == "approve":
            result = await request_human_approval(
                question=decision or "Trade requires human approval",
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
    # v5.6: 8-Agent System additions
    "run_debate": _call_debate,
    "run_conference": _call_conference,
    "manage_memory_system": _call_memory_system,
}


# ---------------------------------------------------------------------------
# v5.6: 8-Agent System — Debate, Conferences, Memory
# ---------------------------------------------------------------------------

# Debate tool definition
DEBATE_TOOL = _tool_def(
    "run_debate",
    "Run adversarial bull/bear debate on a stock. Two researchers argue "
    "opposing positions before every trade decision. Based on TradingAgents "
    "(Columbia/NYU) — ablation studies show systems without debate "
    "underperform by 20-40%.",
    {
        "symbol": {
            "type": "string",
            "description": "Stock symbol to debate (e.g. 'RELIANCE')",
        },
        "analysis_data": {
            "type": "string",
            "description": "JSON string with analysis results from analyst team (fundamentals, technical, sentiment)",
        },
        "additional_context": {
            "type": "string",
            "description": "Optional additional context for the debate (news, catalysts, etc.)",
        },
    },
)

# Conference tool definition
CONFERENCE_TOOL = _tool_def(
    "run_conference",
    "Run a conference to coordinate agents. Four types: "
    "morning (daily capital allocation), trade_review (bull+bear debate + risk gates), "
    "emergency (triggered by extreme market moves), experience (weekly knowledge sharing).",
    {
        "conference_type": {
            "type": "string",
            "enum": ["morning", "trade_review", "emergency", "experience"],
            "description": "Type of conference to run",
        },
        "symbol": {
            "type": "string",
            "description": "Stock symbol for trade_review conference",
        },
        "analyst_reports": {
            "type": "string",
            "description": "JSON string with analyst reports for morning conference",
        },
        "trigger_description": {
            "type": "string",
            "description": "Trigger description for emergency conference (e.g. 'Nifty crashed 4%')",
        },
    },
)

# Memory system tool definition
MEMORY_SYSTEM_TOOL = _tool_def(
    "manage_memory_system",
    "Manage 3-type persistent memory: market (what happened), reflection "
    "(what worked/failed), experience (patterns that repeat). All memories "
    "persist across sessions and compound over time.",
    {
        "action": {
            "type": "string",
            "enum": ["record_market", "record_trade", "record_pattern", "get_context", "get_calibration", "get_stats"],
            "description": "Action: record_market (save market event), record_trade (save trade outcome), record_pattern (save pattern), get_context (get all context for decision), get_calibration (get conviction accuracy), get_stats (memory statistics)",
        },
        "symbol": {
            "type": "string",
            "description": "Stock symbol (required for most actions)",
        },
        "data": {
            "type": "string",
            "description": "JSON string with data to record (event, trade outcome, or pattern)",
        },
    },
)


async def _call_debate(
    symbol: str,
    analysis_data: str,
    additional_context: str = "",
) -> str:
    """Run adversarial bull/bear debate on a stock."""
    try:
        from src.autfin.agents.debate import run_debate, format_debate_result

        result = await run_debate(
            symbol=symbol,
            analysis_data=analysis_data,
            additional_context=additional_context,
        )

        return json.dumps({
            "symbol": result.symbol,
            "verdict": result.verdict,
            "final_conviction": result.final_conviction,
            "confidence_in_verdict": result.confidence_in_verdict,
            "bull_thesis": result.bull_argument.thesis,
            "bull_conviction": result.bull_argument.conviction,
            "bear_thesis": result.bear_argument.thesis,
            "bear_conviction": result.bear_argument.conviction,
            "key_insights": result.key_insights,
            "unresolved_questions": result.unresolved_questions,
            "formatted": format_debate_result(result),
        }, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _call_conference(
    conference_type: str,
    symbol: str = "",
    analyst_reports: str = "",
    trigger_description: str = "",
) -> str:
    """Run a conference to coordinate agents."""
    try:
        from src.autfin.agents.conferences import (
            run_morning_conference,
            run_trade_review,
            run_emergency_conference,
            run_experience_conference,
            MarketRegime,
        )

        if conference_type == "morning":
            reports_data = json.loads(analyst_reports) if analyst_reports else []
            result = await run_morning_conference(
                analyst_reports=reports_data,
                portfolio=[],  # will be filled by supervisor
                available_cash=0,  # will be filled by supervisor
            )
            return json.dumps({
                "conference_type": "morning",
                "regime": result.regime.value,
                "decisions": result.decisions,
                "capital_allocation": result.capital_allocation,
                "risk_alerts": result.risk_alerts,
            }, default=str)

        elif conference_type == "trade_review":
            result = await run_trade_review(
                symbol=symbol,
                bull_argument=analyst_reports,  # reusing field for bull argument
                bear_argument=trigger_description,  # reusing field for bear argument
                risk_assessment="",
                current_position="",
                available_cash=0,
                sector_exposure="",
            )
            return json.dumps({
                "conference_type": "trade_review",
                "symbol": symbol,
                "decision": result.get("decision"),
                "reasoning": result.get("reasoning"),
            }, default=str)

        elif conference_type == "emergency":
            result = await run_emergency_conference(
                trigger_description=trigger_description or "Emergency triggered",
                regime=MarketRegime.HIGH_VOLATILITY,
                portfolio=[],
                available_cash=0,
            )
            return json.dumps({
                "conference_type": "emergency",
                "urgency": "high",
                "decisions": result.decisions,
                "risk_alerts": result.risk_alerts,
            }, default=str)

        elif conference_type == "experience":
            result = await run_experience_conference(
                agent_experiences=[],
                trade_outcomes=[],
            )
            return json.dumps({
                "conference_type": "experience",
                "patterns": result.get("patterns_identified", []),
                "calibration_updates": result.get("calibration_updates", []),
            }, default=str)

        else:
            return json.dumps({"status": "error", "error": f"Unknown conference type: {conference_type}"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


async def _call_memory_system(
    action: str,
    symbol: str = "",
    data: str = "",
) -> str:
    """Manage 3-type persistent memory."""
    try:
        from src.autfin.agents.memory_system import MemorySystem

        memory = MemorySystem()

        if action == "record_market":
            event_data = json.loads(data) if data else {}
            entry = await memory.record_market_event(
                event=event_data.get("event", ""),
                symbols=[symbol] if symbol else event_data.get("symbols", []),
                regime=event_data.get("regime", "normal"),
                confidence=event_data.get("confidence", 0.8),
            )
            return json.dumps({"action": "record_market", "key": entry.key}, default=str)

        elif action == "record_trade":
            trade_data = json.loads(data) if data else {}
            entry = await memory.record_trade_outcome(
                symbol=symbol,
                action=trade_data.get("action", "BUY"),
                entry_price=trade_data.get("entry_price", 0),
                exit_price=trade_data.get("exit_price", 0),
                thesis=trade_data.get("thesis", ""),
                outcome=trade_data.get("outcome", ""),
                pnl_pct=trade_data.get("pnl_pct", 0),
                lessons=trade_data.get("lessons", []),
            )
            return json.dumps({"action": "record_trade", "key": entry.key}, default=str)

        elif action == "record_pattern":
            pattern_data = json.loads(data) if data else {}
            entry = await memory.record_pattern(
                pattern=pattern_data.get("pattern", ""),
                symbols=pattern_data.get("symbols", [symbol] if symbol else []),
                confidence=pattern_data.get("confidence", 0.7),
                evidence=pattern_data.get("evidence", ""),
            )
            return json.dumps({"action": "record_pattern", "key": entry.key}, default=str)

        elif action == "get_context":
            context = await memory.get_context_for_decision(
                symbol=symbol,
                decision_type=data or "general",
            )
            return json.dumps({"action": "get_context", "context": context}, default=str)

        elif action == "get_calibration":
            calibration = await memory.get_calibration_data()
            return json.dumps({"action": "get_calibration", "calibration": calibration}, default=str)

        elif action == "get_stats":
            stats = await memory.get_stats()
            return json.dumps({
                "action": "get_stats",
                "total_memories": stats.total_memories,
                "by_type": stats.by_type,
            }, default=str)

        else:
            return json.dumps({"status": "error", "error": f"Unknown action: {action}"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
