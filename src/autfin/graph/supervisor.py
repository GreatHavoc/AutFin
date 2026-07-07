"""
Supervisor Graph — v5.5 Subagents Coordinator.

Architecture:
  START → supervisor_node → {supervisor_node, human_review_node, END}
  human_review_node → supervisor_node (return after input)

The supervisor is the "Subagents Coordinator" — it calls 5 sub-agents
(MarketData, Analysis, Research, Risk, Memory), observes results, and
decides next steps. Each sub-agent has its own tools and system prompt.

Sub-agents provide:
  - Context isolation (clean context window per invocation)
  - Parallel execution (multiple sub-agents in one LLM response)
  - Domain-specific prompts (better tool selection)
  - 30 tools → 5 tools on supervisor (cleaner routing)

  morning_routine  → full portfolio scan + screening + research + trade
  price_alert      → investigate one symbol (3-agent quick fan-out)
  quarterly_review → deep dive all holdings with DCF
  manual           → user-requested single stock analysis

Self-learning: learning context (calibration, patterns) is preloaded into
state on first iteration and injected into the system prompt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from src.autfin.state.agent_state import AgentState
from src.autfin.graph.error_handler import safe_node, RetryPolicy


# ---------------------------------------------------------------------------
# Runtime context schema — passed to all nodes via Runtime[Context]
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """Runtime context injected into all nodes via Runtime[Context].

    Provides access to:
    - user_id: Current user identifier
    - store: LangGraph BaseStore for cross-thread memory
    - mode: Execution mode (paper/live)
    """
    user_id: str = "default"
    mode: str = "paper"
    store: Optional[Any] = field(default=None, repr=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_GRAPH_ITERATIONS = 20
"""Safety limit on supervisor-node loop iterations within a single run."""

# ---------------------------------------------------------------------------
# Context prompt builder
# ---------------------------------------------------------------------------


def _build_context_prompt(context: str, depth: str, state: AgentState) -> str:
    """
    Build the supervisor system prompt based on the invocation context.

    Returns a prompt that instructs the LLM how to behave in its ReAct loop
    for the given context (morning_routine, price_alert, quarterly_review,
    or manual).  Includes live portfolio snapshot where relevant.
    """
    base = """You are the AutFin v5.5 Coordinator — a PM-level AI running a ReAct loop.

You have 5 SUB-AGENTS, each a specialized domain expert. Call them as tools.
Each sub-agent has its own tools and returns structured results.

SUB-AGENTS (call these as tools):

1. get_market_data(data_type, symbols) — MarketDataAgent
   Portfolio state, current quotes, market status.
   data_type: "portfolio" | "quotes" | "status" | "all"
   Use for: checking holdings, cash, prices, market hours.

2. analyze_stock(symbol, analysis_type, depth) — AnalysisAgent
   Deterministic analysis (no LLM, fast ~3-5s).
   analysis_type: "fundamentals" | "technical" | "sentiment" | "macro" | "valuation" | "distortion" | "regulatory" | "full"
   depth: "quick" | "standard" | "deep"
   Use for: any stock analysis. Call multiple types in parallel.

3. run_debate(symbol, analysis_data, additional_context) — Debate Agent
   Run adversarial bull/bear debate. Two researchers argue opposing positions.
   Based on TradingAgents (Columbia/NYU) — debate outperforms single-agent by 20-40%.
   Use BEFORE any trade decision. Returns verdict (buy/sell/hold) with both arguments.

3. research_stock(action, symbol, symbols, depth, criteria) — ResearchAgent
   Stock research and screening. research_synthesis is LLM-driven (expensive).
   action: "screen" | "research" | "research_batch" | "pitch" | "synthesize"
   Use for: finding candidates, deep-dive, investment pitches, bull/bear debate.

4. execute_trade(action, trade_json, thesis, conviction) — RiskAgent
   Risk checks, capital allocation, order execution, thesis validation.
   action: "risk_check" | "allocate" | "execute" | "validate_thesis"
   Use for: validating trades, sizing positions, placing orders.

5. manage_memory(action, symbol, decision, outcome) — MemoryAgent
   Trading memory, learning, patterns, guardian, HITL, session management.
   action: "retrieve" | "log" | "record" | "calibrate" | "detect" | "guardian" | "approve" | "finalize"
   Use for: accessing history, improving decisions, ending sessions.

6. run_conference(conference_type, symbol, analyst_reports, trigger_description) — Conference Agent
   Coordinate agents via conferences. Four types:
   - morning: daily capital allocation (all analysts report)
   - trade_review: bull+bear debate + risk gates (for specific stock)
   - emergency: triggered by extreme market moves (>3% intraday)
   - experience: weekly knowledge sharing (patterns, calibration)
   Use for: coordinating multiple agents, making group decisions.

7. manage_memory_system(action, symbol, data) — Memory System Agent
   3-type persistent memory: market (what happened), reflection (what worked/failed), experience (patterns).
   action: "record_market" | "record_trade" | "record_pattern" | "get_context" | "get_calibration" | "get_stats"
   Use for: accessing cross-session memory, improving decisions over time.

STRATEGY — BATCH sub-agent calls in a SINGLE response for maximum speed:
All sub-agents run in parallel via asyncio.gather — calling 3 together takes ~5s, not ~15s.

CORRECT (fast, ~5s):
  Response: tool_calls=[analyze_stock("RELIANCE", "fundamentals"), analyze_stock("RELIANCE", "technical"), analyze_stock("RELIANCE", "sentiment")]
  → All 3 run in parallel. One response. Fast.

WRONG (slow, ~15s):
  Response 1: tool_calls=[analyze_stock("RELIANCE", "fundamentals")]
  Response 2: tool_calls=[analyze_stock("RELIANCE", "technical")]
  → 2 sequential LLM round-trips. Slow.

Batching patterns:
- Morning routine: get_market_data("all") + run_conference("morning") + analyze_stock(symbol, "full") (all in ONE response)
  → then run_debate(symbol, analysis_data) → execute_trade("risk_check") → execute_trade("execute")
- Price alert: analyze_stock(symbol, "technical") + analyze_stock(symbol, "sentiment") + analyze_stock(symbol, "distortion") (3 in ONE response)
  → then run_debate(symbol, analysis_data) if trade considered
- Quarterly review: analyze_stock(holding, "deep") on EVERY holding (all in ONE response)
  → then run_debate(holding, analysis_data) on each → manage_memory_system("record_trade", outcome)
- Manual: analyze_stock(symbol, "full") + run_debate(symbol, analysis_data) (2 in ONE response)
  → then execute_trade("risk_check") → execute_trade("execute")

RULES:
- ALWAYS batch sub-agent calls in a single response. Never call them one at a time.
- ALWAYS run_debate BEFORE any trade decision. No exceptions.
- Use run_conference("emergency") if market moves > 3% intraday.
- Use analyze_stock("distortion") to check for price anomalies BEFORE recommending a trade.
- Use analyze_stock("regulatory") to check for SEBI/RBI events BEFORE recommending a trade.
- Depth tiers: quick=skip DCF/peers/moat. standard=all but DCF. deep=everything.
- Respect risk/reward ≥ 2:1. Conviction ≥ 0.5 to trade.
- Sector concentration max 30%. Cash reserve 10%.
- If human approval needed → call manage_memory("approve").
- End session → call manage_memory("finalize").
- Record trade outcomes in memory_system for future learning."""

    context_specific = ""

    if context == "morning_routine":
        holdings = state.get("holdings", [])
        cash = state.get("available_cash", 0)
        context_specific = (
            f"\nMORNING ROUTINE at {datetime.now().isoformat()}. "
            f"Run full scan with depth={depth}. "
            f"Holdings: {len(holdings)}. Cash: ₹{cash:,.2f}"
        )
    elif context == "price_alert":
        symbol = state.get("alert_symbol", "N/A")
        pct = state.get("price_move_pct", 0)
        context_specific = (
            f"\nPRICE ALERT: {symbol} moved {pct:.1%}. Investigate immediately."
        )
    elif context == "quarterly_review":
        holdings = state.get("holdings", [])
        context_specific = (
            f"\nQUARTERLY REVIEW. Deep dive every holding ({len(holdings)} positions). "
            f"depth={depth}."
        )
    elif context == "manual":
        context_specific = f"\nMANUAL REQUEST. Analyze the requested stock at depth={depth}."
    else:
        logger.warning("Supervisor: unknown context '%s' — falling back to manual", context)
        context_specific = f"\nUNKNOWN CONTEXT '{context}'. Treat as manual analysis at depth={depth}."
        context = "manual"

    return base + context_specific + _learning_section(state)


def _learning_section(state: AgentState) -> str:
    """Build a short learning context section for the system prompt."""
    lc = state.get("learning_context", {})
    if not lc:
        return ""

    lines = ["\nSELF-LEARNING CONTEXT (auto-loaded):"]

    calibration = lc.get("calibration", {})
    if calibration:
        lines.append("Conviction calibration:")
        for tier, data in calibration.items():
            if isinstance(data, dict):
                lines.append(
                    f"  {tier}: actual={data.get('actual_win_rate', '?')}, "
                    f"expected={data.get('expected_win_rate', '?')}, "
                    f"delta={data.get('delta', '?')} ({data.get('status', '?')})"
                )

    patterns = lc.get("patterns", [])
    if patterns:
        lines.append("Detected patterns:")
        for p in patterns[:5]:
            if isinstance(p, dict):
                lines.append(f"  - {p.get('pattern', p.get('type', '?'))}: {p.get('recommendation', p.get('assessment', ''))}")

    recent = lc.get("recent_trades", [])
    if recent:
        lines.append(f"Recent trades in journal: {len(recent)}")

    if len(lines) == 1:
        return ""

    lines.append("Use calibrate_conviction and detect_patterns tools to get latest data.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory nodes — cross-session long-term memory via LangGraph Store
# ---------------------------------------------------------------------------


async def load_memories(state: AgentState, runtime: Runtime[Context]) -> dict[str, Any]:
    """
    Load relevant memories from the LangGraph Store before research.

    Searches for:
    - Agent instructions (procedural memory)
    - Past trade outcomes for the current symbol(s) (episodic memory)
    - User preferences (semantic memory)

    Injects found memories into state["context_memory"] for the supervisor
    to reference during its ReAct loop.

    Uses Runtime[Context] pattern for store access (LangGraph best practice).
    """
    from src.autfin.memory.langgraph_store import (
        NS_AGENT_INSTRUCTIONS, NS_TRADE_HISTORY, NS_USER_PREFERENCES,
        store_search,
    )

    store = runtime.store
    if store is None:
        return {"context_memory": {}}

    context_memory: dict[str, Any] = {}

    # Load agent instructions
    try:
        instructions = await store_search(store, NS_AGENT_INSTRUCTIONS, "trading rules", limit=5)
        if instructions:
            context_memory["agent_instructions"] = [i["value"] for i in instructions]
    except Exception:
        pass

    # Load past trade outcomes for symbols in state
    symbols = []
    for h in state.get("holdings", []):
        sym = h.get("tradingsymbol") or h.get("symbol")
        if sym:
            symbols.append(sym)
    alert_sym = state.get("alert_symbol")
    if alert_sym:
        symbols.append(alert_sym)

    trade_history = []
    for sym in symbols[:5]:  # limit lookups
        try:
            trades = await store_search(store, NS_TRADE_HISTORY, sym, limit=3)
            trade_history.extend(trades)
        except Exception:
            pass
    if trade_history:
        context_memory["past_trades"] = [t["value"] for t in trade_history]

    # Load user preferences
    try:
        prefs = await store_search(store, NS_USER_PREFERENCES, "preferences", limit=5)
        if prefs:
            context_memory["user_preferences"] = [p["value"] for p in prefs]
    except Exception:
        pass

    return {"context_memory": context_memory}


async def load_memories_legacy(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Legacy version of load_memories for backward compatibility.

    Uses config-based store access instead of Runtime[Context].
    """
    from src.autfin.memory.langgraph_store import (
        NS_AGENT_INSTRUCTIONS, NS_TRADE_HISTORY, NS_USER_PREFERENCES,
        store_search,
    )

    store = config.get("store") if config else None
    if store is None:
        return {"context_memory": {}}

    context_memory: dict[str, Any] = {}

    try:
        instructions = await store_search(store, NS_AGENT_INSTRUCTIONS, "trading rules", limit=5)
        if instructions:
            context_memory["agent_instructions"] = [i["value"] for i in instructions]
    except Exception:
        pass

    symbols = []
    for h in state.get("holdings", []):
        sym = h.get("tradingsymbol") or h.get("symbol")
        if sym:
            symbols.append(sym)
    alert_sym = state.get("alert_symbol")
    if alert_sym:
        symbols.append(alert_sym)

    trade_history = []
    for sym in symbols[:5]:
        try:
            trades = await store_search(store, NS_TRADE_HISTORY, sym, limit=3)
            trade_history.extend(trades)
        except Exception:
            pass
    if trade_history:
        context_memory["past_trades"] = [t["value"] for t in trade_history]

    try:
        prefs = await store_search(store, NS_USER_PREFERENCES, "preferences", limit=5)
        if prefs:
            context_memory["user_preferences"] = [p["value"] for p in prefs]
    except Exception:
        pass

    return {"context_memory": context_memory}


async def save_memory(state: AgentState, runtime: Runtime[Context] | None = None, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Save research outcomes and lessons learned to the LangGraph Store.

    Called after the supervisor completes a session. Persists:
    - Trade decisions and outcomes (episodic memory)
    - Key research findings (semantic memory)

    Uses Runtime[Context] pattern for store access (LangGraph best practice).
    Falls back to config-based access for legacy callers.
    """
    from datetime import datetime, timezone
    from src.autfin.memory.langgraph_store import (
        NS_TRADE_HISTORY, NS_RESEARCH_CACHE,
        store_put,
    )

    # Get store from runtime or config
    store = None
    if runtime is not None:
        store = runtime.store
    elif config is not None:
        store = config.get("store")

    if store is None:
        return {}

    saved = 0

    # Save trade decisions from this session
    for msg in state.get("messages", []):
        content = getattr(msg, "content", "") or ""
        if not content:
            continue
        # Look for trade execution signals in messages
        if any(kw in content.upper() for kw in ("EXECUTED", "BUY", "SELL", "ORDER")):
            key = f"session_{state.get('thread_id', 'unknown')}_{saved}"
            try:
                await store_put(store, NS_TRADE_HISTORY, key, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "thread_id": state.get("thread_id"),
                    "context": state.get("context"),
                    "summary": content[:500],
                })
                saved += 1
            except Exception:
                pass

    # Save research cache for analyzed symbols
    fundamental_reports = state.get("fundamental_reports", {})
    if isinstance(fundamental_reports, dict):
        reports = fundamental_reports.get("reports", fundamental_reports)
        for sym, report in list(reports.items())[:5]:
            if isinstance(report, dict) and report.get("status") == "success":
                key = f"research_{sym}"
                try:
                    await store_put(store, NS_RESEARCH_CACHE, key, {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": sym,
                        "health_score": report.get("health_score"),
                        "recommendation": report.get("recommendation"),
                        "conviction": report.get("conviction"),
                    })
                except Exception:
                    pass

    return {}


# ---------------------------------------------------------------------------
# Supervisor node — ReAct loop via generate_with_tools()
# ---------------------------------------------------------------------------


async def supervisor_node(
    state: AgentState, config: RunnableConfig | None = None, runtime: Runtime[Context] | None = None
) -> dict[str, Any]:
    """
    Run one ReAct iteration with 5 sub-agent tools.

    Builds a context-aware system prompt, then calls the LLM with
    SUBAGENT_TOOLS and SUBAGENT_CALLABLES via generate_with_tools().
    The LLM may call multiple sub-agents within this single invocation; the
    function returns when the LLM produces a final text response.

    If ``pending_human_approval`` is set and no ``human_decision`` has
    been recorded yet, the node returns immediately so the graph can
    route to the human_review node.

    Uses Runtime[Context] pattern for store access (LangGraph best practice).
    """
    from src.autfin.llm.client import get_llm_client

    errors: list[str] = list(state.get("errors", []) or [])
    context = state.get("context", "morning_routine")
    depth = state.get("depth", "standard")

    # ── Human review gate (priority over tool loop) ──
    if state.get("pending_human_approval") and not state.get("human_decision"):
        logger.info("Supervisor: waiting for human approval on pending trade")
        return {
            "messages": [AIMessage(content="Supervisor: waiting for human approval.")],
            "errors": errors,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # ── Build system prompt ──
    system_prompt = _build_context_prompt(context, depth, state)

    # ── Preload learning context (patterns + calibration) on first iteration ──
    if not state.get("learning_context_loaded"):
        try:
            from src.autfin.learning.feedback_loop import get_learning_summary
            learning_summary = await get_learning_summary()
            state["learning_context"] = learning_summary
            state["learning_context_loaded"] = True
        except Exception:
            state["learning_context"] = {}
            state["learning_context_loaded"] = True

    # ── Load cross-session memories from Store on first iteration ──
    if not state.get("memories_loaded"):
        try:
            if runtime is not None:
                memory_result = await load_memories(state, runtime)
            else:
                # Fallback for legacy callers
                memory_result = await load_memories_legacy(state, config)
            state["context_memory"] = memory_result.get("context_memory", {})
            state["memories_loaded"] = True
        except Exception:
            state["context_memory"] = {}
            state["memories_loaded"] = True

    # ── Load sub-agent tools (v5.5 subagents pattern) ──
    try:
        from src.autfin.agents.subagents import SUBAGENT_TOOLS, SUBAGENT_CALLABLES
        all_tools = SUBAGENT_TOOLS
        all_callables = SUBAGENT_CALLABLES
        logger.info(
            "Supervisor: loaded %d sub-agent tools (%s)",
            len(all_tools), [t["function"]["name"] for t in all_tools],
        )
    except ImportError as exc:
        logger.error("Supervisor: failed to import sub-agent tools — %s", exc)
        errors.append(f"Sub-agent tool import error: {exc}")
        return {
            "messages": [
                AIMessage(
                    content="Supervisor: sub-agent module unavailable. "
                    "Check src/autfin/agents/subagents.py."
                )
            ],
            "errors": errors,
        }

    # ── LLM client ──
    llm = get_llm_client()
    if not llm.is_available:
        logger.error("Supervisor: LLM client unavailable — cannot run ReAct loop")
        errors.append("LLM client unavailable")
        return {
            "messages": [AIMessage(content="Supervisor: LLM unavailable. Cannot continue.")],
            "errors": errors,
        }

    # ── Token budget guard ──
    total_tokens = state.get("total_tokens", 0) or 0
    budget_limit = state.get("budget_limit", 0.0) or 0.0
    if budget_limit > 0 and total_tokens >= budget_limit:
        logger.warning("Supervisor: token budget exhausted (%d / %d)", total_tokens, budget_limit)
        return {
            "messages": [AIMessage(content="Supervisor: token budget exhausted. Ending session.")],
            "errors": errors + ["Token budget exhausted"],
            "iteration_count": (state.get("iteration_count", 0) or 0) + 1,
        }

    # ── Run ReAct loop with tool calling ──
    logger.info("Supervisor: starting ReAct loop (context=%s, depth=%s)", context, depth)
    
    result = None
    try:
        result = await llm.generate_with_tools(
            system_prompt=system_prompt,
            tools=all_tools,
            tool_callables=all_callables,
            model=llm.deep_model,
            max_tokens=4096,
            max_iterations=20,
            state=state,
        )
    except Exception as exc:
        logger.warning("Supervisor: LLM tool-calling failed — %s", exc)
        errors.append(f"LLM tool error: {exc}")
    
    # ── Rate-limit guard: if LLM fails repeatedly, end the session ──
    if result is None:
        llm_failures = len([m for m in state.get("messages", []) 
                           if hasattr(m, "content") and "LLM" in str(m.content)])
        if llm_failures >= 1:
            logger.warning("Supervisor: %d consecutive LLM failures — ending session", llm_failures)
            return {
                "messages": [AIMessage(content="Supervisor: LLM unavailable (rate-limited). Ending session.")],
                "errors": errors,
                "routing_history": ["__end__"],
            }
        return {
            "messages": [AIMessage(content="Supervisor: LLM call failed (rate-limited). Retrying...")],
            "errors": errors,
        }
    if result:
        logger.info("Supervisor: ReAct loop completed with result (len=%d)", len(result))
        return {
            "messages": [AIMessage(content=result)],
            "errors": errors,
            "iteration_count": (state.get("iteration_count", 0) or 0) + 1,
        }

    logger.info("Supervisor: ReAct loop produced no output — no action needed")
    return {
        "messages": [AIMessage(content="Supervisor: no action needed.")],
        "errors": errors,
        "iteration_count": (state.get("iteration_count", 0) or 0) + 1,
    }


# ---------------------------------------------------------------------------
# Human review node — interrupt-based HITL
# ---------------------------------------------------------------------------


async def human_review_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """
    Pause execution for human review of a pending trade or decision.

    Uses LangGraph's ``interrupt()`` to suspend the graph.  The caller
    must resume with a decision dict containing at least a ``"decision"``
    key (``"approved"`` or ``"rejected"``).

    After resume, clears ``pending_human_approval`` and records the
    decision in ``human_decision`` so the supervisor can proceed.
    """
    errors: list[str] = list(state.get("errors", []) or [])
    pending = state.get("pending_human_approval", {})

    if not pending:
        logger.debug("Human review: no pending approval — skipping")
        return {"errors": errors}

    logger.info("Human review: pausing for approval on trade %s", pending.get("symbol", "unknown"))

    # interrupt() must be the first side-effectful call — no try/except
    decision = interrupt({
        "type": "human_review",
        "trade": pending,
        "question": pending.get("question", "Approve this trade?"),
    })

    decision_value = (
        decision.get("decision", "rejected")
        if isinstance(decision, dict)
        else "approved" if decision else "rejected"
    )

    logger.info("Human review: decision = %s", decision_value)

    return {
        "human_decision": decision_value,
        "pending_human_approval": None,
        "messages": [
            AIMessage(content=f"Human decided: {decision_value}")
        ],
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_from_supervisor(state: AgentState) -> str:
    """
    Determine the next graph node after the supervisor finishes.

    Priority:
      1. If ``pending_human_approval`` is set with no ``human_decision``
         → route to ``human_review``.
      2. If the routing history ends with ``"__end__"`` (set by the
         ``finalize_session`` tool) → route to ``END``.
      3. Otherwise → loop back to ``supervisor`` for another iteration.
    """
    # Safety: guard against infinite loops
    # Count iterations from state metadata — supervisor_node never appends
    # to routing_history, so we track via a dedicated counter.
    iteration_count: int = state.get("iteration_count", 0) or 0
    if iteration_count >= MAX_GRAPH_ITERATIONS:
        logger.warning(
            "Supervisor: max graph iterations (%d) reached — forcing end",
            MAX_GRAPH_ITERATIONS,
        )
        return "__end__"

    # Human approval takes priority
    if state.get("pending_human_approval") and not state.get("human_decision"):
        logger.info("Supervisor route → human_review")
        return "human_review"

    # Tools signal end by appending "__end__" to routing_history
    routing = state.get("routing_history") or []
    if routing and routing[-1] == "__end__":
        logger.info("Supervisor route → END (finalize_session called)")
        # Save memories before ending session
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(save_memory(state, config))
        except Exception:
            pass
        return "__end__"

    logger.debug("Supervisor route → supervisor (continue ReAct loop)")
    return "supervisor"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_supervisor_graph() -> StateGraph:
    """
    Build the v5.5 subagents coordinator graph.

    Three nodes:
      - ``supervisor``    — runs the ReAct tool-calling loop with 5 sub-agents
      - ``human_review``  — pauses for HITL via interrupt()
      - ``END``           — terminal state

    Edges:
      START → supervisor
      supervisor → supervisor (continue loop)
      supervisor → human_review (when HITL needed)
      supervisor → END (when session is finalized)
      human_review → supervisor (return after input)

    Returns an uncompiled ``StateGraph`` with ``context_schema=Context``
    for Runtime[Context] access in nodes.
    """
    builder = StateGraph(AgentState, context_schema=Context)

    builder.add_node(
        "supervisor",
        safe_node(
            supervisor_node,
            node_name="supervisor",
            circuit_breaker_name="llm",
            degrade_on_failure=True,
        ),
    )
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "supervisor": "supervisor",
            "human_review": "human_review",
            "__end__": END,
        },
    )

    builder.add_edge("human_review", "supervisor")

    return builder


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def compile_supervisor_graph(checkpointer=None, store=None):
    """
    Compile the v5.5 subagents coordinator graph for execution.

    Args:
        checkpointer: Optional LangGraph checkpointer for crash recovery.
        store: Optional LangGraph Store for long-term memory.

    Returns:
        Compiled runnable graph with observability callback attached if
        the observability handler is available.
    """
    builder = build_supervisor_graph()

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store

    compiled = builder.compile(**compile_kwargs)

    try:
        from src.autfin.observability.handler import get_callback_handler

        handler = get_callback_handler()
        if handler is not None:
            compiled = compiled.with_config({"callbacks": [handler]})
    except Exception:
        logger.debug("Failed to attach observability handler", exc_info=True)

    return compiled
