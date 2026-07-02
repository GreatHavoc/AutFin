"""
Supervisor Graph — v5.3 Subagents Coordinator.

Architecture:
  START → supervisor_node → {supervisor_node, human_review_node, END}
  human_review_node → supervisor_node (return after input)

The supervisor is the "Subagents Coordinator" — it calls agent tools
(fundamentals, technical, sentiment, macro, research, risk, portfolio),
observes results, and decides next steps. Each agent tool makes its own
LLM call with clean context (context isolation).

  morning_routine  → full portfolio scan + screening + research + trade
  price_alert      → investigate one symbol (3-agent quick fan-out)
  quarterly_review → deep dive all holdings with DCF
  manual           → user-requested single stock analysis

Agent tools give context isolation, parallel execution, and full
LangSmith visibility per agent call.

Self-learning: learning context (calibration, patterns) is preloaded into
state on first iteration and injected into the system prompt.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from src.autfin.state.agent_state import AgentState
from src.autfin.graph.error_handler import safe_node, RetryPolicy

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
    base = """You are the AutFin v5.4 Coordinator — a PM-level AI running a ReAct loop.

You have 30 tools available. 9 are deterministic (fast, no LLM). 1 is LLM-driven (research_synthesis).
Think, plan, then call tools. Observe results. Re-plan.

DETERMINISTIC TOOLS (fast, no LLM — call these first):
Analysis:
- fundamentals_analysis(symbol, depth) — Piotroski, Magic Formula, ratios, health score. ~3s.
- technical_analysis(symbol, depth) — RSI, MACD, Bollinger, patterns, signal. ~2s.
- sentiment_analysis(symbol) — FinBERT + VADER on news + Reddit. ~5s.
- macro_analysis() — Global macro (Fed, oil, VIX, DXY, FII flows). ~3s.
- valuation_analysis(symbol, depth) — DCF scenarios + reverse DCF. ~2s.
- distortion_audit(symbol, depth) — Fake breakouts, pump patterns, distribution. ~2s.
- regulatory_analysis(symbol, depth) — NSE/SEBI/RBI announcements. ~3s.
- risk_assessment(trade_json) — 14 risk checks with veto power. ~0.5s.
- portfolio_allocation() — Current holdings, cash, P&L. ~1s.

LLM-DRIVEN (expensive — use only for final decisions):
- research_synthesis(symbols, depth) — Bull/Bear debate. Most expensive. Use for final decisions only.

Data & Utilities (direct tools):
Data: get_portfolio_state, get_market_quotes, get_market_status
Screening: screen_candidates
Research: research_stock(symbol, depth), research_batch(symbols, depth)
Analysis: build_stock_pitch, validate_thesis, analyze_macro_context
Risk & Execution: run_risk_checks, allocate_capital, execute_order
Memory: retrieve_learnings, log_decision, record_learning, calibrate_conviction, detect_patterns
HITL: request_human_approval
Guardian: check_guardian
Session: finalize_session

STRATEGY — BATCH deterministic tools in a SINGLE response for maximum speed:
All deterministic tools run in parallel via asyncio.gather — calling 5 together takes ~5s, not ~25s.

CORRECT (fast, ~5s):
  Response: tool_calls=[fundamentals_analysis("RELIANCE"), technical_analysis("RELIANCE"), sentiment_analysis("RELIANCE"), distortion_audit("RELIANCE"), regulatory_analysis("RELIANCE")]
  → All 5 run in parallel. One response. Fast.

WRONG (slow, ~25s):
  Response 1: tool_calls=[fundamentals_analysis("RELIANCE")]
  Response 2: tool_calls=[technical_analysis("RELIANCE")]
  → 2 sequential LLM round-trips. Slow.

Batching patterns:
- Morning routine: fundamentals_analysis + technical_analysis + sentiment_analysis + distortion_audit + regulatory_analysis (all 5 in ONE response)
  → then risk_assessment → execute_order
- Price alert: technical_analysis(quick) + sentiment_analysis + distortion_audit (3 in ONE response) → validate_thesis
- Quarterly review: fundamentals_analysis(deep) + valuation_analysis(deep) + regulatory_analysis(deep) on EVERY holding (all in ONE response)
- Manual: fundamentals_analysis + technical_analysis + distortion_audit on requested symbol (3 in ONE response)

RULES:
- ALWAYS batch deterministic tools in a single response. Never call them one at a time.
- Use research_synthesis ONLY for final investment decisions (Bull/Bear debate). It's LLM-driven and expensive.
- Use distortion_audit to check for price anomalies BEFORE recommending a trade.
- Use regulatory_analysis to check for SEBI/RBI events BEFORE recommending a trade.
- Depth tiers: quick=skip DCF/peers/moat. standard=all but DCF. deep=everything.
- Respect risk/reward ≥ 2:1. Conviction ≥ 0.5 to trade.
- Sector concentration max 30%. Cash reserve 10%.
- If human approval needed → call request_human_approval.
- End session → call finalize_session."""

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


async def load_memories(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Load relevant memories from the LangGraph Store before research.

    Searches for:
    - Agent instructions (procedural memory)
    - Past trade outcomes for the current symbol(s) (episodic memory)
    - User preferences (semantic memory)

    Injects found memories into state["context_memory"] for the supervisor
    to reference during its ReAct loop.
    """
    from src.autfin.memory.langgraph_store import (
        NS_AGENT_INSTRUCTIONS, NS_TRADE_HISTORY, NS_USER_PREFERENCES,
        store_search,
    )

    store = config.get("store") if config else None
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


async def save_memory(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Save research outcomes and lessons learned to the LangGraph Store.

    Called after the supervisor completes a session. Persists:
    - Trade decisions and outcomes (episodic memory)
    - Key research findings (semantic memory)
    """
    from datetime import datetime, timezone
    from src.autfin.memory.langgraph_store import (
        NS_TRADE_HISTORY, NS_RESEARCH_CACHE,
        store_put,
    )

    store = config.get("store") if config else None
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
    state: AgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """
    Run one ReAct iteration with the full supervisor tool set.

    Builds a context-aware system prompt, then calls the LLM with
    SUPERVISOR_TOOLS and SUPERVISOR_CALLABLES via generate_with_tools().
    The LLM may call multiple tools within this single invocation; the
    function returns when the LLM produces a final text response.

    If ``pending_human_approval`` is set and no ``human_decision`` has
    been recorded yet, the node returns immediately so the graph can
    route to the human_review node.
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
            memory_result = await load_memories(state, config)
            state["context_memory"] = memory_result.get("context_memory", {})
            state["memories_loaded"] = True
        except Exception:
            state["context_memory"] = {}
            state["memories_loaded"] = True

    # ── Load tools (old supervisor tools + new agent tools) ──
    try:
        from src.autfin.tools.supervisor_tools import SUPERVISOR_TOOLS, SUPERVISOR_CALLABLES
    except ImportError as exc:
        logger.error("Supervisor: failed to import supervisor tools — %s", exc)
        errors.append(f"Tool import error: {exc}")
        return {
            "messages": [
                AIMessage(
                    content="Supervisor: tool module unavailable. "
                    "Check src/autfin/tools/supervisor_tools.py."
                )
            ],
            "errors": errors,
        }

    # Merge agent tools (v5.3 Subagents)
    try:
        from src.autfin.agents.agent_tools import TOOL_DEFINITIONS, TOOL_CALLABLES
        all_tools = SUPERVISOR_TOOLS + TOOL_DEFINITIONS
        all_callables = {**SUPERVISOR_CALLABLES, **TOOL_CALLABLES}
        logger.info(
            "Supervisor: loaded %d tools (%d old + %d new)",
            len(all_tools), len(SUPERVISOR_TOOLS), len(TOOL_DEFINITIONS),
        )
    except ImportError:
        logger.info("Supervisor: agent tools not available — using old tools only")
        all_tools = SUPERVISOR_TOOLS
        all_callables = SUPERVISOR_CALLABLES

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
    Build the v3.0 single-supervisor ReAct loop graph.

    Three nodes:
      - ``supervisor``    — runs the ReAct tool-calling loop
      - ``human_review``  — pauses for HITL via interrupt()
      - ``END``           — terminal state

    Edges:
      START → supervisor
      supervisor → supervisor (continue loop)
      supervisor → human_review (when HITL needed)
      supervisor → END (when session is finalized)
      human_review → supervisor (return after input)

    Returns an uncompiled ``StateGraph``.
    """
    builder = StateGraph(AgentState)

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
    Compile the v5.3 supervisor graph for execution.

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
