"""
Supervisor Graph — v3.0 single-supervisor ReAct loop.

Architecture:
  START → supervisor_node → {supervisor_node, human_review_node, END}
  human_review_node → supervisor_node (return after input)

The supervisor runs a ReAct loop: it calls tools via generate_with_tools(),
observes results, decides next step. Context drives behavior:

  morning_routine  → full portfolio scan + screening + research + trade
  price_alert      → investigate one symbol (3-agent quick fan-out)
  quarterly_review → deep dive all holdings with DCF
  manual           → user-requested single stock analysis

All agent capabilities are accessed as tools, not as subgraph nodes.
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
    base = """You are the AutFin v3.0 Supervisor — a PM-level AI running a ReAct loop.

You have 20 tools available. Think, plan, then call tools. Observe results. Re-plan.

AVAILABLE TOOLS:
Data: get_portfolio_state, get_market_quotes, get_market_status
Screening: screen_candidates
Research: research_stock(symbol, depth), research_batch(symbols, depth)
Analysis: build_stock_pitch, validate_thesis, analyze_macro_context
Risk & Execution: run_risk_checks, allocate_capital, execute_order
Memory: retrieve_learnings, log_decision, record_learning, calibrate_conviction, detect_patterns
HITL: request_human_approval
Guardian: check_guardian
Session: finalize_session

RULES:
- For morning_routine: first get_portfolio_state, then check_guardian, then screen_candidates,
  research_batch on top candidates, rank, build pitches, risk_check, allocate, execute.
- For price_alert: get_market_quotes for the alerted symbol, research_stock(quick), validate_thesis.
  Decide: noise → ignore, weak → full research, broken → sell.
- For quarterly_review: get_portfolio_state, then research_stock(deep) on EVERY holding.
- For manual: research_stock on the requested symbol at requested depth.
- Before ANY trade: retrieve_learnings for context from past trades.
- After ANY trade: log_decision to record the outcome.
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

    # ── Load tools ──
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
            tools=SUPERVISOR_TOOLS,
            tool_callables=SUPERVISOR_CALLABLES,
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


def compile_supervisor_graph(checkpointer=None):
    """
    Compile the v3.0 supervisor graph for execution.

    Args:
        checkpointer: Optional LangGraph checkpointer for crash recovery.

    Returns:
        Compiled runnable graph with observability callback attached if
        the observability handler is available.
    """
    builder = build_supervisor_graph()
    compiled = builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()

    try:
        from src.autfin.observability.handler import get_callback_handler

        handler = get_callback_handler()
        if handler is not None:
            compiled = compiled.with_config({"callbacks": [handler]})
    except Exception:
        logger.debug("Failed to attach observability handler", exc_info=True)

    return compiled
