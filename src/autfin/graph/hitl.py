"""
Human-in-the-Loop (HITL) graph nodes for AutFin.

Provides trade gate and human review nodes that pause the graph
before large trades using LangGraph's interrupt() / Command(resume=) mechanism.

Flow:
  trade_gate → [if large trade] human_review → interrupt()
  trade_gate → [if small trade] pass through to executor

Rules:
  - interrupt() must be the FIRST call in a node (re-runs on resume)
  - Do NOT wrap interrupt() in try/except
  - Side effects before interrupt() must be idempotent
  - Only JSON-serializable values in interrupt payloads
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from src.autfin.state.agent_state import AgentState
from src.autfin.config.settings import AutFinSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs_human_approval(trade_queue: list[dict], auto_execute_max: float) -> list[dict]:
    """
    Return trades that exceed the auto-execute threshold and need human approval.

    A trade needs HITL if:
      - execution_mode is "auto-small" AND trade_value > auto_execute_max_amount
      - OR execution_mode is "recommendation" (ALL trades need approval)
    """
    large_trades: list[dict] = []
    for trade in trade_queue:
        qty = int(float(trade.get("qty", 0)))
        price = float(trade.get("price", 0))
        trade_value = qty * price
        if trade_value > auto_execute_max:
            large_trades.append({
                **trade,
                "trade_value": round(trade_value, 2),
            })
    return large_trades


# ---------------------------------------------------------------------------
# Trade Gate Node — decides whether to pause for HITL
# ---------------------------------------------------------------------------

async def trade_gate_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """
    Gate before executor — pauses for human approval on large trades.

    Checks the trade_queue against auto_execute_max_amount threshold.
    If ANY trade exceeds the threshold, calls interrupt() to pause
    and wait for human approval.

    Rules followed:
      - interrupt() is the FIRST meaningful call (no side effects before it)
      - Payload is JSON-serializable only
      - No try/except around interrupt()
    """
    settings = AutFinSettings()
    execution_mode = settings.execution_mode
    auto_execute_max = float(settings.auto_execute_max_amount or 5000.0)
    trade_queue = list(state.get("trade_queue") or [])

    messages: list[Any] = []
    errors: list[str] = list(state.get("errors", []) or [])

    # Paper mode — no HITL needed, all trades are simulated
    if execution_mode == "paper":
        messages.append(AIMessage(
            content="Trade Gate: paper mode — all trades are simulated, no approval needed."
        ))
        return {
            "pending_human_approval": None,
            "human_decision": None,
            "messages": messages,
            "errors": errors,
        }

    # No trades — nothing to gate (check BEFORE mode logic to avoid interrupt with empty queue)
    if not trade_queue:
        messages.append(AIMessage(
            content="Trade Gate: trade queue empty, no approval needed."
        ))
        return {
            "pending_human_approval": None,
            "human_decision": None,
            "messages": messages,
            "errors": errors,
        }

    # Recommendation mode — ALL trades need human approval regardless of size
    if execution_mode == "recommendation":
        # All trades need approval
        large_trades = trade_queue
    else:
        # Auto-small mode: Find trades that need human approval (above threshold)
        large_trades = _needs_human_approval(trade_queue, auto_execute_max)

    if not large_trades:
        # All trades are within auto-execute limit — no HITL needed
        messages.append(AIMessage(
            content=(
                f"Trade Gate: {len(trade_queue)} trade(s) within auto-execute limit "
                f"(₹{auto_execute_max:,.2f}), proceeding without approval."
            )
        ))
        return {
            "pending_human_approval": None,
            "human_decision": None,
            "messages": messages,
            "errors": errors,
        }

    # Build approval payload for interrupt()
    approval_payload = {
        "action": "approve_trades",
        "trades": large_trades,
        "total_value": sum(t.get("trade_value", 0) for t in large_trades),
        "message": (
            f"{len(large_trades)} trade(s) exceed auto-execute limit "
            f"(₹{auto_execute_max:,.2f}). Total value: "
            f"₹{sum(t.get('trade_value', 0) for t in large_trades):,.2f}. "
            f"Approve execution?"
        ),
        "trade_details": [
            {
                "symbol": t.get("symbol", "UNKNOWN"),
                "action": t.get("action", "UNKNOWN"),
                "qty": t.get("qty", 0),
                "price": t.get("price", 0),
                "trade_value": t.get("trade_value", 0),
                "reason": t.get("reason", ""),
            }
            for t in large_trades
        ],
    }

    logger.info(
        "Trade Gate: %d trade(s) need HITL approval (total ₹%s)",
        len(large_trades),
        f"{approval_payload['total_value']:,.2f}",
    )

    # interrupt() is the FIRST call — no side effects before it
    # This pauses the graph and returns the payload to the caller
    result = interrupt(approval_payload)

    # After resume: result contains the human decision
    # The human can approve, reject, or modify the trades
    if isinstance(result, dict):
        approved = result.get("approved", False)
        modified_trades = result.get("trades")
    else:
        approved = bool(result)
        modified_trades = None

    if approved:
        # Human approved — update trade_queue with potentially modified trades
        final_trades = modified_trades if modified_trades is not None else trade_queue
        messages.append(AIMessage(
            content=(
                f"Trade Gate: Human approved {len(final_trades)} trade(s). "
                f"Proceeding to execution."
            )
        ))
        return {
            "pending_human_approval": None,
            "human_decision": "approved",
            "trade_queue": final_trades,
            "messages": messages,
            "errors": errors,
        }
    else:
        # Human rejected — clear the trade queue
        messages.append(AIMessage(
            content=(
                f"Trade Gate: Human rejected {len(large_trades)} trade(s). "
                f"Trade queue cleared."
            )
        ))
        return {
            "pending_human_approval": None,
            "human_decision": "rejected",
            "trade_queue": [],
            "messages": messages,
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Human Review Node — for explicit human_review routing
# ---------------------------------------------------------------------------

async def human_review_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """
    Pause for explicit human review of the current state.

    This is used by the Supervisor when it determines a human needs
    to review the current analysis before proceeding. Unlike trade_gate,
    this is a general-purpose review node.
    """
    trade_queue = list(state.get("trade_queue") or [])
    risk_assessments = state.get("risk_assessments") or {}
    messages: list[Any] = []
    errors: list[str] = list(state.get("errors", []) or [])

    # Build a summary of what needs review
    review_summary = {
        "action": "human_review",
        "trades_pending": len(trade_queue),
        "risk_vetoes": sum(
            1 for ra in risk_assessments.values()
            if isinstance(ra, dict) and not ra.get("approved", True)
        ),
        "message": "Human review requested. Please review the current analysis.",
    }

    if trade_queue:
        review_summary["trade_details"] = [
            {
                "symbol": t.get("symbol", "UNKNOWN"),
                "action": t.get("action", "UNKNOWN"),
                "qty": t.get("qty", 0),
                "price": t.get("price", 0),
            }
            for t in trade_queue
        ]

    # interrupt() pauses the graph
    result = interrupt(review_summary)

    # After resume
    if isinstance(result, dict):
        approved = result.get("approved", False)
        feedback = result.get("feedback", "")
    else:
        approved = bool(result)
        feedback = ""

    if approved:
        messages.append(AIMessage(
            content=f"Human Review: Approved.{f' Feedback: {feedback}' if feedback else ''}"
        ))
        return {
            "pending_human_approval": None,
            "human_decision": "approved",
            "trade_queue": trade_queue,
            "messages": messages,
            "errors": errors,
        }
    else:
        messages.append(AIMessage(
            content=f"Human Review: Rejected.{f' Feedback: {feedback}' if feedback else ''}"
        ))
        return {
            "pending_human_approval": None,
            "human_decision": "rejected",
            "trade_queue": [],
            "messages": messages,
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_trade_gate(state: AgentState) -> str:
    """Route after trade_gate — continue to executor or skip if rejected."""
    human_decision = state.get("human_decision")
    trade_queue = state.get("trade_queue") or []

    if human_decision == "rejected" or not trade_queue:
        return "skip_to_report"
    return "to_executor"
