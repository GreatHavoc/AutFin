"""AutFin v7.0 — Observability Helpers

Custom metrics, metadata, and tracing utilities.
Logfire configuration lives in __main__.py (run `python -m autfin`).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import logfire

# ─── Agent Metadata Helpers ───

def create_agent_metadata(
    agent_name: str,
    user_id: str,
    session_id: str,
    context: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create structured metadata for agent runs."""
    return {
        "agent": agent_name,
        "user_id": user_id,
        "session_id": session_id,
        "context": context,
        "environment": os.getenv("ENVIRONMENT", "development"),
        **kwargs,
    }


def add_run_metadata(
    agent: Any,
    user_id: str,
    session_id: str,
    context: str,
    **kwargs: Any,
) -> Any:
    """Add metadata to agent for this run."""
    metadata = create_agent_metadata(
        agent_name=agent.name, user_id=user_id, session_id=session_id, context=context, **kwargs
    )
    return agent.override(metadata=metadata)


# ─── Custom Logfire Metrics ───

def record_trade_metric(
    symbol: str,
    action: str,
    pnl_pct: float,
    risk_reward: float,
    conviction: float,
) -> None:
    """Record trade execution metric."""
    logfire.info(
        "Trade executed",
        symbol=symbol,
        action=action,
        pnl_pct=pnl_pct,
        risk_reward=risk_reward,
        conviction=conviction,
        _tags=["trade", "execution"],
    )


def record_risk_metric(
    symbol: str,
    verdict: str,
    risk_reward: float,
    conviction: float,
    portfolio_heat: float,
) -> None:
    """Record risk assessment metric."""
    logfire.info(
        "Risk assessment",
        symbol=symbol,
        verdict=verdict,
        risk_reward=risk_reward,
        conviction=conviction,
        portfolio_heat=portfolio_heat,
        _tags=["risk", "assessment"],
    )


def record_debate_metric(
    symbol: str,
    decision: str,
    confidence: float,
    bull_strength: float,
    bear_strength: float,
) -> None:
    """Record debate outcome metric."""
    logfire.info(
        "Debate completed",
        symbol=symbol,
        decision=decision,
        confidence=confidence,
        bull_strength=bull_strength,
        bear_strength=bear_strength,
        _tags=["debate", "decision"],
    )


def record_portfolio_metric(
    total_value: float,
    total_pnl: float,
    cash_pct: float,
    sector_exposure: dict[str, float],
) -> None:
    """Record portfolio snapshot metric."""
    logfire.info(
        "Portfolio snapshot",
        total_value=total_value,
        total_pnl=total_pnl,
        cash_pct=cash_pct,
        sector_exposure=sector_exposure,
        _tags=["portfolio", "snapshot"],
    )


# ─── Context Managers for Spans ───

@contextmanager
def trace_operation(name: str, **attributes: Any):
    """Context manager for custom trace spans."""
    with logfire.span(name, **attributes) as span:
        yield span


@contextmanager
def trace_agent_run(
    agent_name: str,
    user_id: str,
    session_id: str,
    context: str,
):
    """Context manager for full agent run trace."""
    with logfire.span(
        f"agent_run.{agent_name}",
        agent=agent_name,
        user_id=user_id,
        session_id=session_id,
        context=context,
    ) as span:
        yield span
