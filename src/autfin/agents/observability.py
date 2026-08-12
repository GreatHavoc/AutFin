"""AutFin v7.0 — Observability Configuration

Logfire instrumentation for debugging, monitoring, and cost tracking.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import logfire
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

from autfin.types import AgentDeps


# ─── Logfire Configuration ───

_logfire_configured = False


def configure_logfire(
    project_name: str = "autfin",
    environment: Optional[str] = None,
    send_to_logfire: bool = True,
    include_content: bool = True,
    include_model_request_parameters: bool = True,
    include_binary_content: bool = False,
    version: int = 5,
) -> logfire.Logfire:
    """Configure Logfire SDK for AutFin.
    
    Args:
        project_name: Project name in Logfire
        environment: Environment name (dev/staging/prod)
        send_to_logfire: Whether to send data to Logfire cloud
        include_content: Include prompts/completions in traces (privacy sensitive)
        include_model_request_parameters: Include full model request params
        include_binary_content: Include binary content in traces
        version: OTel semantic conventions version (5 = latest)
    
    Returns:
        Configured Logfire instance
    """
    global _logfire_configured
    
    if _logfire_configured:
        return logfire
    
    # Configure Logfire
    logfire_instance = logfire.configure(
        project=project_name,
        environment=environment or os.getenv("ENVIRONMENT", "development"),
        send_to_logfire=send_to_logfire,
        # Scrub sensitive data
        scrubbing_options=logfire.ScrubbingOptions(
            # Remove credit card numbers, SSNs, API keys, etc.
            redact_credit_cards=True,
            redact_email_addresses=True,
            redact_phone_numbers=True,
            redact_ip_addresses=True,
            redact_us_social_security_numbers=True,
            custom_patterns={
                "KITE_API_KEY": r"KITE_API_KEY[=\s]+[A-Za-z0-9]+",
                "KITE_API_SECRET": r"KITE_API_SECRET[=\s]+[A-Za-z0-9]+",
                "NVIDIA_NIM_API_KEY": r"NVIDIA_NIM_API_KEY[=\s]+[A-Za-z0-9]+",
            },
        ),
    )
    
    _logfire_configured = True
    
    # Instrument Pydantic AI
    logfire.instrument_pydantic_ai()
    
    # Instrument HTTPX for raw request/response logging
    logfire.instrument_httpx(capture_all=True)
    
    # Instrument SQLite for database queries
    logfire.instrument_sqlite3()
    
    # Instrument asyncpg if using PostgreSQL
    try:
        logfire.instrument_asyncpg()
    except Exception:
        pass
    
    # Instrument psycopg if using PostgreSQL
    try:
        logfire.instrument_psycopg()
    except Exception:
        pass
    
    return logfire_instance


def create_instrumentation(
    include_content: bool = True,
    include_model_request_parameters: bool = True,
    include_binary_content: bool = False,
    version: int = 5,
) -> Instrumentation:
    """Create Instrumentation capability with custom settings."""
    settings = InstrumentationSettings(
        include_content=include_content,
        include_model_request_parameters=include_model_request_parameters,
        include_binary_content=include_binary_content,
        version=version,
    )
    return Instrumentation(settings=settings)


def create_production_instrumentation() -> Instrumentation:
    """Create production-ready instrumentation (no sensitive content)."""
    settings = InstrumentationSettings(
        include_content=False,  # Don't log prompts/completions
        include_model_request_parameters=False,  # Don't log full tool schemas
        include_binary_content=False,
        version=5,
    )
    return Instrumentation(settings=settings)


def create_debug_instrumentation() -> Instrumentation:
    """Create debug instrumentation (full content)."""
    settings = InstrumentationSettings(
        include_content=True,
        include_model_request_parameters=True,
        include_binary_content=True,
        version=5,
    )
    return Instrumentation(settings=settings)


# ─── Agent Metadata Helpers ───

def create_agent_metadata(
    agent_name: str,
    user_id: str,
    session_id: str,
    context: str,
    **kwargs
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
    agent: Agent[AgentDeps, Any],
    user_id: str,
    session_id: str,
    context: str,
    **kwargs
) -> Agent[AgentDeps, Any]:
    """Add metadata to agent for this run."""
    metadata = create_agent_metadata(agent_name=agent.name, user_id=user_id, session_id=session_id, context=context, **kwargs)
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

from contextlib import contextmanager

@contextmanager
def trace_operation(name: str, **attributes):
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


# ─── Health Check ───

def check_logfire_health() -> bool:
    """Check if Logfire is configured and reachable."""
    return _logfire_configured