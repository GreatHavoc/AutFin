"""
AutFin v7.0 — Pydantic AI Agent Implementations

New agents built with Pydantic AI capabilities (SubAgents, DynamicWorkflow, Planning, Memory, CodeMode, Guardrails, etc.)
"""
from __future__ import annotations

from .base import create_agent, create_subagent, get_model
from .market_data import market_data_agent
from .analysis import analysis_agent
from .research import research_agent
from .risk import risk_agent
from .debate import debate_agent
from .conference import conference_agent
from .memory import memory_agent
from .quant import quant_agent
from .guardrails import (
    input_guardrails,
    output_guardrails,
    tool_guardrails,
    get_guardrails_for_agent,
)
from .managed_prompts import (
    supervisor_instructions,
    market_data_instructions,
    analysis_instructions,
    risk_instructions,
    debate_instructions,
    research_instructions,
    memory_instructions,
    quant_instructions,
    conference_instructions,
    get_managed_prompt,
)
from .supervisor import (
    supervisor_agent,
    subagents_capability,
    morning_workflow,
    quarterly_workflow,
    planning_capability,
    memory_capability,
    system_reminders_capability,
)
from .observability import (
    configure_logfire,
    create_instrumentation,
    create_production_instrumentation,
    create_debug_instrumentation,
    create_agent_metadata,
    add_run_metadata,
    record_trade_metric,
    record_risk_metric,
    record_debate_metric,
    record_portfolio_metric,
    trace_operation,
    trace_agent_run,
    check_logfire_health,
)

__all__ = [
    # Base
    "create_agent",
    "create_subagent",
    "get_model",
    # Sub-agents
    "market_data_agent",
    "analysis_agent",
    "research_agent",
    "risk_agent",
    "debate_agent",
    "conference_agent",
    "memory_agent",
    "quant_agent",
    # Guardrails
    "input_guardrails",
    "output_guardrails",
    "tool_guardrails",
    "get_guardrails_for_agent",
    # Managed Prompts
    "supervisor_instructions",
    "market_data_instructions",
    "analysis_instructions",
    "risk_instructions",
    "debate_instructions",
    "research_instructions",
    "memory_instructions",
    "quant_instructions",
    "conference_instructions",
    "get_managed_prompt",
    # Observability
    "configure_logfire",
    "create_instrumentation",
    "create_production_instrumentation",
    "create_debug_instrumentation",
    "create_agent_metadata",
    "add_run_metadata",
    "record_trade_metric",
    "record_risk_metric",
    "record_debate_metric",
    "record_portfolio_metric",
    "trace_operation",
    "trace_agent_run",
    "check_logfire_health",
    # Supervisor & capabilities
    "supervisor_agent",
    "subagents_capability",
    "morning_workflow",
    "quarterly_workflow",
    "planning_capability",
    "memory_capability",
    "system_reminders_capability",
]