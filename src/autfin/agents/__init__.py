"""
AutFin v7.0 — Pydantic AI Agent Implementations

New agents built with Pydantic AI capabilities (SubAgents, Planning, Memory, CodeMode, Guardrails, etc.)
"""
from __future__ import annotations

from .analysis import analysis_agent
from .base import create_agent, create_subagent, get_model
from .conference import conference_agent
from .debate import debate_agent
from .guardrails import (
    get_guardrails_for_agent,
    input_guardrails,
    output_guardrails,
    tool_guardrails,
)
from .managed_prompts import (
    analysis_instructions,
    conference_instructions,
    debate_instructions,
    get_managed_prompt,
    market_data_instructions,
    memory_instructions,
    quant_instructions,
    research_instructions,
    risk_instructions,
    supervisor_instructions,
)
from .market_data import market_data_agent
from .memory import memory_agent
from .observability import (
    add_run_metadata,
    create_agent_metadata,
    record_debate_metric,
    record_portfolio_metric,
    record_risk_metric,
    record_trade_metric,
    trace_agent_run,
    trace_operation,
)
from .quant import quant_agent
from .research import research_agent
from .risk import risk_agent
from .supervisor import (
    memory_capability,
    planning_capability,
    subagents_capability,
    supervisor_agent,
    system_reminders_capability,
    workflow_capability,
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
    "create_agent_metadata",
    "add_run_metadata",
    "record_trade_metric",
    "record_risk_metric",
    "record_debate_metric",
    "record_portfolio_metric",
    "trace_operation",
    "trace_agent_run",
    # Supervisor & capabilities
    "supervisor_agent",
    "subagents_capability",
    "workflow_capability",
    "planning_capability",
    "memory_capability",
    "system_reminders_capability",
]