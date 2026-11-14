"""AutFin v7.0 — Supervisor Agent (Orchestrator)

Coordinates 7 sub-agents via SubAgents + DynamicWorkflow capabilities.
"""
from __future__ import annotations

from typing import Literal

from pydantic_ai import ApprovalRequired, RunContext
from pydantic_ai.capabilities import Thinking, WebSearch
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow, WorkflowAgent
from pydantic_ai_harness.memory import Memory
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.system_reminders import Reminder, SystemReminders

from autfin.agents.analysis import analysis_agent
from autfin.agents.base import create_agent, get_model
from autfin.agents.compaction import get_history_compaction
from autfin.agents.conference import conference_agent
from autfin.agents.debate import debate_agent
from autfin.agents.managed_prompts import supervisor_instructions

# Import sub-agents
from autfin.agents.market_data import market_data_agent
from autfin.agents.memory import memory_agent
from autfin.agents.persistence import get_step_persistence
from autfin.agents.research import research_agent
from autfin.agents.risk import risk_agent
from autfin.agents.safety import get_artifact_scanner
from autfin.memory.store import get_memory_store
from autfin.tools import learning_toolset
from autfin.types import (
    AgentDeps,
    SupervisorOutput,
    TradeOrder,
)

# ─── SubAgents Capability ──────────────────────────────────────────────
subagents_capability = SubAgents(
    id="sub_agents",
    agents=[
        SubAgent(
            market_data_agent,
            name="market_data",
            description="Portfolio, quotes, market status, risk alerts",
            usage_limits=UsageLimits(request_limit=10, total_tokens_limit=10000),
            timeout_seconds=30,
            max_calls=3,
        ),
        SubAgent(
            analysis_agent,
            name="analyze",
            description="Technical + fundamental + sentiment + valuation analysis",
            usage_limits=UsageLimits(request_limit=20, total_tokens_limit=30000),
            timeout_seconds=120,
            max_calls=10,
        ),
        SubAgent(
            research_agent,
            name="research",
            description="Deep research, screening, thesis generation",
            usage_limits=UsageLimits(request_limit=15, total_tokens_limit=25000),
            timeout_seconds=180,
            max_calls=5,
        ),
        SubAgent(
            risk_agent,
            name="risk",
            description="Position sizing, stop-loss, correlation, portfolio heat",
            usage_limits=UsageLimits(request_limit=10, total_tokens_limit=15000),
            timeout_seconds=60,
            max_calls=10,
        ),
        SubAgent(
            debate_agent,
            name="debate",
            description="Bull/Bear adversarial debate on trade candidates",
            usage_limits=UsageLimits(request_limit=8, total_tokens_limit=15000),
            timeout_seconds=120,
            max_calls=5,
        ),
        SubAgent(
            conference_agent,
            name="conference",
            description="Multi-agent coordination conferences",
            usage_limits=UsageLimits(request_limit=10, total_tokens_limit=20000),
            timeout_seconds=180,
            max_calls=3,
        ),
        SubAgent(
            memory_agent,
            name="memory",
            description="Cross-session memory: events, trades, patterns, calibration",
            usage_limits=UsageLimits(request_limit=10, total_tokens_limit=10000),
            timeout_seconds=30,
            max_calls=5,
        ),
    ],
    models={
        "fast": get_model("fast"),
        "standard": get_model("deep"),
        "deep": get_model("reasoning"),
    },
    forward_usage=True,
    inherit_tools=False,
    shared_capabilities=[],
    tool_retries=2,
    contain_errors=True,
)


# ─── DynamicWorkflow for Complex Orchestration ─────────────────────────
# The MODEL writes the orchestration script via run_workflow tool.
# Sub-agents become async functions: await market_data_agent(task="...")
workflow_capability = DynamicWorkflow(
    agents=[
        WorkflowAgent(market_data_agent, name="market_data"),
        WorkflowAgent(analysis_agent, name="analysis"),
        WorkflowAgent(research_agent, name="research"),
        WorkflowAgent(debate_agent, name="debate"),
        WorkflowAgent(risk_agent, name="risk"),
        WorkflowAgent(conference_agent, name="conference"),
        WorkflowAgent(memory_agent, name="memory"),
    ],
    max_agent_calls=50,
    sub_agent_usage_limits=UsageLimits(request_limit=20, total_tokens_limit=50000),
    forward_usage=True,
    resource_limits={"max_duration_seconds": 300},
    inherit_model=False,
    id="supervisor_workflow",
    defer_loading=True,
    description="Multi-agent orchestration: market data, analysis, research, debate, risk, conference",
    tool_name="run_workflow",
)


# ─── Planning Capability ───────────────────────────────────────────────
# Planning no longer places a cache breakpoint of its own: configure caching
# on the model (e.g. ModelSettings(anthropic_cache_messages=True)) instead.
planning_capability = Planning(
    store=None,  # In-memory by default, can use SqlitePlanStore
    enable_subtasks=True,
    inject=True,
)


# ─── Memory Capability ────────────────────────────────────────────────
# SqliteMemoryStore, not FileStore: a file store resolves its directory from the
# run's workspace, and no workspace backend exists on this platform, so FileStore
# raised UserError in before_run on every live run. See memory/store.py.
memory_capability = Memory(
    get_memory_store(),
    # Memory is injected into every run's context, so what the agents write there
    # is a second chance to plant an instruction.
    scanner=get_artifact_scanner(),
    namespace=lambda ctx: ctx.deps.user_id,
    max_tokens=5000,
    max_lines=500,
    inject_memory=True,
)


# ─── System Reminders ──────────────────────────────────────────────────
system_reminders_capability = SystemReminders(
    id="system_reminders",
    reminders=[
        Reminder("Stay focused on risk management. Check risk/reward >= 2:1 before any trade.", tag=None),
        Reminder("Verify conviction >= 0.5 before any trade execution.", tag=None),
        Reminder("ALWAYS run debate before any trade decision. No exceptions.", tag=None),
        Reminder("Batch sub-agent calls in a single response for speed.", tag=None),
        Reminder("For multi-step work load the run_workflow capability and write one script "
                  "that fans out to the sub-agents; do not run them one at a time.", tag=None),
    ],
    dynamic_reminders=[],
)


# ─── Supervisor Agent ──────────────────────────────────────────────────
supervisor_agent = create_agent(
    name="supervisor",
    output_type=SupervisorOutput,
    model_type="deep",
    instructions=supervisor_instructions.default,
    capabilities=[
        subagents_capability,
        workflow_capability,
        planning_capability,
        memory_capability,
        system_reminders_capability,
        # The daemon continues one conversation across trading days, so the
        # supervisor's history grows by a full routine per day: cheap tiers
        # first, the summarizer only when clearing tool results is not enough.
        get_history_compaction(),
        # Crash recovery: snapshots each settled step of the daemon's run to
        # data/autfin.db, so a crashed routine resumes instead of restarting.
        get_step_persistence(),
        Thinking(effort="high"),
        WebSearch(local="duckduckgo"),
    ],
    toolsets=[learning_toolset],
    model_settings=ModelSettings(temperature=0.0, max_tokens=8192),
)


# ─── Supervisor Tools ──────────────────────────────────────────────────

@supervisor_agent.tool
async def analyze_symbol(
    ctx: RunContext[AgentDeps],
    symbol: str,
    depth: "Literal['quick', 'standard', 'deep']" = "standard",
) -> SupervisorOutput:
    """Run full analysis on a single symbol."""
    # Analysis is delegated through the SubAgents capability: this tool exists
    # only to shape the supervisor's decision, the dimensions are computed by
    # the analyze sub-agent.
    return SupervisorOutput(
        action="continue",
        message=f"Analysis requested for {symbol} at {depth} depth",
        trade_queue=[],
        next_context="analyze",
        metadata={"symbol": symbol, "depth": depth},
    )


@supervisor_agent.tool
async def execute_trade(
    ctx: RunContext[AgentDeps],
    order: TradeOrder,
) -> SupervisorOutput:
    """Execute a trade order (paper or live)."""
    kite = ctx.deps.kite_client
    if not kite:
        return SupervisorOutput(
            action="error",
            message="No Kite client available",
            trade_queue=[],
        )

    # B1: a live order pauses for a human before it reaches Kite. Paper mode never
    # contacts the broker, so approving there adds nothing.
    if ctx.deps.mode != "paper" and not ctx.tool_call_approved:
        raise ApprovalRequired(
            metadata={"symbol": order.symbol, "action": order.action, "quantity": order.quantity}
        )

    if ctx.deps.mode == "paper":
        # Paper trading - simulate
        return SupervisorOutput(
            action="complete",
            message=f"PAPER TRADE: {order.action} {order.quantity} {order.symbol} @ {order.price}",
            trade_queue=[],
            metadata={"paper": True, "order": order.model_dump()},
        )

    # Live trading
    try:
        kite_order = kite.place_order(
            tradingsymbol=order.symbol,
            exchange="NSE",
            transaction_type=order.action.upper(),
            quantity=order.quantity,
            order_type=order.order_type.upper(),
            price=float(order.price) if order.price else 0,
            trigger_price=float(order.trigger_price) if order.trigger_price else 0,
            product=order.product,
            tag=order.tag or "autfin",
        )
        return SupervisorOutput(
            action="complete",
            message=f"LIVE TRADE: {order.action} {order.quantity} {order.symbol} - Order ID: {kite_order}",
            trade_queue=[],
            metadata={"paper": False, "order_id": kite_order},
        )
    except Exception as e:
        return SupervisorOutput(
            action="error",
            message=f"Trade execution failed: {e}",
            trade_queue=[],
        )


# ─── Export ────────────────────────────────────────────────────────────
__all__ = [
    "supervisor_agent",
    "market_data_agent",
    "analysis_agent",
    "research_agent",
    "risk_agent",
    "debate_agent",
    "conference_agent",
    "memory_agent",
    "subagents_capability",
    "workflow_capability",
    "planning_capability",
    "memory_capability",
    "system_reminders_capability",
]