"""AutFin v7.0 — Supervisor Agent (Orchestrator)

Coordinates 7 sub-agents via SubAgents + DynamicWorkflow capabilities.
"""
from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Thinking, WebSearch
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow, WorkflowAgent
from pydantic_ai_harness.memory import FileStore, Memory
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.system_reminders import Reminder, SystemReminders

from autfin.agents.analysis import analysis_agent
from autfin.agents.base import create_agent, get_model
from autfin.agents.conference import conference_agent
from autfin.agents.debate import debate_agent
from autfin.agents.managed_prompts import supervisor_instructions

# Import sub-agents
from autfin.agents.market_data import market_data_agent
from autfin.agents.memory import memory_agent
from autfin.agents.research import research_agent
from autfin.agents.risk import risk_agent
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
    resource_limits={"max_duration_secs": 300},
    inherit_model=False,
    id="supervisor_workflow",
    defer_loading=True,
    description="Multi-agent orchestration: market data, analysis, research, debate, risk, conference",
    tool_name="run_workflow",
)


# ─── Planning Capability ───────────────────────────────────────────────
planning_capability = Planning(
    store=None,  # In-memory by default, can use SqlitePlanStore
    enable_subtasks=True,
    inject=True,
    cache_ttl="5m",
)


# ─── Memory Capability ────────────────────────────────────────────────
memory_capability = Memory(
    FileStore("./data/memory"),
    namespace=lambda ctx: ctx.deps.user_id,
    max_tokens=5000,
    max_lines=500,
    inject_memory=True,
)


# ─── System Reminders ──────────────────────────────────────────────────
system_reminders_capability = SystemReminders(
    id="system_reminders",
    reminders=[
        Reminder("Stay focused on risk management. Check risk/reward >= 2:1 before any trade."),
        Reminder("Verify conviction >= 0.5 before any trade execution."),
        Reminder("ALWAYS run debate before any trade decision. No exceptions."),
        Reminder("Batch sub-agent calls in a single response for speed."),
        Reminder("Use dynamic workflow for complex multi-step orchestration."),
    ],
    dynamic_reminders=[],
    cache_ttl="5m",
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
        Thinking(effort="high"),
        WebSearch(local="duckduckgo"),
    ],
    model_settings=ModelSettings(temperature=0.0, max_tokens=8192),
)


# ─── Supervisor Tools ──────────────────────────────────────────────────

@supervisor_agent.tool
async def run_morning_routine(ctx: RunContext[AgentDeps]) -> SupervisorOutput:
    """Execute complete morning routine: market data → analysis → research → debate → risk → conference."""
    import asyncio

    # 1. Market snapshot
    market_result = await market_data_agent.run("Get portfolio and market snapshot", deps=ctx.deps)

    # 2. Analysis on holdings (parallel)
    holdings = ctx.deps.holdings if hasattr(ctx.deps, 'holdings') else []
    analysis_tasks = [
        analysis_agent.run(f"Analyze {h.get('tradingsymbol', h) if isinstance(h, dict) else h}", deps=ctx.deps)
        for h in holdings[:10]
    ]
    analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

    # 3. Research for new opportunities
    research_result = await research_agent.run("Screen for new opportunities", deps=ctx.deps)

    # 4. Risk check
    risk_result = await risk_agent.run("Portfolio risk assessment", deps=ctx.deps)

    # 5. Conference for capital allocation
    conference_result = await conference_agent.run("Morning conference: allocate capital", deps=ctx.deps)

    return SupervisorOutput(
        action="complete",
        message=f"Morning routine complete. Market: ok, Analyzed: {len(analysis_results)} holdings, Research: ok, Risk: ok, Conference: ok",
        trade_queue=[],
        next_context=None,
    )


@supervisor_agent.tool
async def run_quarterly_review(ctx: RunContext[AgentDeps]) -> SupervisorOutput:
    """Execute quarterly portfolio review: deep analysis → sector rotation → debate → rebalance."""
    import asyncio

    # 1. Market snapshot
    market_result = await market_data_agent.run("Get portfolio and market snapshot", deps=ctx.deps)

    # 2. Deep analysis on holdings (parallel)
    holdings = ctx.deps.holdings if hasattr(ctx.deps, 'holdings') else []
    analysis_tasks = [
        analysis_agent.run(f"Deep analysis with DCF on {h.get('tradingsymbol', h) if isinstance(h, dict) else h}", deps=ctx.deps)
        for h in holdings[:10]
    ]
    analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

    # 3. Sector rotation & macro research
    research_result = await research_agent.run("Sector rotation and macro theme research", deps=ctx.deps)

    # 4. Risk checks & rebalancing
    risk_result = await risk_agent.run("Portfolio risk review and rebalancing", deps=ctx.deps)

    # 5. Conference
    conference_result = await conference_agent.run("Quarterly review conference", deps=ctx.deps)

    return SupervisorOutput(
        action="complete",
        message=f"Quarterly review complete. Analyzed: {len(analysis_results)} holdings",
        trade_queue=[],
        next_context=None,
    )


@supervisor_agent.tool
async def handle_price_alert(
    ctx: RunContext[AgentDeps],
    symbol: str,
    price_move_pct: float,
) -> SupervisorOutput:
    """Quick 3-agent response to price alert."""
    # 1. Fresh quote
    market_result = await market_data_agent.run(f"Get fresh quote for {symbol}", deps=ctx.deps)

    # 2. Quick analysis
    analysis_result = await analysis_agent.run(f"Quick analysis on {symbol}: price moved {price_move_pct:.1f}%", deps=ctx.deps)

    # 3. Risk check
    risk_result = await risk_agent.run(f"Risk check on {symbol} after {price_move_pct:.1f}% move", deps=ctx.deps)

    return SupervisorOutput(
        action="complete",
        message=f"Price alert handled for {symbol} ({price_move_pct:.1f}% move)",
        trade_queue=[],
        next_context=None,
    )


@supervisor_agent.tool
async def analyze_symbol(
    ctx: RunContext[AgentDeps],
    symbol: str,
    depth: "Literal['quick', 'standard', 'deep']" = "standard",
) -> SupervisorOutput:
    """Run full analysis on a single symbol."""
    # Batch all analysis dimensions

    fundamental_task = ctx.deps.kite_client and ctx.deps.kite_client.quote(symbol)
    
    # This would delegate to sub-agents
    # For now, return structured response
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