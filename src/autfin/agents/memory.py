"""AutFin v7.0 — Memory Agent

Cross-session memory: market events, trade outcomes, patterns, calibration.
"""
from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext
from pydantic_ai_harness.memory import FileStore, Memory

from autfin.agents.base import create_agent
from autfin.agents.managed_prompts import memory_instructions
from autfin.types import (
    AgentDeps,
    ExecutionReport,
    MemoryResult,
    TradeOrder,
)

memory_agent = create_agent(
    name="memory",
    output_type=MemoryResult,
    model_type="fast",
    instructions=memory_instructions.default,
    capabilities=[
        Memory(
            FileStore("./data/memory"),
            namespace=lambda ctx: ctx.deps.user_id,
            max_tokens=5000,
            max_lines=500,
            inject_memory=True,
        ),
    ],
)


@memory_agent.tool
async def record_market_event(
    ctx: RunContext[AgentDeps],
    event_type: Literal["regime_change", "volatility_spike", "correlation_breakdown", "sector_rotation", "macro_event"],
    description: str,
    symbols: list[str] | None = None,
    metadata: dict | None = None,
) -> MemoryResult:
    """Record a market event to memory."""
    from datetime import datetime

    entry = {
        "type": "market_event",
        "event_type": event_type,
        "description": description,
        "symbols": symbols or [],
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "regime": metadata.get("regime", "unknown") if metadata else "unknown",
    }

    # Write to memory using the Memory capability
    from pydantic_ai_harness.memory import MemoryToolset
    mem_capability = Memory(
        FileStore("./data/memory"),
        namespace=ctx.deps.user_id,
    )
    toolset = MemoryToolset(mem_capability)
    result = await toolset.write_memory(ctx, content=str(entry), file="MARKET_EVENTS.md")

    return MemoryResult(
        action="record_market",
        result={"status": "recorded", "entry": entry},
        files_updated=["MARKET_EVENTS.md"],
    )


@memory_agent.tool
async def record_trade_outcome(
    ctx: RunContext[AgentDeps],
    order: TradeOrder,
    execution: ExecutionReport,
    analysis_context: dict | None = None,
) -> MemoryResult:
    """Record trade outcome with full context for calibration."""
    from datetime import datetime

    pnl_pct = 0.0
    if execution.avg_price and order.price:
        if order.action == "buy":
            pnl_pct = float((execution.avg_price - order.price) / order.price * 100)
        else:
            pnl_pct = float((order.price - execution.avg_price) / order.price * 100)

    entry = {
        "type": "trade_outcome",
        "symbol": order.symbol,
        "action": order.action,
        "quantity": order.quantity,
        "entry_price": float(order.price) if order.price else 0,
        "exit_price": float(execution.avg_price) if execution.avg_price else 0,
        "pnl_pct": pnl_pct,
        "risk_reward": order.risk_reward,
        "conviction": order.conviction,
        "analysis_context": analysis_context or {},
        "timestamp": datetime.now().isoformat(),
        "status": execution.status,
    }

    # Write to memory
    from pydantic_ai_harness.memory import MemoryToolset
    mem_capability = Memory(
        FileStore("./data/memory"),
        namespace=ctx.deps.user_id,
    )
    toolset = MemoryToolset(mem_capability)
    result = await toolset.write_memory(ctx, content=str(entry), file="TRADE_OUTCOMES.md")

    return MemoryResult(
        action="record_trade",
        result={"status": "recorded", "entry": entry},
        files_updated=["TRADE_OUTCOMES.md"],
    )


@memory_agent.tool
async def record_pattern(
    ctx: RunContext[AgentDeps],
    pattern_name: str,
    description: str,
    symbols: list[str],
    conditions: dict,
    success_rate: float,
    sample_size: int,
) -> MemoryResult:
    """Record a discovered trading pattern."""
    from datetime import datetime

    entry = {
        "type": "pattern",
        "pattern_name": pattern_name,
        "description": description,
        "symbols": symbols,
        "conditions": conditions,
        "success_rate": success_rate,
        "sample_size": sample_size,
        "discovered_at": datetime.now().isoformat(),
    }

    from pydantic_ai_harness.memory import MemoryToolset
    mem_capability = Memory(
        FileStore("./data/memory"),
        namespace=ctx.deps.user_id,
    )
    toolset = MemoryToolset(mem_capability)
    result = await toolset.write_memory(ctx, content=str(entry), file="PATTERNS.md")

    return MemoryResult(
        action="record_pattern",
        result={"status": "recorded", "entry": entry},
        files_updated=["PATTERNS.md"],
    )


@memory_agent.tool
async def get_context_for_symbol(
    ctx: RunContext[AgentDeps],
    symbol: str,
    context_type: Literal["market", "trades", "patterns", "all"] = "all",
) -> MemoryResult:
    """Retrieve relevant memories for a symbol."""
    from pydantic_ai_harness.memory import MemoryToolset
    mem_capability = Memory(
        FileStore("./data/memory"),
        namespace=ctx.deps.user_id,
    )
    toolset = MemoryToolset(mem_capability)

    result = {}
    if context_type in ("market", "all"):
        market_mem = await toolset.search_memory(ctx, query=f"{symbol} market event")
        result["market_events"] = market_mem

    if context_type in ("trades", "all"):
        trade_mem = await toolset.search_memory(ctx, query=f"{symbol} trade outcome")
        result["trade_outcomes"] = trade_mem

    if context_type in ("patterns", "all"):
        pattern_mem = await toolset.search_memory(ctx, query=f"{symbol} pattern")
        result["patterns"] = pattern_mem

    return MemoryResult(
        action="get_context",
        result={"symbol": symbol, "context": result},
        files_updated=[],
    )


@memory_agent.tool
async def get_calibration(
    ctx: RunContext[AgentDeps],
    agent_name: str | None = None,
    context: str | None = None,
) -> MemoryResult:
    """Get conviction calibration stats (accuracy of confidence estimates)."""
    from pydantic_ai_harness.memory import MemoryToolset
    mem_capability = Memory(
        FileStore("./data/memory"),
        namespace=ctx.deps.user_id,
    )
    toolset = MemoryToolset(mem_capability)

    query_parts = ["calibration"]
    if agent_name:
        query_parts.append(agent_name)
    if context:
        query_parts.append(context)

    calib_mem = await toolset.search_memory(ctx, query=" ".join(query_parts))

    # Parse calibration data from memory
    # This would aggregate conviction vs actual outcomes
    calibration = {
        "total_trades": 0,
        "correct_high_conviction": 0,
        "correct_low_conviction": 0,
        "brier_score": None,
        "reliability_diagram": [],
    }

    return MemoryResult(
        action="get_calibration",
        result={"calibration": calibration, "raw": calib_mem},
        files_updated=[],
    )


@memory_agent.tool
async def get_memory_stats(ctx: RunContext[AgentDeps]) -> MemoryResult:
    """Get memory system statistics."""
    from pydantic_ai_harness.memory import MemoryToolset
    mem_capability = Memory(
        FileStore("./data/memory"),
        namespace=ctx.deps.user_id,
    )
    toolset = MemoryToolset(mem_capability)

    # Read all memory files
    files = ["MEMORY.md", "MARKET_EVENTS.md", "TRADE_OUTCOMES.md", "PATTERNS.md"]
    stats = {"files": {}}

    for f in files:
        try:
            content = await toolset.read_memory(ctx, file=f)
            lines = content.count("\n")
            chars = len(content)
            stats["files"][f] = {"lines": lines, "chars": chars}
        except Exception:
            stats["files"][f] = {"lines": 0, "chars": 0}

    return MemoryResult(
        action="get_stats",
        result={"stats": stats},
        files_updated=[],
    )