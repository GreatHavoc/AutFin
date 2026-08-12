"""AutFin v7.0 — Market Data Agent

Fetches real-time portfolio, quotes, and market status.
Uses KiteToolset for all broker interactions.
"""
from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import WebSearch
from pydantic_ai_harness.filesystem import FileSystem

from autfin.agents.base import create_agent, get_model
from autfin.agents.managed_prompts import market_data_instructions
from autfin.agents.observability import create_debug_instrumentation
from autfin.tools import kite_toolset, market_data_toolset
from autfin.types import (
    AgentDeps,
    MarketSnapshotOutput,
    Quote,
    Holding,
    MarketStatus,
)


market_data_agent = create_agent(
    name="market_data",
    output_type=MarketSnapshotOutput,
    model_type="fast",
    instructions=market_data_instructions.default,
    capabilities=[
        WebSearch(local="duckduckgo"),
        FileSystem(root_dir="./data/market"),
        create_debug_instrumentation(),
    ],
    toolsets=[market_data_toolset],
)


@market_data_agent.tool
async def get_watchlist_quotes(ctx: RunContext[AgentDeps], watchlist: str = "") -> dict:
    """Fetch quotes for watchlist symbols."""
    if not watchlist:
        return {"quotes": {}}
    
    # Delegate to the toolset's get_quotes
    from autfin.tools.kite_toolset import get_quotes
    return await get_quotes(ctx, watchlist)


@market_data_agent.tool
async def check_risk_alerts(ctx: RunContext[AgentDeps], quotes: dict) -> list[str]:
    """Check for risk alerts in quotes (gap >3%, circuit breakers, volume spikes)."""
    alerts = []

    for symbol, data in quotes.items():
        try:
            change_pct = data.get("change_pct", 0) or data.get("net_change_pct", 0)
            if abs(change_pct) > 3:
                direction = "UP" if change_pct > 0 else "DOWN"
                alerts.append(f"{symbol}: Gap {direction} {abs(change_pct):.1f}%")

            # Check for circuit breakers
            lower_circuit = data.get("lower_circuit_limit")
            upper_circuit = data.get("upper_circuit_limit")
            ltp = data.get("last_price") or data.get("ltp")
            if lower_circuit and ltp and ltp <= lower_circuit * 1.01:
                alerts.append(f"{symbol}: Near lower circuit ({ltp:.2f})")
            if upper_circuit and ltp and ltp >= upper_circuit * 0.99:
                alerts.append(f"{symbol}: Near upper circuit ({ltp:.2f})")

            # Volume spike
            volume = data.get("volume", 0)
            avg_volume = data.get("average_volume", 1)
            if avg_volume > 0 and volume > avg_volume * 3:
                alerts.append(f"{symbol}: Volume spike {volume/avg_volume:.1f}x avg")

        except Exception:
            continue

    return alerts