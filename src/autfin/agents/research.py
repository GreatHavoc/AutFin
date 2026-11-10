"""AutFin v7.0 — Research Agent

Deep research, screening, and thesis generation.
"""
from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Thinking, WebSearch
from pydantic_ai_harness.planning import Planning

from autfin.agents.base import create_agent
from autfin.agents.managed_prompts import research_instructions
from autfin.tools import research_toolset
from autfin.tools.browser_agent import get_browser_toolset
from autfin.types import (
    AgentDeps,
    ResearchReport,
    ScreeningCandidate,
    ScreeningResult,
)

research_agent = create_agent(
    name="research",
    output_type=ResearchReport,
    model_type="deep",
    instructions=research_instructions.default,
    capabilities=[
        Planning(store=None, enable_subtasks=True),  # In-memory by default
        Thinking(effort="high"),
        WebSearch(local="duckduckgo"),
    ],
    toolsets=[research_toolset, get_browser_toolset()],
)


@research_agent.tool
async def deep_research(ctx: RunContext[AgentDeps], symbol: str) -> ResearchReport:
    """Conduct deep research on a single symbol."""
    from autfin.agents.researcher import conduct_deep_research

    return await conduct_deep_research(symbol, ctx.deps.as_of)


@research_agent.tool
async def screen_universe(
    ctx: RunContext[AgentDeps],
    criteria: dict | None = None,
    sector: str | None = None,
    max_results: int = 20
) -> ScreeningResult:
    """Screen NSE universe for opportunities."""
    from autfin.tools.screening import run_screening_pipeline

    result = await run_screening_pipeline()
    candidate_details = result.get("candidate_details", [])

    candidates = [
        ScreeningCandidate(
            symbol=c["symbol"],
            composite_score=c.get("composite_score", 0),
            fundamental_score=c.get("earnings_yield", 0),
            technical_score=c.get("roic", 0),
            sentiment_score=0,
            valuation_score=c.get("pe", 0) or 0,
            sector=c.get("sector", ""),
            market_cap=c.get("market_cap", 0),
            recommendation="buy" if c.get("composite_score", 0) > 0.5 else "hold",
        )
        for c in candidate_details
    ]

    # Sector distribution
    sector_dist = {}
    for c in candidates:
        sector_dist[c.sector] = sector_dist.get(c.sector, 0) + 1

    top_picks = [c.symbol for c in candidates[:5]]

    return ScreeningResult(
        candidates=candidates,
        top_picks=top_picks,
        sector_distribution=sector_dist,
    )


@research_agent.tool
async def generate_thesis(
    ctx: RunContext[AgentDeps],
    symbol: str,
    analysis_data: dict | None = None
) -> ResearchReport:
    """Generate investment thesis from analysis data."""
    from autfin.agents.researcher import generate_investment_thesis

    return await generate_investment_thesis(symbol, analysis_data, ctx.deps.as_of)


@research_agent.tool
async def research_sector(ctx: RunContext[AgentDeps], sector: str) -> dict:
    """Research sector trends, rotation, top picks."""
    from autfin.agents.global_macro import analyze_sector

    return await analyze_sector(sector, ctx.deps.as_of)


@research_agent.tool
async def find_catalysts(ctx: RunContext[AgentDeps], symbol: str) -> list[str]:
    """Find upcoming catalysts for a symbol (earnings, events, policy)."""
    from autfin.tools.fundamental_tools import get_upcoming_catalysts

    return await get_upcoming_catalysts(symbol, ctx.deps.as_of)