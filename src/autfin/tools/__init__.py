"""AutFin v7.0 — Combined Toolsets

Exports all toolsets and provides combined toolsets for agents.
"""
from __future__ import annotations

from pydantic_ai import FunctionToolset
from pydantic_ai.toolsets import CombinedToolset

from autfin.tools.kite_toolset import kite_toolset, get_kite_toolset
from autfin.tools.fundamental_toolset import fundamental_toolset, get_fundamental_toolset
from autfin.tools.technical_toolset import technical_toolset, get_technical_toolset
from autfin.tools.screening_toolset import screening_toolset, get_screening_toolset
from autfin.tools.sentiment_toolset import sentiment_toolset, get_sentiment_toolset

from autfin.types import AgentDeps


# ─── Individual Toolsets ───
__all__ = [
    "kite_toolset",
    "fundamental_toolset", 
    "technical_toolset",
    "screening_toolset",
    "sentiment_toolset",
    "get_kite_toolset",
    "get_fundamental_toolset",
    "get_technical_toolset",
    "get_screening_toolset",
    "get_sentiment_toolset",
    "market_data_toolset",
    "analysis_toolset",
    "research_toolset",
    "risk_toolset",
]


# ─── Agent-Specific Combined Toolsets ───

# MarketDataAgent: portfolio, quotes, market status
market_data_toolset = CombinedToolset[AgentDeps]([
    kite_toolset,
])


# AnalysisAgent: fundamental, technical, sentiment
analysis_toolset = CombinedToolset[AgentDeps]([
    fundamental_toolset,
    technical_toolset,
    sentiment_toolset,
])


# ResearchAgent: screening, fundamental
research_toolset = CombinedToolset[AgentDeps]([
    screening_toolset,
    fundamental_toolset,
])


# RiskAgent: kite (portfolio), technical (ATR)
risk_toolset = CombinedToolset[AgentDeps]([
    kite_toolset,
    technical_toolset,
])


# ─── Toolset Factory for Agents ───
def get_toolsets_for_agent(agent_name: str) -> list[FunctionToolset[AgentDeps]]:
    """Get appropriate toolsets for a given agent."""
    toolset_map = {
        "market_data": [kite_toolset],
        "analyze": [fundamental_toolset, technical_toolset, sentiment_toolset],
        "research": [screening_toolset, fundamental_toolset],
        "risk": [kite_toolset, technical_toolset],
        "debate": [],  # Debate uses analysis data, no direct tools
        "conference": [kite_toolset],  # For portfolio data
        "memory": [],  # Memory uses its own capability
    }
    return toolset_map.get(agent_name, [])


def create_agent_with_toolsets(
    name: str,
    output_type: type,
    instructions: str,
    model_type: str = "deep",
    **kwargs
):
    """Create an agent with pre-configured toolsets."""
    from autfin.agents.base import create_agent
    
    toolsets = get_toolsets_for_agent(name)
    return create_agent(
        name=name,
        output_type=output_type,
        instructions=instructions,
        model_type=model_type,
        # Toolsets are passed via capabilities in Pydantic AI
        # They will be added as toolsets in the agent
    )