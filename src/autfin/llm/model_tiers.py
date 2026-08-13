"""
Model tier registry -- route tasks to the right model by complexity.

DEEP  = ThinkingMachines Inkling -- complex reasoning (debate, conferences, research synthesis)
MEDIUM = Meta Muse Glimmer 30B -- standard analysis (fundamentals, technical, valuation)
FAST   = Meta Muse Glimmer 30B -- lightweight tasks (market data, memory, macro)
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ModelTier(Enum):
    DEEP = "deep"
    MEDIUM = "medium"
    FAST = "fast"

# Maps tier -> model name (resolved at runtime from LLMClient)
MODEL_REGISTRY: dict[ModelTier, str] = {
    ModelTier.DEEP: "nvidia/nemotron-3.5-lightning-30b-a3b",
    ModelTier.MEDIUM: "nvidia/nemotron-3.5-lightning-30b-a3b",
    ModelTier.FAST: "nvidia/nemotron-3.5-lightning-30b-a3b",
}

# Maps agent/subagent name -> model tier
AGENT_TIERS: dict[str, ModelTier] = {
    # Analysts -- medium complexity (deterministic tools, structured output)
    "analyze_fundamentals": ModelTier.MEDIUM,
    "analyze_technical": ModelTier.MEDIUM,
    "analyze_sentiment": ModelTier.MEDIUM,
    "analyze_valuation": ModelTier.MEDIUM,
    "analyze_distortion": ModelTier.MEDIUM,
    "analyze_regulatory": ModelTier.FAST,
    "analyze_macro": ModelTier.FAST,
    # Synthesis -- deep reasoning
    "research_synthesis": ModelTier.DEEP,
    "run_debate": ModelTier.DEEP,
    "run_conference": ModelTier.DEEP,
    # Data & memory -- lightweight
    "get_market_data": ModelTier.FAST,
    "manage_memory_system": ModelTier.FAST,
    # Trade execution -- medium
    "execute_trade": ModelTier.MEDIUM,
}


def get_model_for_agent(agent_name: str) -> str:
    """Resolve the model name for a given agent/subagent.

    Falls back to the DEEP model for unknown agents.
    """
    tier = AGENT_TIERS.get(agent_name, ModelTier.DEEP)
    model = MODEL_REGISTRY.get(tier, MODEL_REGISTRY[ModelTier.DEEP])
    return model


def get_tier_for_agent(agent_name: str) -> ModelTier:
    """Get the ModelTier for an agent name."""
    return AGENT_TIERS.get(agent_name, ModelTier.DEEP)