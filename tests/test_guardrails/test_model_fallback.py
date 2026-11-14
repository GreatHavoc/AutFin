"""Tests for the tier fallback (phase 4, priority 1).

Measured before this: NIM's throttle is per-model and counted in tokens per
minute, so a throttled deep tier ended the whole run. One analysis sub-agent
spent 11k input tokens; two of those in the same minute and the run dies.
"""
from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.exceptions import ModelRateLimitError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel


def _flat(agent):
    caps = list(agent.root_capability.capabilities)
    out = []
    while caps:
        cap = caps.pop()
        out.append(cap)
        caps.extend(getattr(cap, "capabilities", []) or [])
    return out


def _fallback_of(agent):
    return [c for c in _flat(agent) if isinstance(c, ModelFallback)]


def test_every_agent_falls_back_to_the_other_tier():
    import autfin.agents as agents
    from autfin.types import MODEL_CONFIG

    for name in ("market_data", "analysis", "research", "risk", "memory", "quant", "supervisor"):
        agent = getattr(agents, f"{name}_agent")
        found = _fallback_of(agent)
        assert len(found) == 1, name
        fallback_id = getattr(found[0].models[0], "model_name", None)
        own_id = getattr(agent.model, "model_name", None)
        # the fallback must be the *other* tier, never the model already serving
        assert fallback_id != own_id, name
        assert fallback_id in {MODEL_CONFIG.deep, MODEL_CONFIG.fast}, name


@pytest.mark.asyncio
async def test_a_throttled_step_is_served_by_the_fallback():
    """The failure this closes: a 429 on the primary ended the run."""

    def throttled(messages, info):
        raise ModelRateLimitError(model_name="primary", message="429 Too Many Requests")

    def served(messages, info):
        return ModelResponse(parts=[TextPart(content="served by the fallback")])

    agent = Agent(
        FunctionModel(throttled),
        capabilities=[ModelFallback(FunctionModel(served))],
    )
    result = await agent.run("do the thing")
    assert str(result.output) == "served by the fallback"
    # the response names the model that actually answered
    assert [m.model_name for m in result.all_messages() if hasattr(m, "model_name")] == [
        "function:served:"
    ]


def test_no_fallback_when_the_tiers_are_the_same_model(monkeypatch):
    """A fallback to the same id only retries the endpoint that just refused."""
    import autfin.agents.base as base

    class Same:
        deep = "same/model"
        fast = "same/model"
        reasoning = "same/model"
        code = "same/model"

    monkeypatch.setattr(base, "MODEL_CONFIG", Same())
    agent = base.create_agent(
        name="same_tier_probe", output_type=str, instructions="x", model_type="deep"
    )
    assert _fallback_of(agent) == [], "the pointless fallback was still attached"