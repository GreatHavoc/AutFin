"""Tests for DetectLoops and Caching (phase 4, priority 2).

The supervisor's own system reminders tell the model to batch its sub-agent
calls; nothing detected when it looped anyway. And every request resent the
instructions plus every tool schema with no cache setting in play.
"""
from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.detect_loops import DetectLoops


def _flat(agent):
    caps = list(agent.root_capability.capabilities)
    out = []
    while caps:
        cap = caps.pop()
        out.append(cap)
        caps.extend(getattr(cap, "capabilities", []) or [])
    return out


def test_every_agent_carries_caching():
    import autfin.agents as agents

    for name in ("market_data", "analysis", "research", "risk", "memory", "quant", "supervisor"):
        agent = getattr(agents, f"{name}_agent")
        assert sum(1 for c in _flat(agent) if isinstance(c, Caching)) == 1, name


def test_the_supervisor_carries_loop_detection():
    import autfin.agents as agents

    found = [c for c in _flat(agents.supervisor_agent) if isinstance(c, DetectLoops)]
    assert len(found) == 1
    # nudge, not error: a loop costs budget, it is not a reason to kill the run
    assert found[0].action == "nudge"


@pytest.mark.asyncio
async def test_a_repeated_tool_call_is_detected():
    """The detection machinery, observed through the `error` action it shares with
    the nudge: the same call repeated past the threshold is a loop, and a loop is
    not simply re-run forever."""
    from pydantic_ai_harness.detect_loops import LoopDetectedError

    def stubborn(messages, info):
        # ctx is the RunContext slot the framework fills in for every tool call
        return ModelResponse(parts=[ToolCallPart("square", {"ctx": "a", "x": 4})])

    def square(ctx, x: int) -> int:
        """Square a number."""
        return x * x

    agent = Agent(
        FunctionModel(stubborn),
        tools=[square],
        capabilities=[DetectLoops(repeated_call_threshold=3, action="error")],
    )
    # a request cap too, so a detection that failed to fire cannot spin the test
    with pytest.raises((LoopDetectedError, UsageLimitExceeded)) as excinfo:
        await agent.run("square four", usage_limits=UsageLimits(request_limit=12))

    assert isinstance(excinfo.value, LoopDetectedError), (
        f"the loop was never detected: {type(excinfo.value).__name__}"
    )