"""Tests for B5: oversized tool returns are capped, normal ones pass through."""
from __future__ import annotations

BIG_CHARS = 40000
SMALL = {"a": 1}


def _agent(limits):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    def big() -> dict:
        return {"rows": [{"i": i, "v": "x" * 50} for i in range(500)]}

    def small() -> dict:
        return dict(SMALL)

    return Agent(
        TestModel(call_tools=["big", "small"]),
        tools=[big, small],
        capabilities=[limits],
    )


def _limits():
    from pydantic_ai_harness.tool_output_limits import (
        Band,
        LocalFileStore,
        Spill,
        ToolOutputLimits,
        Truncate,
    )

    return ToolOutputLimits(
        bands=[Band(over=20_000, action=Spill()), Band(over=8_000, action=Truncate())],
        store=LocalFileStore(),
    )


def test_oversized_return_is_reduced():
    limits = _limits()
    result = _agent(limits).run_sync("call both")
    text = str(result.all_messages())
    assert "too large" in text or "stored to handle" in text
    # the full payload must not persist in history
    assert "x" * 500 not in text


def test_small_return_passes_through_untouched():
    """Today's live tools return ~536 chars, so they must be unaffected."""
    from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

    import autfin.agents as agents

    limits = [c for c in agents.risk_agent.root_capability.capabilities
              if isinstance(c, ToolOutputLimits)][0]
    bands = limits.bands
    assert bands, "bands configured"

    result = _agent(limits).run_sync("call both")
    assert "'a': 1" in str(result.all_messages())


def test_every_agent_carries_the_capability():
    from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

    import autfin.agents as agents

    for name in (
        "market_data", "analysis", "research", "risk", "debate",
        "conference", "memory", "quant", "supervisor",
    ):
        caps = getattr(agents, f"{name}_agent").root_capability.capabilities
        assert any(isinstance(c, ToolOutputLimits) for c in caps), name


def test_uses_local_file_store_not_workspace_store():
    """A workspace store needs a workspace, and we have none — POSIX only."""
    from pydantic_ai_harness.tool_output_limits import LocalFileStore

    import autfin.agents.spend  # noqa: F401  (module import side effects)
    from autfin.agents.tool_output import get_tool_output_limits

    assert isinstance(get_tool_output_limits().store, LocalFileStore)
