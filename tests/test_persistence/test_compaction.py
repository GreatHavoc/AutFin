"""Tests for B4: the daemon's history is bounded before the window is.

`run.py` continues one conversation across trading days, so the supervisor's
history grows by a full morning routine per day and every run resends all of it.
These tests pin the escalation order (cheap tier first, summarizer last) and the
fractional target that keeps the config right on a model whose window the
pricing registry does not know.
"""
from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import CompactionPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
)

BIG = "Z" * 2000


def big_tool(ctx, symbol: str) -> str:
    """Return financials."""
    return BIG


def _summarizer(messages, info):
    from pydantic_ai.messages import ModelResponse, TextPart

    return ModelResponse(parts=[TextPart(content="SUMMARY OF THE HISTORY")])


def _agent(target_tokens: int) -> Agent:
    return Agent(
        # three tool calls, so the history holds pairs the cheap tier can clear
        TestModel(call_tools=["big_tool", "big_tool", "big_tool"]),
        tools=[big_tool],
        capabilities=[
            TieredCompaction(
                tiers=[
                    ClearToolResults(keep_pairs=1, max_tokens=1),
                    SummarizingCompaction(
                        model=FunctionModel(_summarizer), keep_messages=2, max_messages=1
                    ),
                ],
                target_tokens=target_tokens,
            )
        ],
    )


def _parts(result):
    return [p for m in result.all_messages() for p in m.parts]


# ── what the supervisor actually carries ────────────────────────────────────


def test_the_supervisor_carries_tiered_compaction():
    from pydantic_ai_harness.compaction import TieredCompaction

    import autfin.agents as agents

    def flat(agent):
        caps = list(agent.root_capability.capabilities)
        out = []
        while caps:
            cap = caps.pop()
            out.append(cap)
            caps.extend(getattr(cap, "capabilities", []) or [])
        return out

    found = [c for c in flat(agents.supervisor_agent) if isinstance(c, TieredCompaction)]
    assert len(found) == 1
    tiers = found[0].tiers
    assert [type(t).__name__ for t in tiers] == ["ClearToolResults", "SummarizingCompaction"]
    # a fraction, not an absolute token count: the NIM models report no context
    # window, so an absolute target would be a guess about an unknown deployment
    assert found[0].target_fraction is not None
    assert found[0].target_tokens is None
    assert found[0].fallback_context_window == 200_000
    # the summarizer runs on the cheap tier, not the deep one
    assert tiers[1].model is not None


# ── escalation order, driven through a real run ─────────────────────────────


@pytest.mark.asyncio
async def test_the_cheap_tier_clears_tool_results_before_summarizing():
    """A target the cheap tier can reach must not pay for a summarizer call."""
    agent = _agent(target_tokens=600)
    result = await agent.run("get financials for RELIANCE")

    returns = [p for p in _parts(result) if isinstance(p, ToolReturnPart)]
    assert returns, "the tool still ran"
    assert any("cleared" in str(p.content).lower() for p in returns), [
        str(p.content)[:60] for p in returns
    ]
    assert not any(isinstance(p, CompactionPart) for p in _parts(result)), (
        "the summarizer ran although clearing tool results was enough"
    )


@pytest.mark.asyncio
async def test_a_history_the_cheap_tier_cannot_shrink_reaches_the_summarizer():
    """Below what clearing can reclaim, the expensive tier is reached."""
    agent = _agent(target_tokens=400)
    result = await agent.run("get financials for RELIANCE")

    summaries = [p for p in _parts(result) if isinstance(p, CompactionPart)]
    assert summaries, "the summarizer never ran"
    assert "SUMMARY OF THE HISTORY" in str(summaries[0].content)
