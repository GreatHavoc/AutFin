"""Tests for B1: order tools pause for a human before reaching the broker."""
from __future__ import annotations

import pytest


def _agent(mode: str, kite=None):
    from pydantic_ai import Agent, DeferredToolRequests
    from pydantic_ai.capabilities import HandleDeferredToolCalls
    from pydantic_ai.models.test import TestModel

    from autfin.agents.order_approval import resolve_order_approvals
    from autfin.agents.supervisor import execute_trade
    from autfin.types import AgentDeps

    agent = Agent(
        TestModel(call_tools=["execute_trade"]),
        deps_type=AgentDeps,
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=resolve_order_approvals)],
    )
    agent.tool(execute_trade)
    return agent, AgentDeps(mode=mode, kite_client=kite or object())


def test_live_order_pauses_for_approval():
    """The gap this closes: guardrails checked arguments, nothing gated execution."""
    agent, deps = _agent("live")
    result = agent.run_sync("place an order", deps=deps)

    assert type(result.output).__name__ == "DeferredToolRequests"
    assert [c.tool_name for c in result.output.approvals] == ["execute_trade"]
    # the metadata the tool attached is what a reviewer would see
    meta = list(result.output.metadata.values())[0]
    assert "symbol" in meta and "quantity" in meta


def test_approved_live_order_resumes_and_executes():
    agent, deps = _agent("live")

    class Kite:
        def place_order(self, **kw):
            return "ORDER-1"

    agent2, deps2 = _agent("live", Kite())
    result = agent2.run_sync("place an order", deps=deps2)
    approvals = {c.tool_call_id: True for c in result.output.approvals}
    resumed = agent2.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=result.output.build_results(approvals=approvals),
        deps=deps2,
    )
    text = str(resumed.output)
    assert "LIVE TRADE" in text and "ORDER-1" in text


def test_denied_live_order_never_reaches_the_broker():
    agent, deps = _agent("live")

    class Kite:
        def place_order(self, **kw):
            raise AssertionError("broker must not be called for a denied order")

    result = agent.run_sync("place an order", deps=deps)
    approvals = {c.tool_call_id: False for c in result.output.approvals}
    resumed = agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=result.output.build_results(approvals=approvals),
        deps=deps,
    )
    assert "denied" in str(resumed.all_messages()).lower()


def test_paper_mode_does_not_block():
    """Paper mode never contacts Kite, so auto-approving saves a pointless pause."""
    agent, deps = _agent("paper")
    result = agent.run_sync("place an order", deps=deps)
    assert type(result.output).__name__ != "DeferredToolRequests"
    assert "PAPER TRADE" in str(result.output)


def test_the_shared_kite_toolset_is_wrapped():
    from autfin.tools import kite_toolset

    assert type(kite_toolset).__name__ == "ApprovalRequiredToolset"


def test_every_agent_carries_the_handler():
    import autfin.agents as agents
    from pydantic_ai.capabilities import HandleDeferredToolCalls

    for name in (
        "market_data", "analysis", "research", "risk", "debate",
        "conference", "memory", "quant", "supervisor",
    ):
        caps = getattr(agents, f"{name}_agent").root_capability.capabilities
        assert any(isinstance(c, HandleDeferredToolCalls) for c in caps), name
