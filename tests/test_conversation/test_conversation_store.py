"""Tests for the SQLite conversation store and native dataset serialisation."""
from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_conversation_round_trips_and_continues(tmp_path):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from autfin.conversation import SqliteConversationStore

    store = SqliteConversationStore(tmp_path / "conv.db")
    agent = Agent(TestModel())

    first = await agent.run("first turn")
    await store.save(first)

    loaded = await store.load(first.conversation_id)
    assert loaded is not None
    assert len(loaded.messages) == 2
    assert loaded.usage.requests == 1

    second = await agent.run("second turn", conversation=loaded)
    await store.save(second)

    assert second.conversation_id == first.conversation_id
    assert second.usage.requests == 2  # usage carries across runs, not resets


@pytest.mark.asyncio
async def test_save_twice_appends_rather_than_rewrites(tmp_path):
    """Core's conflict guard: the second save must not clobber the first."""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from autfin.conversation import SqliteConversationStore

    store = SqliteConversationStore(tmp_path / "conv.db")
    agent = Agent(TestModel())

    first = await agent.run("first turn")
    await store.save(first)

    reloaded = await store.load(first.conversation_id)
    second = await agent.run("second turn", conversation=reloaded)
    await store.save(second)

    again = await store.load(first.conversation_id)
    assert len(again.messages) == 4  # two turns, request+response each


@pytest.mark.asyncio
async def test_load_unknown_conversation_returns_none(tmp_path):
    from autfin.conversation import SqliteConversationStore

    store = SqliteConversationStore(tmp_path / "conv.db")
    assert await store.load("nope") is None


@pytest.mark.asyncio
async def test_conflicting_save_is_rejected_not_silently_overwritten(tmp_path):
    """Two writers on one conversation must raise, not silently overwrite."""
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.test import TestModel

    from autfin.conversation import SqliteConversationStore

    store = SqliteConversationStore(tmp_path / "conv.db")
    agent = Agent(TestModel())
    first = await agent.run("first turn")
    await store.save(first)

    stale = await store.load(first.conversation_id)
    await store.save(first)  # advances the stored revision

    with pytest.raises(UserError):
        await store.save(stale)  # stale copy → conflict, not overwrite


@pytest.mark.asyncio
async def test_fork_shares_history(tmp_path):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from autfin.conversation import SqliteConversationStore

    store = SqliteConversationStore(tmp_path / "conv.db")
    agent = Agent(TestModel())
    first = await agent.run("first turn")
    await store.save(first)

    forked_conversation = await store.fork(first.conversation_id)
    assert forked_conversation is not None
    assert forked_conversation.conversation_id != first.conversation_id
    assert len(forked_conversation.messages) == 2


# ── native dataset serialisation ───────────────────────────────────────────
def test_dataset_round_trips_through_the_native_api(tmp_path):
    """save_dataset/load_dataset reimplemented this and flattened evaluators to
    strings with default=str."""
    from pydantic_evals import Dataset

    from autfin.evals import create_autfin_dataset

    source = create_autfin_dataset(include_llm_judges=False, include_span_eval=True)
    out = tmp_path / "ds.json"
    source.to_file(out)

    # Native serialisation needs the custom evaluator types named explicitly —
    # the hand-rolled loader hid that by stringifying evaluators, which lost
    # their behaviour on reload.
    from autfin.evals.dataset import (
        AnalysisCompletenessEvaluator,
        PositionSizingEvaluator,
        RiskVetoEvaluator,
        TradeQualityEvaluator,
    )

    loaded = Dataset.from_file(
        out,
        custom_evaluator_types=[
            TradeQualityEvaluator,
            RiskVetoEvaluator,
            AnalysisCompletenessEvaluator,
            PositionSizingEvaluator,
        ],
    )
    assert len(loaded.cases) == len(source.cases)
    assert [c.name for c in loaded.cases] == [c.name for c in source.cases]
    assert len(loaded.evaluators) == len(source.evaluators)


def test_evals_package_no_longer_exports_handrolled_serialisation():
    import autfin.evals as evals

    assert not hasattr(evals, "save_dataset")
    assert not hasattr(evals, "load_dataset")
    assert not hasattr(evals, "run_evaluations")


# ── deferred orders survive the round-trip (B18) ─────────────────────────────


def test_a_paused_order_survives_the_store_round_trip(tmp_path):
    """A live order that pauses for approval must still be resumable after the
    conversation is saved and loaded: the deferred requests are part of the
    record, not just the messages."""
    from pydantic_ai import Agent, DeferredToolRequests
    from pydantic_ai.capabilities import HandleDeferredToolCalls
    from pydantic_ai.models.test import TestModel

    from autfin.agents.order_approval import resolve_order_approvals
    from autfin.agents.supervisor import execute_trade
    from autfin.conversation import SqliteConversationStore
    from autfin.types import AgentDeps

    store = SqliteConversationStore(tmp_path / "conv.db")

    def build():
        agent = Agent(
            TestModel(call_tools=["execute_trade"]),
            deps_type=AgentDeps,
            output_type=[str, DeferredToolRequests],
            capabilities=[HandleDeferredToolCalls(handler=resolve_order_approvals)],
        )
        agent.tool(execute_trade)
        return agent

    agent = build()
    result = agent.run_sync("place an order", deps=AgentDeps(mode="live", kite_client=object()))
    assert type(result.output).__name__ == "DeferredToolRequests"

    async def round_trip():
        await store.save(result)
        return await store.load(result.conversation_id)

    loaded = asyncio.run(round_trip())
    assert loaded is not None
    assert len(loaded.deferred_tool_requests.approvals) == 1
    assert [c.tool_name for c in loaded.deferred_tool_requests.approvals] == ["execute_trade"]
    # and the messages are still there to resume against
    assert len(loaded.messages) == len(result.all_messages())