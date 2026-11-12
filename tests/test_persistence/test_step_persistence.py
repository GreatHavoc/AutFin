"""Tests for the supervisor's StepPersistence crash-recovery layer.

The `runs` table this replaces recorded a run's start and end and nothing else —
after a crash you learned that a run had happened, not how far it got. These
tests pin the real contract: snapshots, an event trail, a tool-effect ledger,
and continuation from the last settled step.
"""
from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import StepPersistence
from pydantic_ai_harness.step_persistence import SqliteStepStore, continue_run, fork_run


def _store(tmp_path):
    return SqliteStepStore(database=tmp_path / "steps.db")


@pytest.mark.asyncio
async def test_supervisor_carries_the_shared_capability():
    from autfin.agents import supervisor_agent

    caps = supervisor_agent.root_capability.capabilities
    found = [c for c in caps if isinstance(c, StepPersistence)]
    assert len(found) == 1
    assert found[0].agent_name == "supervisor"
    # one shared store on the AutFin db, not a per-capability default
    from autfin.agents.persistence import get_step_store

    assert found[0].store is get_step_store()


@pytest.mark.asyncio
async def test_singletons_are_shared_not_rebuilt():
    from autfin.agents.persistence import get_step_persistence, get_step_store

    assert get_step_persistence() is get_step_persistence()
    assert get_step_store() is get_step_store()
    assert get_step_persistence().store is get_step_store()


@pytest.mark.asyncio
async def test_a_run_leaves_a_record_events_and_a_snapshot(tmp_path):
    store = _store(tmp_path)
    agent = Agent(TestModel(), capabilities=[StepPersistence(store=store, agent_name="t")])

    await agent.run("hello")

    runs = await store.list_runs()
    assert len(runs) == 1
    assert runs[0].agent_name == "t"

    # the store key is the derived (agent_name, run_id) pair, not result.run_id
    key = runs[0].run_id
    kinds = [e.kind for e in await store.list_events(run_id=key)]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_completed"

    snapshot = await store.latest_snapshot(run_id=key)
    assert snapshot is not None
    assert snapshot.state == "complete"
    assert snapshot.messages, "a settled step must persist its history"


@pytest.mark.asyncio
async def test_continue_and_fork_return_the_settled_history(tmp_path):
    store = _store(tmp_path)
    agent = Agent(TestModel(), capabilities=[StepPersistence(store=store, agent_name="t")])

    await agent.run("first turn")
    await agent.run("second turn")

    first, second = await store.list_runs()
    resumed = await continue_run(store, run_id=first.run_id)
    assert len(resumed) == 2, "the settled request/response pair comes back"

    forked = await fork_run(store, run_id=second.run_id)
    assert len(forked) == 2
    assert first.run_id != second.run_id, "each run gets its own store key"


@pytest.mark.asyncio
async def test_a_completed_tool_call_settles_in_the_ledger(tmp_path):
    store = _store(tmp_path)

    def square(ctx, x: int) -> int:
        """Square a number."""
        return x * x

    agent = Agent(
        TestModel(call_tools=["square"]),
        tools=[square],
        capabilities=[StepPersistence(store=store, agent_name="t")],
    )
    await agent.run("square 4")

    key = (await store.list_runs())[0].run_id
    kinds = [e.kind for e in await store.list_events(run_id=key)]
    assert any("tool" in k for k in kinds), kinds
    assert await store.list_unresolved_tool_effects(run_id=key) == [], (
        "a completed tool call must not read as unknown_after_crash"
    )


def test_the_dead_runs_table_is_gone():
    """record_run_start/record_run_end had zero callers and carried no recovery."""
    from autfin.memory import database

    assert not hasattr(database, "record_run_start")
    assert not hasattr(database, "record_run_end")
