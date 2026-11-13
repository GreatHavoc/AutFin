"""The Memory capability runs live, which FileStore never could here.

`Memory(FileStore(...))` resolves its directory from the run's workspace and
raised `UserError` in `before_run` on every live run, because the only local
workspace backend refuses non-POSIX platforms. These tests pin the store swap:
SQLite, no workspace, shared by both agents that carry the capability.
"""
from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.memory import Memory, SqliteMemoryStore


def _answer(messages, info):
    return ModelResponse(parts=[TextPart(content="acknowledged")])


def _flat(agent):
    caps = list(agent.root_capability.capabilities)
    out = []
    while caps:
        cap = caps.pop()
        out.append(cap)
        caps.extend(getattr(cap, "capabilities", []) or [])
    return out


def _memory_capabilities(agent):
    return [c for c in _flat(agent) if isinstance(c, Memory)]


def test_both_agents_use_the_sqlite_store_not_a_file_store():
    import autfin.agents as agents
    from autfin.memory.store import get_memory_store

    for name in ("supervisor", "memory"):
        found = _memory_capabilities(getattr(agents, f"{name}_agent"))
        assert len(found) == 1, name
        assert isinstance(found[0].store, SqliteMemoryStore), name
        # one process-wide store, so the two agents share rows
        assert found[0].store is get_memory_store(), name


@pytest.mark.asyncio
async def test_the_capability_runs_with_no_workspace_attached():
    """before_run is exactly where FileStore raised UserError."""
    from autfin.memory.store import get_memory_store

    agent = Agent(
        FunctionModel(_answer),
        capabilities=[Memory(get_memory_store(), namespace="probe", inject_memory=True)],
    )
    result = await agent.run("remember that RELIANCE earnings are on Friday")
    assert str(result.output) == "acknowledged"


@pytest.mark.asyncio
async def test_memory_rows_round_trip(tmp_path):
    """Write, list and read through the store the agents now use."""
    from pydantic_ai_harness.memory import SqliteMemoryStore

    store = SqliteMemoryStore(database=tmp_path / "mem.db")
    await store.write("notes.md", "RELIANCE earnings are on Friday", expected_version=None)
    assert await store.list_paths(limit=10) == ["notes.md"]
    got = await store.read("notes.md", max_chars=2000)
    assert got is not None and "RELIANCE" in got.content