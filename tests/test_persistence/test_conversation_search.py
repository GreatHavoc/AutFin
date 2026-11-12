"""Tests for the memory agent's ConversationSearch capability (B8).

The learning retriever's `retrieve_learnings()` claims semantic search but
silently falls back to sha256 "hash embeddings" (sentence-transformers is not a
dependency), so its similarity scores are noise. ConversationSearch replaces
that kind of guesswork for conversation recall: BM25 over the history
StepPersistence already wrote to the shared step store, scoped to the calling
conversation.
"""
from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import ConversationSearch, StepPersistence
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import SqliteStepStore

QUERY = "REPLINANCE conviction"
RARE = "REPLINANCE"


def _store(tmp_path):
    return SqliteStepStore(database=tmp_path / "steps.db")


def _searcher(store):
    """A FunctionModel that asks the search tool once, then answers."""

    def search_then_answer(messages: list[ModelMessage], info) -> ModelResponse:
        if any(isinstance(p, ToolCallPart) and p.tool_name == "search_conversation_history" for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[ToolCallPart("search_conversation_history", {"query": QUERY})]
        )

    return Agent(
        FunctionModel(search_then_answer),
        capabilities=[
            StepPersistence(store=store, agent_name="t"),
            ConversationSearch(SnapshotHistorySource(store), scope="conversation"),
        ],
    )


def _seeder(store):
    return Agent(
        FunctionModel(lambda messages, info: ModelResponse(
            parts=[TextPart(f"the {RARE} conviction settled at 0.82 after the dip")]
        )),
        capabilities=[StepPersistence(store=store, agent_name="t")],
    )


def _tool_returns(result):
    return [
        str(part.content)
        for m in result.all_messages()
        for part in m.parts
        if part.part_kind == "tool-return"
    ]


def test_memory_agent_carries_conversation_search_on_the_shared_store():
    import autfin.agents as agents

    def flat(agent):
        caps = list(agent.root_capability.capabilities)
        out = []
        while caps:
            cap = caps.pop()
            out.append(cap)
            caps.extend(getattr(cap, "capabilities", []) or [])
        return out

    found = [c for c in flat(agents.memory_agent) if isinstance(c, ConversationSearch)]
    assert len(found) == 1
    assert found[0].scope == "conversation"


@pytest.mark.asyncio
async def test_the_same_conversation_recalls_the_earlier_text(tmp_path):
    store = _store(tmp_path)
    seeder = _seeder(store)
    await seeder.run("first turn", conversation_id="conv-1")

    searcher = _searcher(store)
    result = await searcher.run("recall it", conversation_id="conv-1")

    returns = _tool_returns(result)
    assert any("0.82" in r for r in returns), returns


@pytest.mark.asyncio
async def test_another_conversations_history_is_out_of_scope(tmp_path):
    store = _store(tmp_path)
    seeder = _seeder(store)
    await seeder.run("first turn", conversation_id="conv-1")

    searcher = _searcher(store)
    result = await searcher.run("recall it", conversation_id="conv-2")

    returns = " ".join(_tool_returns(result))
    assert "0.82" not in returns, returns
    assert "conversation" in returns, returns
