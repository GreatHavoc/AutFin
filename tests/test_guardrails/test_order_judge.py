"""Tests for the live-order judge — the live-trading rung of the safety ladder.

The five capabilities the harness ships for human-in-the-loop: four are declined
with recorded reasons (see agents/safety.py), and ToolCallJudge is attached
because it runs in before_tool_execute, sees the call's validated arguments, and
can stop the tool body before it runs.
"""
from __future__ import annotations

from types import SimpleNamespace

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

from autfin.agents.order_approval import resolve_order_approvals
from autfin.agents.safety import JUDGED_TOOLS, get_order_judge
from autfin.agents.supervisor import execute_trade
from autfin.types import AgentDeps


def _flat(agent):
    caps = list(agent.root_capability.capabilities)
    out = []
    while caps:
        cap = caps.pop()
        out.append(cap)
        caps.extend(getattr(cap, "capabilities", []) or [])
    return out


def test_the_supervisor_carries_the_order_judge():
    import autfin.agents as agents

    found = [c for c in _flat(agents.supervisor_agent) if isinstance(c, ToolCallJudge)]
    assert len(found) == 1
    # fail closed, and the judge sees only the call's arguments
    assert found[0].on_uncertain == "block"
    assert found[0].include_conversation is False


def test_paper_mode_judges_nothing_and_live_mode_judges_only_the_order():
    """The selector is a ToolSelectorFunc, so the gate is per-run: paper mode
    matches nothing and the judge costs nothing."""
    judge = get_order_judge()
    selector = judge.tools

    for mode, expected in (("paper", False), ("live", True)):
        ctx = SimpleNamespace(deps=SimpleNamespace(mode=mode))
        assert selector(ctx, SimpleNamespace(name="execute_trade")) is expected, mode
        # every other tool is unjudged, in every mode
        assert selector(ctx, SimpleNamespace(name="analyze_symbol")) is False, mode
    assert "execute_trade" in JUDGED_TOOLS


def _agent(judge_verdict: str, mode: str):
    """A supervisor-shaped agent whose judge answers with a fixed verdict.

    The judge's model is an agent with `output_type=Literal['yes','no','unsure']`,
    so it must answer through the generated output tool: a plain-text response
    fails output validation, which reads as `answer=None` and falls through to
    `on_uncertain`.
    """
    judge_model = FunctionModel(
        lambda messages, info: ModelResponse(
            parts=[ToolCallPart("final_result", {"response": judge_verdict})]
        )
    )

    def judge(ctx, tool_def) -> bool:
        return bool(ctx.deps.mode == "live") and tool_def.name in JUDGED_TOOLS

    agent = Agent(
        TestModel(call_tools=["execute_trade"]),
        deps_type=AgentDeps,
        output_type=[str, DeferredToolRequests],
        capabilities=[
            HandleDeferredToolCalls(handler=resolve_order_approvals),
            ToolCallJudge(model=judge_model, question="does it breach the limits?", tools=judge),
        ],
    )
    agent.tool(execute_trade)
    return agent


def _run(agent, mode):
    return agent.run_sync(
        "place an order", deps=AgentDeps(mode=mode, kite_client=object())
    )


def _text(result):
    """Every part of every message as text: tool returns carry the outcome, and
    tool calls carry the arguments."""
    out = []
    for m in result.all_messages():
        for p in m.parts:
            out.append(str(getattr(p, "content", None) or getattr(p, "args", None) or p))
    return " ".join(out)


def test_a_judged_risky_order_is_blocked_before_the_tool_body_runs():
    agent = _agent("yes", "live")  # the judge says: breaches the limits
    result = _run(agent, "live")

    text = _text(result)
    assert "blocked by a tool-call judge" in text, text


def test_a_judged_sound_order_proceeds():
    agent = _agent("no", "live")  # the judge says: within the limits
    result = _run(agent, "live")

    text = _text(result)
    assert "blocked by a tool-call judge" not in text, text


def test_paper_mode_never_asks_the_judge():
    """The daemon's default mode: the selector matches nothing, so the verdict
    the judge would give is irrelevant."""
    agent = _agent("yes", "paper")
    result = _run(agent, "paper")

    text = _text(result)
    assert "blocked by a tool-call judge" not in text, text
    assert "PAPER TRADE" in text, text