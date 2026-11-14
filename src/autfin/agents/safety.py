"""Untrusted-input defenses, attached to the graph rather than promised by it.

Two gaps this closes:

1. **Indirect prompt injection.** Every tool that returns third-party text —
   the Obscura browser scrape in `tools/browser_agent.py`, the sentiment tools
   that read Reddit and Moneycontrol, yfinance news — hands the model text an
   outside party wrote. `PromptInjectionDefender` classifies each locally
   executed tool result and withholds the ones that carry instructions aimed at
   the agent, replacing them with a note so the content never reaches the model.
   It was installed as a dependency in the upgrade and never attached, which is
   why the plan called this the highest unmitigated risk in the repo.

2. **Agent-written artifacts.** Memory is injected into every run's context, so
   anything an agent writes there is a second chance to plant an instruction.
   `ArtifactScanner` runs over what is stored, and flags the patterns that
   matter — role override, exfiltration, invisible unicode, injection, HTML
   comments.

Both are process-wide singletons: the defender is stateless per result, and the
scanner is a set of patterns, so one instance shared by every agent is the same
work done once per call rather than per agent.
"""
from __future__ import annotations

from pydantic_ai_harness.artifact_scanning import ArtifactScanner
from pydantic_ai_harness.prompt_injection_defender import PromptInjectionDefender
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

from autfin.agents.base import get_model

# Explicit id: a defender on a durable-capable agent needs one, and naming it now
# means a durable capability can be added later without a migration.
DEFENDER_ID = "injection_defense"

# Pattern detection only. `semantic_detection=True` adds a local ML classifier
# that also catches injection under unrecognised fields, at the cost of loading a
# model into every process; revisit when the scraped corpus outgrows patterns.
_injection_defender: PromptInjectionDefender | None = None
_artifact_scanner: ArtifactScanner | None = None


def get_injection_defense() -> PromptInjectionDefender:
    """Classify tool results and withhold the ones carrying instructions."""
    global _injection_defender
    if _injection_defender is None:
        _injection_defender = PromptInjectionDefender(
            block_high_risk=True, id=DEFENDER_ID
        )
    return _injection_defender


def get_artifact_scanner() -> ArtifactScanner:
    """Scan what the agents write into memory before it is stored."""
    global _artifact_scanner
    if _artifact_scanner is None:
        _artifact_scanner = ArtifactScanner()
    return _artifact_scanner


# ── the live-trading rung of the safety ladder ──────────────────────────────
#
# The harness ships five human-in-the-loop capabilities. Four are declined here
# with measured reasons, and one is attached:
#
#   AskUser       needs an `answerer`; this daemon runs unattended at 09:15 with
#                 no interactive surface, and B1 already pauses the one decision
#                 that needs a human (a live order) by bubbling DeferredToolRequests.
#   Permissions   gates tools; with a `mode='full'` default it is a no-op, and the
#                 repo exposes no file/shell tools to gate. Order tools are already
#                 wrapped in ApprovalRequiredToolset.
#   ApprovalReviewer  reviews every call with a second model. Overlaps the judge
#                 below at twice the cost, on the endpoint whose tokens-per-minute
#                 throttle is the binding constraint.
#   TrajectoryJudge   one model call every 10 steps regardless of mode. Worth it
#                 for an interactive operator; on this endpoint it trades the
#                 routine's reliability for diagnostics paper mode does not need.
#
# ToolCallJudge is the one that earns its model call: it runs in
# before_tool_execute, sees the call's validated arguments, and can stop the tool
# body before it runs — and it is scoped by a ToolSelector predicate that reads
# the run's mode, so paper mode (the default, and what the daemon runs) pays
# nothing at all.

JUDGE_QUESTION = (
    "A trading agent proposes this order. Does it breach the stated risk limits "
    "(risk/reward at least 2.0, conviction at least 0.5, position within the "
    "configured cap of capital)? Answer yes if it breaches them, no if it does "
    "not, and unsure if the arguments do not say."
)
JUDGED_TOOLS = ("execute_trade",)


def _judge_live_orders_only(ctx, tool_def) -> bool:
    """Live runs only, and only the tool that can move money.

    A ToolSelectorFunc receives the run context, so the gate is per-run: paper
    mode matches nothing and the judge costs nothing, live mode judges the order.
    """
    return bool(getattr(ctx.deps, "mode", None) == "live") and tool_def.name in JUDGED_TOOLS


def get_order_judge() -> ToolCallJudge:
    """A second model reviews a live order before the tool body runs.

    Fail closed: `unsure` blocks. The judge sees only the tool name and its
    validated arguments — `include_conversation=True` would widen its own
    prompt-injection surface, because the conversation holds text the agent read
    from pages and tool results.
    """
    return ToolCallJudge(
        # the fast tier, not the deep one the supervisor runs: a judge that is
        # the same model as the caller rubber-stamps its own reasoning
        get_model("fast"),
        question=JUDGE_QUESTION,
        tools=_judge_live_orders_only,
        include_conversation=False,
        on_uncertain="block",
    )