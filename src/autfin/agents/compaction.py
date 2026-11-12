"""Bound the daemon's history before the model's window does (B4).

`run.py` loads the `daemon:morning` conversation and saves each day's run into
it, so the supervisor's history grows by one full morning routine per trading
day — and every run resends all of it, plus the instructions and every tool
schema. Nothing bounded that: after a quarter of trading the prompt is mostly
yesterday's mornings.

`TieredCompaction` runs cheap strategies first and only reaches the expensive
one when the cheap passes cannot reclaim enough:

1. `ClearToolResults` — tool returns are the biggest bytes in the history
   (screening and financial payloads), and clearing old ones reclaims the most
   per unit of risk. The last `keep_pairs` request/return pairs stay intact.
2. `SummarizingCompaction` on the *fast* tier — a summarizer costs a model call,
   so it is the last resort, and it runs on the cheap model, not the deep one.

The target is a `target_fraction` of the model's context window rather than an
absolute token count, because this repo's endpoint reports no window at all:
`resolve_context_window()` returns `None` for the NIM models, so an absolute
count would be a guess about a deployment the registry does not describe, while
the fraction stays right whether the window resolves to 8k, 128k or falls back
to the 200k default.
"""
from __future__ import annotations

from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
)

from autfin.agents.base import get_model

# Compact when the history passes half the window: leave headroom for the
# instructions, every tool schema, and the reply itself.
TARGET_FRACTION = 0.5

# How many of the most recent request/return pairs survive the cheap tier. Three
# covers the tool calls of the step currently in flight.
KEEP_TOOL_PAIRS = 3

# The summarizer keeps this many recent messages verbatim and folds the rest.
KEEP_MESSAGES = 20


def get_history_compaction() -> TieredCompaction:
    """One shared capability for the supervisor: tiers are driven per request, so
    the instance holds no per-run state that two agents would need separate."""
    return TieredCompaction(
        tiers=[
            # max_tokens=1 is a formality: TieredCompaction drives each tier's
            # compact() directly and decides when to stop, so this tier's own
            # trigger never fires on its own. (A tier must still declare one.)
            ClearToolResults(keep_pairs=KEEP_TOOL_PAIRS, max_tokens=1),
            # The summarizer runs on the fast tier: this is the expensive tier, and
            # paying deep-tier prices to compress history would eat the savings.
            # Like the cheap tier, its own trigger is a formality — TieredCompaction
            # drives compact() directly — but a strategy must still declare one.
            SummarizingCompaction(
                model=get_model("fast"), keep_messages=KEEP_MESSAGES, max_messages=1
            ),
        ],
        target_fraction=TARGET_FRACTION,
    )
