"""Cap oversized tool returns (B5).

Measured against the live toolset before choosing thresholds: `get_technical_data`
returns ~536 chars, `get_financials` ~400, `calculate_sma(250 bars)` ~1.7k. Nothing
currently floods the context — the 200-symbol financials dump lives in
`fetch_universe_data`, which is only reachable from the unwired curator.

So this is a guard set well above today's payloads rather than a fix for a
measured problem: every return below the smallest band passes through untouched,
and the spill only engages when something genuinely large appears.

The store is LocalFileStore, not the default WorkspaceStore: spills land in the
run's workspace, and we have no workspace — LocalWorkspace is POSIX-only and
DockerSandbox hard-codes that same runner, so a spill-capable agent would fail at
its start without this.
"""
from __future__ import annotations

import os

from pydantic_ai_harness import ToolOutputLimits
from pydantic_ai_harness.tool_output_limits import Band, LocalFileStore, Spill, Truncate

# Characters. Generous because a return under this is cheap; the point is catching
# a payload that would be re-sent on every later request of the run.
SPILL_OVER = int(os.getenv("AUTFIN_TOOL_SPILL_CHARS", "20000"))
TRUNCATE_OVER = int(os.getenv("AUTFIN_TOOL_TRUNCATE_CHARS", "8000"))

_limits: ToolOutputLimits | None = None


def get_tool_output_limits() -> ToolOutputLimits:
    """One shared instance, matching the B1/B2 pattern."""
    global _limits
    if _limits is None:
        _limits = ToolOutputLimits(
            bands=[
                Band(over=SPILL_OVER, action=Spill()),
                Band(over=TRUNCATE_OVER, action=Truncate()),
            ],
            # LocalFileStore writes to the temp dir; a workspace needs POSIX here.
            store=LocalFileStore(cleanup_after=None),
        )
    return _limits
