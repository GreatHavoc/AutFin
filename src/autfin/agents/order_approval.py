"""Order tools need a human yes before they reach the broker (B1).

The gap this closes: `guardrails.validate_trade_args` checks the *arguments* of an
order (quantity positive, price present, symbol sane), but nothing gates
*execution* — paper mode is safe only because it never calls Kite at all.

Two places reach an order:
  - `kite_toolset.place_order`, carried by market_data / risk / conference
  - the supervisor's own `execute_trade` tool

Both now pause, and the policy is in one place: paper mode auto-approves (nothing
leaves the box), live mode leaves the calls unresolved so the run ends with
`DeferredToolRequests` and a person decides.
"""
from __future__ import annotations

import logging

from pydantic_ai import DeferredToolRequests, DeferredToolResults

logger = logging.getLogger(__name__)

ORDER_TOOLS = frozenset({"place_order", "execute_trade"})


def _is_order_tool(tool_name: str) -> bool:
    """Only order tools pause. Everything else runs as before."""
    return tool_name in ORDER_TOOLS


async def resolve_order_approvals(ctx, requests: DeferredToolRequests):
    """Resolve pending approvals.

    Returns `None` for anything this policy declines to decide, which core
    treats as "bubble up": the unresolved calls surface as DeferredToolRequests
    output for the caller to put in front of a person.
    """
    if getattr(ctx.deps, "mode", "paper") == "paper":
        # Paper mode never contacts Kite, so a human yes adds nothing.
        logger.info("Paper mode: auto-approving %d order call(s)", len(requests.approvals))
        return requests.build_results(approve_all=True)

    logger.warning(
        "Live mode: %d order call(s) await approval %s",
        len(requests.approvals),
        [c.tool_name for c in requests.approvals],
    )
    return None
