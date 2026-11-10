"""AutFin v7.0 — Learning Toolset

Exposes the self-learning engine (`autfin/learning/`) to agents: conviction
calibration, similar past setups, detected patterns, and trade-outcome capture.

Unlike the LLM-free learning modules, these tools take `RunContext[AgentDeps]`
so a supervisor or risk agent can pull calibration into a decision.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, RunContext

from autfin.types import AgentDeps


def create_learning_toolset() -> FunctionToolset[AgentDeps]:
    """Create a FunctionToolset with all self-learning tools."""
    toolset = FunctionToolset[AgentDeps]()

    @toolset.tool
    async def get_calibration(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
        """Conviction calibration by tier: how often each conviction level
        actually won, versus how often it should have.

        Returns {tier: {sample_size, expected_win_rate, actual_win_rate, delta,
        status}}. A negative delta means that tier is overconfident — discount it.
        An empty dict means no trade outcomes recorded yet.
        """
        from autfin.learning.calibrator import CalibrationTracker

        return await CalibrationTracker().get_calibration()

    @toolset.tool
    async def get_learning_context(
        ctx: RunContext[AgentDeps],
        symbol: str,
        action: str,
        sector: str = "",
        raw_conviction: float = 0.5,
    ) -> dict[str, Any]:
        """Pull the trading context for a candidate trade: similar past setups,
        conviction calibration, risk patterns, and a calibrated conviction.

        Call this before sizing an order. `conviction_adjustment` contains
        `adjusted_conviction` — use that, not `raw_conviction`, when deciding.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            action: Proposed action ("BUY" or "SELL")
            sector: Optional sector name (e.g., "Energy")
            raw_conviction: Uncalibrated conviction 0.0-1.0
        """
        from autfin.learning.feedback_loop import pre_trade_context

        return await pre_trade_context(
            symbol=symbol,
            action=action,
            sector=sector or None,
            raw_conviction=raw_conviction,
        )

    @toolset.tool
    async def record_trade_outcome(
        ctx: RunContext[AgentDeps],
        symbol: str,
        action: str,
        conviction: float,
        pnl_pct: float,
        thesis_summary: str,
        sector: str = "",
    ) -> dict[str, Any]:
        """Record a closed trade so future calibration can learn from it.

        Call this after a position is closed, not when it is opened.
        `pnl_pct` is a fraction, e.g. 0.12 for +12%.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            action: Action taken ("BUY" or "SELL")
            conviction: Conviction at entry, 0.0-1.0
            pnl_pct: Realised return as a fraction (0.12 = +12%)
            thesis_summary: Short description of the thesis and why it worked
            sector: Optional sector name (e.g., "Energy")
        """
        from autfin.learning.feedback_loop import post_trade_learn

        result = await post_trade_learn(
            symbol=symbol,
            action=action,
            conviction=conviction,
            pnl_pct=pnl_pct,
            thesis_summary=thesis_summary,
            sector=sector or None,
        )
        return {
            "recorded": result.get("recorded", False),
            "error": result.get("error"),
            "updated_calibration": result.get("updated_calibration", {}),
            "new_patterns": result.get("new_patterns", []),
        }

    @toolset.tool
    async def get_learning_summary(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
        """Overall learning state: tier calibration, detected risk patterns,
        and recent recorded trades. Useful for a periodic context refresh.
        """
        from autfin.learning.feedback_loop import get_learning_summary

        return await get_learning_summary()

    @toolset.tool
    async def calibrate_conviction(
        ctx: RunContext[AgentDeps],
        raw_conviction: float,
        sector: str = "",
    ) -> float:
        """Calibrate a conviction score against historical outcomes.

        Returns the adjusted conviction (0.1-0.99). Without calibration data
        the raw score is returned unchanged.

        Args:
            raw_conviction: Uncalibrated conviction 0.0-1.0
            sector: Optional sector name (e.g., "Energy")
        """
        from autfin.learning.calibrator import calibrate_conviction as _calibrate

        return await _calibrate(raw_conviction, sector or None)

    return toolset


# Create the default toolset instance
learning_toolset = create_learning_toolset()


def get_learning_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared learning toolset instance."""
    return learning_toolset
