"""Feedback loop orchestrator — closes the loop: retrieve → decide → execute → observe → learn."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def pre_trade_context(
    symbol: str,
    action: str,
    sector: Optional[str] = None,
    raw_conviction: float = 0.5,
) -> dict:
    """
    Gather learning context before a trade decision.
    Called by the supervisor before execute_order to enrich the decision.

    Returns:
        {
            similar_setups: [...],       # past trades with same symbol/sector
            calibration: {...},          # conviction tier calibration
            patterns: [...],             # detected risk patterns
            conviction_adjustment: {...}, # calibrated conviction
        }
    """
    try:
        from autfin.learning.calibrator import CalibrationTracker
        from autfin.learning.retriever import retrieve_learnings, search_similar_setups

        query = f"{action} {symbol}"
        if sector:
            query += f" {sector}"

        similar_task = search_similar_setups(symbol=symbol, action=action, limit=5)
        retriever_task = retrieve_learnings(query=query, filters={"symbol": symbol}, limit=5)
        tracker = CalibrationTracker()
        calibration_task = tracker.get_calibration()
        adjustment_task = tracker.suggested_conviction_adjustment(raw_conviction, sector)

        similar, retriever_result, calibration, adjustment = await _gather(
            similar_task, retriever_task, calibration_task, adjustment_task
        )

        all_similar = similar + retriever_result.get("similar_setups", [])
        seen = set()
        deduped = []
        for s in all_similar:
            key = (s.get("symbol"), s.get("date"))
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        return {
            "similar_setups": deduped[:10],
            "calibration": calibration,
            "patterns": retriever_result.get("patterns", []),
            "conviction_adjustment": adjustment,
        }
    except Exception as e:
        logger.debug("pre_trade_context failed: %s", e, exc_info=True)
        return {
            "similar_setups": [],
            "calibration": {},
            "patterns": [],
            "conviction_adjustment": {},
            "error": str(e),
        }


async def post_trade_learn(
    symbol: str,
    action: str,
    conviction: float,
    pnl_pct: float,
    thesis_summary: str,
    sector: Optional[str] = None,
    outcome: str = "completed",
) -> dict:
    """
    Record a completed trade outcome and return updated calibration.
    Called after a position is closed (sell). The scheduler was removed from
    scope, so this is now invoked directly by the trade-close flow.

    Returns: {recorded: bool, updated_calibration: {...}, new_patterns: [...]}
    """
    try:
        from autfin.learning.calibrator import (
            CalibrationTracker,
            detect_patterns,
            record_learning,
        )

        recorded = await record_learning(
            symbol=symbol,
            action=action,
            conviction=conviction,
            pnl_pct=pnl_pct,
            thesis_summary=thesis_summary,
            outcome=outcome,
            sector=sector,
        )

        tracker = CalibrationTracker()
        calibration = await tracker.get_calibration()
        patterns = await detect_patterns(min_samples=5)

        return {
            "recorded": recorded,
            "updated_calibration": calibration,
            "new_patterns": patterns,
        }
    except Exception as e:
        logger.debug("post_trade_learn failed: %s", e, exc_info=True)
        return {"recorded": False, "error": str(e)}


async def get_learning_summary() -> dict:
    """
    Get a full learning summary for the supervisor's periodic context refresh.
    Returns calibration, patterns, and recent trade history.
    """
    try:
        from autfin.learning.calibrator import CalibrationTracker, detect_patterns
        from autfin.learning.retriever import retrieve_learnings

        tracker = CalibrationTracker()
        calibration_task = tracker.get_calibration()
        patterns_task = detect_patterns(min_samples=5)
        recent_task = retrieve_learnings(query="recent trades", limit=10)

        calibration, patterns, recent = await _gather(
            calibration_task, patterns_task, recent_task
        )

        return {
            "calibration": calibration,
            "patterns": patterns,
            "recent_trades": recent.get("similar_setups", []) if isinstance(recent, dict) else [],
        }
    except Exception as e:
        logger.debug("get_learning_summary failed: %s", e, exc_info=True)
        return {"calibration": {}, "patterns": [], "recent_trades": [], "error": str(e)}


async def _gather(*aw):
    """Run multiple awaitables concurrently, returning results in order."""
    import asyncio
    results = await asyncio.gather(*aw, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, Exception):
            logger.debug("gather item failed: %s", r)
            out.append(None)
        else:
            out.append(r)
    return out
