"""Conviction calibration — tracks actual vs predicted win rates and auto-adjusts."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CalibrationTracker:
    """Tracks conviction calibration across tiers, sectors, and time periods."""

    def __init__(self):
        self._cache: dict = {}

    async def get_calibration(self) -> dict:
        """Get current calibration state using learning_retriever."""
        from src.autfin.learning.retriever import _compute_calibration
        try:
            from src.autfin.memory.database import get_connection
            db = get_connection()
            return _compute_calibration(db)
        except Exception as e:
            logger.debug("Calibration fetch failed: %s", e, exc_info=True)
            return {}

    async def suggested_conviction_adjustment(
        self, raw_conviction: float, sector: str | None = None
    ) -> dict:
        """Suggest conviction adjustment based on historical calibration."""
        calibration = await self.get_calibration()

        tier = (
            "high_conv" if raw_conviction >= 0.7
            else "medium_conv" if raw_conviction >= 0.4
            else "low_conv"
        )
        tier_data = calibration.get(tier, {})
        delta = tier_data.get("delta", 0)

        adjusted = max(0.1, min(0.99, raw_conviction + delta))

        return {
            "raw_conviction": raw_conviction,
            "tier": tier,
            "calibration_delta": delta,
            "adjusted_conviction": round(adjusted, 3),
            "sample_size": tier_data.get("sample_size", 0),
            "note": (
                f"{tier_data.get('status', 'unknown')} — adjusted by {delta:+.3f}"
                if tier_data
                else "No calibration data yet"
            ),
        }


async def calibrate_conviction(
    raw_conviction: float, sector: str | None = None
) -> float:
    """Calibrate a raw conviction score based on historical outcomes. Returns adjusted score."""
    tracker = CalibrationTracker()
    result = await tracker.suggested_conviction_adjustment(raw_conviction, sector)
    return result["adjusted_conviction"]


async def detect_patterns(min_samples: int = 10) -> list[dict]:
    """Detect recurring patterns in trade outcomes."""
    try:
        from src.autfin.memory.database import get_connection
        db = get_connection()

        patterns: list[dict] = []

        sector_rows = db.execute(
            "SELECT sector, COUNT(*) as cnt, "
            "SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins "
            "FROM learning_journal WHERE outcome='completed' "
            "GROUP BY sector HAVING cnt >= ?",
            [min_samples]
        ).fetchall()

        for row in sector_rows:
            sector, total, wins = row[0], row[1], row[2] or 0
            win_rate = wins / max(total, 1)
            patterns.append({
                "type": "sector_performance",
                "sector": sector,
                "trades": total,
                "win_rate": round(win_rate, 3),
                "assessment": (
                    "STRONG" if win_rate >= 0.6
                    else "AVERAGE" if win_rate >= 0.4
                    else "WEAK"
                ),
            })

        dow_rows = db.execute(
            "SELECT strftime('%w', created_at) as dow, COUNT(*), "
            "SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) "
            "FROM learning_journal WHERE outcome='completed' "
            "GROUP BY dow HAVING COUNT(*) >= 3"
        ).fetchall()

        dow_names = {
            "0": "Sunday", "1": "Monday", "2": "Tuesday",
            "3": "Wednesday", "4": "Thursday", "5": "Friday", "6": "Saturday",
        }
        for row in dow_rows:
            dow, total, wins = row[0], row[1], row[2] or 0
            patterns.append({
                "type": "day_pattern",
                "day": dow_names.get(dow, dow),
                "trades": total,
                "win_rate": round(wins / max(total, 1), 3),
            })

        return patterns
    except Exception as e:
        logger.debug("Pattern detection failed: %s", e, exc_info=True)
        return []


async def record_learning(
    symbol: str,
    action: str,
    conviction: float,
    pnl_pct: float,
    thesis_summary: str,
    outcome: str = "completed",
    sector: str | None = None,
) -> bool:
    """Record a trade outcome in the learning journal and update online calibration."""
    try:
        from src.autfin.memory.database import get_connection
        db = get_connection()
        db.execute(
            "INSERT INTO learning_journal "
            "(symbol, action, conviction, pnl_pct, thesis_summary, outcome, sector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                symbol, action, conviction, pnl_pct, thesis_summary,
                outcome, sector, datetime.now(tz=timezone.utc).isoformat(),
            ]
        )
        db.commit()
        logger.info(
            "Learning recorded: %s %s @ conv %.2f → %.1f%%",
            action, symbol, conviction, pnl_pct * 100,
        )

        # Update online learner with this trade outcome
        try:
            from src.autfin.learning.online import get_online_learner
            learner = get_online_learner()
            # Convert PnL to binary outcome: 1.0 = win, 0.0 = loss
            binary_outcome = 1.0 if pnl_pct > 0 else 0.0
            learner.update(
                conviction=conviction,
                outcome=binary_outcome,
                sector=sector,
                pnl_pct=pnl_pct,
            )
        except Exception as e:
            logger.debug("Online learner update failed: %s", e)

        return True
    except Exception as e:
        logger.warning("Failed to record learning: %s", e)
        return False
