"""Regression detection for trading performance degradation.

Monitors win rate, drawdown, calibration, and loss streaks for performance
degradation. Provides auto-pause logic for critical regressions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """An alert when performance degrades below threshold."""
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    severity: str    # "warning", "critical"
    recommendation: str


class RegressionDetector:
    """Monitors performance metrics for regression."""

    def __init__(
        self,
        min_trades_for_alert: int = 20,
        win_rate_warning: float = 0.40,
        win_rate_critical: float = 0.30,
        drawdown_warning: float = 0.15,
        drawdown_critical: float = 0.25,
        loss_streak_warning: int = 3,
        loss_streak_critical: int = 5,
    ):
        self.min_trades = min_trades_for_alert
        self.win_rate_warning = win_rate_warning
        self.win_rate_critical = win_rate_critical
        self.drawdown_warning = drawdown_warning
        self.drawdown_critical = drawdown_critical
        self.loss_streak_warning = loss_streak_warning
        self.loss_streak_critical = loss_streak_critical

    async def check_regression(self) -> List[RegressionAlert]:
        """Check all metrics for regression. Returns list of alerts."""
        alerts: List[RegressionAlert] = []
        alerts.extend(await self._check_win_rate())
        alerts.extend(await self._check_loss_streak())
        alerts.extend(await self._check_calibration())

        for alert in alerts:
            logger.warning(
                "Regression alert [%s]: %s — %s",
                alert.severity, alert.metric_name, alert.recommendation,
            )

        return alerts

    async def _check_win_rate(self) -> List[RegressionAlert]:
        """Check if recent win rate has degraded."""
        try:
            from src.autfin.memory.database import get_connection
            db = get_connection()
            recent = db.execute(
                "SELECT pnl_pct FROM learning_journal WHERE outcome='completed' "
                "ORDER BY created_at DESC LIMIT ?",
                [self.min_trades],
            ).fetchall()
        except Exception:
            return []

        if len(recent) < self.min_trades:
            return []

        wins = sum(1 for r in recent if r[0] and r[0] > 0)
        win_rate = wins / len(recent)
        baseline = 0.50

        alerts: List[RegressionAlert] = []
        if win_rate < self.win_rate_critical:
            alerts.append(RegressionAlert(
                metric_name="win_rate",
                current_value=round(win_rate, 4),
                baseline_value=baseline,
                degradation_pct=round((baseline - win_rate) / baseline * 100, 1),
                severity="critical",
                recommendation=f"Win rate {win_rate:.0%} is critically low. Consider pausing trading and reviewing strategy.",
            ))
        elif win_rate < self.win_rate_warning:
            alerts.append(RegressionAlert(
                metric_name="win_rate",
                current_value=round(win_rate, 4),
                baseline_value=baseline,
                degradation_pct=round((baseline - win_rate) / baseline * 100, 1),
                severity="warning",
                recommendation=f"Win rate {win_rate:.0%} is below target. Monitor closely.",
            ))
        return alerts

    async def _check_loss_streak(self) -> List[RegressionAlert]:
        """Check for consecutive losses."""
        try:
            from src.autfin.memory.database import get_connection
            db = get_connection()
            recent = db.execute(
                "SELECT pnl_pct FROM learning_journal WHERE outcome='completed' "
                "ORDER BY created_at DESC LIMIT 10",
            ).fetchall()
        except Exception:
            return []

        streak = 0
        for row in recent:
            if row[0] and row[0] < 0:
                streak += 1
            else:
                break

        if streak >= self.loss_streak_critical:
            return [RegressionAlert(
                metric_name="loss_streak",
                current_value=float(streak),
                baseline_value=0.0,
                degradation_pct=float(streak * 10),
                severity="critical",
                recommendation=f"{streak} consecutive losses. Pause trading and review recent decisions.",
            )]
        elif streak >= self.loss_streak_warning:
            return [RegressionAlert(
                metric_name="loss_streak",
                current_value=float(streak),
                baseline_value=0.0,
                degradation_pct=float(streak * 10),
                severity="warning",
                recommendation=f"{streak} consecutive losses. Monitor closely.",
            )]
        return []

    async def _check_calibration(self) -> List[RegressionAlert]:
        """Check if conviction calibration has degraded."""
        alerts: List[RegressionAlert] = []
        try:
            from src.autfin.learning.retriever import _compute_calibration
            from src.autfin.memory.database import get_connection
            db = get_connection()
            calibration = _compute_calibration(db)
        except Exception:
            return []

        for tier, data in calibration.items():
            if isinstance(data, dict) and data.get("sample_size", 0) >= 10:
                delta = data.get("delta", 0)
                if delta < -0.15:
                    alerts.append(RegressionAlert(
                        metric_name=f"calibration_{tier}",
                        current_value=data.get("actual_win_rate", 0),
                        baseline_value=data.get("expected_win_rate", 0.5),
                        degradation_pct=round(abs(delta) * 100, 1),
                        severity="warning",
                        recommendation=f"{tier} tier is overconfident by {abs(delta):.0%}. Apply conviction discount.",
                    ))
        return alerts

    async def should_auto_pause(self) -> tuple[bool, str]:
        """Determine if trading should be auto-paused.

        Returns:
            (should_pause, reason)
        """
        alerts = await self.check_regression()
        critical = [a for a in alerts if a.severity == "critical"]

        if len(critical) >= 2:
            return True, f"Multiple critical regressions detected: {[a.metric_name for a in critical]}"
        if any(a.metric_name == "loss_streak" and a.current_value >= self.loss_streak_critical for a in critical):
            return True, f"{self.loss_streak_critical}+ consecutive losses — auto-pause triggered"
        if any(a.metric_name == "win_rate" and a.current_value < self.win_rate_critical for a in critical):
            return True, "Win rate critically low — auto-pause triggered"
        return False, ""


async def check_performance_regression() -> List[RegressionAlert]:
    """Convenience function for scheduler integration."""
    detector = RegressionDetector()
    return await detector.check_regression()
