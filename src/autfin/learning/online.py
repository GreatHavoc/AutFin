"""Online learning for AutFin — incremental calibration updates.

Implements incremental Bayesian updates to calibration buckets after
each trade closes. Instead of batch-recomputing calibration from all
history, this module updates running statistics in O(1) per trade.

Uses online Platt scaling: after each trade outcome, update the
bucket's running mean predicted conviction and mean actual outcome.
This gives us real-time calibration without rescanning the full DB.

All arithmetic is deterministic — no LLM involved.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BucketStats:
    """Running statistics for a single calibration bucket."""
    n: int = 0
    sum_conviction: float = 0.0
    sum_outcome: float = 0.0
    sum_conv_sq: float = 0.0
    sum_conv_outcome: float = 0.0

    @property
    def mean_conviction(self) -> float:
        return self.sum_conviction / max(self.n, 1)

    @property
    def mean_outcome(self) -> float:
        return self.sum_outcome / max(self.n, 1)

    @property
    def calibration_error(self) -> float:
        """Mean absolute difference between predicted and actual."""
        return abs(self.mean_conviction - self.mean_outcome)

    @property
    def brier_score(self) -> float:
        """Brier score: mean squared difference between predicted and actual."""
        if self.n == 0:
            return 0.0
        return self.sum_conv_outcome / self.n - 2 * self.mean_conviction * self.mean_outcome + self.mean_outcome ** 2

    @property
    def variance(self) -> float:
        """Variance of conviction predictions."""
        if self.n < 2:
            return 0.0
        return (self.sum_conv_sq / self.n) - (self.mean_conviction ** 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean_conviction": round(self.mean_conviction, 4),
            "mean_outcome": round(self.mean_outcome, 4),
            "calibration_error": round(self.calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "variance": round(self.variance, 6),
        }


def _bucket_for_conviction(conviction: float) -> str:
    """Map conviction to a bucket name."""
    if conviction >= 0.8:
        return "very_high"
    elif conviction >= 0.6:
        return "high"
    elif conviction >= 0.4:
        return "medium"
    elif conviction >= 0.2:
        return "low"
    else:
        return "very_low"


class OnlineLearner:
    """Incrementally updates calibration buckets after each trade.

    Usage:
        learner = OnlineLearner()
        # After a trade closes:
        learner.update(conviction=0.75, outcome=1.0)  # won
        learner.update(conviction=0.30, outcome=0.0)  # lost
        # Check calibration:
        report = learner.get_report()
    """

    def __init__(self):
        self._buckets: Dict[str, BucketStats] = defaultdict(BucketStats)
        self._sector_buckets: Dict[str, Dict[str, BucketStats]] = defaultdict(lambda: defaultdict(BucketStats))
        self._total_trades = 0
        self._total_pnl = 0.0
        self._recent_errors: List[float] = []
        self._max_recent = 50

    def update(
        self,
        conviction: float,
        outcome: float,
        sector: Optional[str] = None,
        pnl_pct: float = 0.0,
    ):
        """Incrementally update calibration with a new trade outcome.

        Args:
            conviction: Predicted conviction (0.0-1.0)
            outcome: Actual outcome (1.0 = win, 0.0 = loss)
            sector: Optional sector for sector-specific calibration
            pnl_pct: Trade P&L percentage
        """
        bucket = _bucket_for_conviction(conviction)

        # Update global bucket
        b = self._buckets[bucket]
        b.n += 1
        b.sum_conviction += conviction
        b.sum_outcome += outcome
        b.sum_conv_sq += conviction * conviction
        b.sum_conv_outcome += conviction * outcome

        # Update sector-specific bucket
        if sector:
            sb = self._sector_buckets[sector][bucket]
            sb.n += 1
            sb.sum_conviction += conviction
            sb.sum_outcome += outcome
            sb.sum_conv_sq += conviction * conviction
            sb.sum_conv_outcome += conviction * outcome

        # Track overall stats
        self._total_trades += 1
        self._total_pnl += pnl_pct

        # Track recent errors for drift detection
        error = abs(conviction - outcome)
        self._recent_errors.append(error)
        if len(self._recent_errors) > self._max_recent:
            self._recent_errors.pop(0)

        logger.debug(
            "Online update: conv=%.2f, outcome=%.1f, bucket=%s, error=%.3f",
            conviction, outcome, bucket, error,
        )

    def get_bucket(self, conviction: float) -> BucketStats:
        """Get stats for a specific conviction bucket."""
        bucket = _bucket_for_conviction(conviction)
        return self._buckets[bucket]

    def get_sector_bucket(self, sector: str, conviction: float) -> BucketStats:
        """Get sector-specific stats for a conviction bucket."""
        bucket = _bucket_for_conviction(conviction)
        return self._sector_buckets[sector][bucket]

    def get_all_buckets(self) -> Dict[str, Dict[str, Any]]:
        """Get all bucket stats."""
        return {k: v.to_dict() for k, v in self._buckets.items()}

    def get_sector_report(self, sector: str) -> Dict[str, Dict[str, Any]]:
        """Get calibration report for a specific sector."""
        if sector not in self._sector_buckets:
            return {}
        return {k: v.to_dict() for k, v in self._sector_buckets[sector].items()}

    def get_adjustment(self, raw_conviction: float) -> float:
        """Suggest conviction adjustment based on online calibration.

        Returns adjusted conviction by pulling toward the bucket's
        mean outcome (how that conviction tier actually performs).
        """
        bucket = self.get_bucket(raw_conviction)
        if bucket.n < 5:
            # Not enough data, return raw
            return raw_conviction

        # Pull toward the bucket's actual performance
        target = bucket.mean_outcome
        # Blend: 70% raw, 30% calibrated (conservative)
        adjusted = raw_conviction * 0.7 + target * 0.3
        return max(0.05, min(0.95, adjusted))

    def get_drift(self) -> Dict[str, Any]:
        """Check if recent calibration is drifting from historical."""
        if len(self._recent_errors) < 10:
            return {
                "drifting": False,
                "recent_error": 0.0,
                "note": "Insufficient recent data",
            }

        recent = self._recent_errors[-10:]
        older = self._recent_errors[:-10] if len(self._recent_errors) > 10 else self._recent_errors

        recent_error = sum(recent) / len(recent)
        older_error = sum(older) / len(older) if older else recent_error

        # Drift: recent error > older error by 20%+
        drift_pct = (
            (recent_error - older_error) / max(older_error, 0.01) * 100
            if older_error > 0
            else 0.0
        )
        drifting = drift_pct > 20.0

        return {
            "drifting": drifting,
            "recent_error": round(recent_error, 4),
            "older_error": round(older_error, 4),
            "drift_pct": round(drift_pct, 1),
            "note": (
                "Calibration is drifting — recent errors are increasing"
                if drifting
                else "Calibration is stable"
            ),
        }

    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive online learning report."""
        all_buckets = self.get_all_buckets()
        drift = self.get_drift()

        # Overall calibration error
        total_error = 0.0
        total_n = 0
        for stats in self._buckets.values():
            total_error += stats.calibration_error * stats.n
            total_n += stats.n
        avg_error = total_error / max(total_n, 1)

        # Win rate
        total_wins = sum(b.sum_outcome for b in self._buckets.values())
        win_rate = total_wins / max(total_n, 1)

        return {
            "total_trades": self._total_trades,
            "total_pnl_pct": round(self._total_pnl, 2),
            "avg_calibration_error": round(avg_error, 4),
            "win_rate": round(win_rate, 3),
            "buckets": all_buckets,
            "drift": drift,
            "total_samples": total_n,
        }


# Module-level singleton
_learner: Optional[OnlineLearner] = None


def get_online_learner() -> OnlineLearner:
    """Get the global online learner instance."""
    global _learner
    if _learner is None:
        _learner = OnlineLearner()
    return _learner


def reset_online_learner():
    """Reset the global learner (for testing)."""
    global _learner
    _learner = None
