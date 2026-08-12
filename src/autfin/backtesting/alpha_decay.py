"""Alpha decay tracker — monitors how quickly strategy alpha degrades over time.

Tracks rolling Sharpe, return, and win-rate across consecutive walk-forward
folds. When alpha decays below configurable thresholds, signals that the
strategy needs re-optimization or retirement.

Usage:
    from src.autfin.backtesting.alpha_decay import AlphaDecayTracker

    tracker = AlphaDecayTracker()
    tracker.add_fold({"fold": 0, "test_sharpe": 1.2, "test_return_pct": 8.5})
    tracker.add_fold({"fold": 1, "test_sharpe": 0.9, "test_return_pct": 5.1})
    report = tracker.report()
    # report.alpha_decayed == True if decay exceeds threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FoldSnapshot:
    """Snapshot of a single walk-forward fold's OOS metrics."""
    fold: int
    test_sharpe: float
    test_return_pct: float
    test_max_drawdown: float = 0.0
    test_win_rate: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    n_bars: int = 0


@dataclass
class DecayReport:
    """Output of AlphaDecayTracker.report()."""
    n_folds: int
    sharpe_trend: float  # linear slope of Sharpe across folds (-1..+1)
    return_trend: float  # linear slope of return across folds
    latest_sharpe: float
    earliest_sharpe: float
    sharpe_drop_pct: float  # % drop from first to last fold
    return_drop_pct: float  # % drop from first to last fold
    avg_sharpe: float
    min_sharpe: float
    alpha_decayed: bool  # True if decay exceeds thresholds
    decay_reason: str    # Human-readable reason if decayed
    recommendation: str


class AlphaDecayTracker:
    """Track alpha decay across walk-forward folds.

    Args:
        sharpe_decay_threshold: Max allowable fractional drop in Sharpe
            (0.5 = 50% drop triggers decay signal)
        min_sharpe: Minimum acceptable absolute Sharpe ratio
        min_folds: Minimum folds before decay can be assessed
    """

    def __init__(
        self,
        sharpe_decay_threshold: float = 0.5,
        min_sharpe: float = 0.3,
        min_folds: int = 3,
    ):
        self.sharpe_decay_threshold = sharpe_decay_threshold
        self.min_sharpe = min_sharpe
        self.min_folds = min_folds
        self._folds: List[FoldSnapshot] = []

    def add_fold(self, fold_data: Dict[str, Any]) -> None:
        """Add a walk-forward fold result.

        Args:
            fold_data: Dict with keys from FoldResult (fold, test_sharpe, etc.)
        """
        self._folds.append(FoldSnapshot(
            fold=fold_data.get("fold", len(self._folds)),
            test_sharpe=fold_data.get("test_sharpe", 0.0),
            test_return_pct=fold_data.get("test_return_pct", 0.0),
            test_max_drawdown=fold_data.get("test_max_drawdown", 0.0),
            test_win_rate=fold_data.get("test_win_rate", 0.0),
            best_params=fold_data.get("best_params", {}),
            n_bars=fold_data.get("n_bars", 0),
        ))

    def report(self) -> DecayReport:
        """Generate alpha decay report from accumulated fold data."""
        if not self._folds:
            return DecayReport(
                n_folds=0, sharpe_trend=0.0, return_trend=0.0,
                latest_sharpe=0.0, earliest_sharpe=0.0,
                sharpe_drop_pct=0.0, return_drop_pct=0.0,
                avg_sharpe=0.0, min_sharpe=0.0,
                alpha_decayed=False, decay_reason="No folds tracked",
                recommendation="Add fold data before assessing decay",
            )

        sharpes = [f.test_sharpe for f in self._folds]
        returns = [f.test_return_pct for f in self._folds]

        # Linear trend via least-squares slope
        n = len(sharpes)
        x_mean = (n - 1) / 2
        sharpe_mean = sum(sharpes) / n
        return_mean = sum(returns) / n

        num_sharpe = sum((i - x_mean) * (s - sharpe_mean) for i, s in enumerate(sharpes))
        den_sharpe = sum((i - x_mean) ** 2 for i in range(n)) or 1
        sharpe_trend = num_sharpe / den_sharpe

        num_return = sum((i - x_mean) * (r - return_mean) for i, r in enumerate(returns))
        den_return = den_sharpe  # same denominator
        return_trend = num_return / den_return

        earliest = sharpes[0]
        latest = sharpes[-1]
        sharpe_drop = ((earliest - latest) / earliest * 100) if earliest != 0 else 0.0

        earliest_ret = returns[0]
        latest_ret = returns[-1]
        return_drop = ((earliest_ret - latest_ret) / abs(earliest_ret) * 100) if earliest_ret != 0 else 0.0

        avg_sharpe = sum(sharpes) / n
        min_sharpe = min(sharpes)

        # Decay detection
        decayed = False
        reasons = []

        if n >= self.min_folds:
            if sharpe_drop > self.sharpe_decay_threshold * 100:
                decayed = True
                reasons.append(f"Sharpe dropped {sharpe_drop:.0f}% (threshold: {self.sharpe_decay_threshold * 100:.0f}%)")

            if min_sharpe < self.min_sharpe:
                decayed = True
                reasons.append(f"Min Sharpe {min_sharpe:.2f} below floor {self.min_sharpe}")

            if sharpe_trend < -0.1:
                reasons.append(f"Negative Sharpe trend: {sharpe_trend:.3f}/fold")

        decay_reason = "; ".join(reasons) if reasons else "No decay detected"

        if decayed:
            recommendation = (
                "Strategy alpha has decayed — re-optimize parameters or "
                "retire strategy. Consider expanding param_grid or adding "
                "new signal sources."
            )
        elif sharpe_trend < -0.05:
            recommendation = (
                "Early decay signal — monitor next fold. If trend continues, "
                "re-optimize."
            )
        else:
            recommendation = "Strategy alpha is stable — continue paper trading"

        return DecayReport(
            n_folds=n,
            sharpe_trend=round(sharpe_trend, 4),
            return_trend=round(return_trend, 4),
            latest_sharpe=round(latest, 3),
            earliest_sharpe=round(earliest, 3),
            sharpe_drop_pct=round(sharpe_drop, 2),
            return_drop_pct=round(return_drop, 2),
            avg_sharpe=round(avg_sharpe, 3),
            min_sharpe=round(min_sharpe, 3),
            alpha_decayed=decayed,
            decay_reason=decay_reason,
            recommendation=recommendation,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decay report to dict."""
        r = self.report()
        return {
            "n_folds": r.n_folds,
            "sharpe_trend": r.sharpe_trend,
            "return_trend": r.return_trend,
            "latest_sharpe": r.latest_sharpe,
            "earliest_sharpe": r.earliest_sharpe,
            "sharpe_drop_pct": r.sharpe_drop_pct,
            "return_drop_pct": r.return_drop_pct,
            "avg_sharpe": r.avg_sharpe,
            "min_sharpe": r.min_sharpe,
            "alpha_decayed": r.alpha_decayed,
            "decay_reason": r.decay_reason,
            "recommendation": r.recommendation,
        }
