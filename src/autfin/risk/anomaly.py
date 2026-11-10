"""Rolling z-score anomaly detection for AutFin.

Monitors key metrics for drift and anomalies:
- Win rate drops below historical average
- Calibration error increases
- Portfolio drawdown exceeds normal range
- Trade frequency spikes

Uses rolling z-scores with configurable windows and thresholds.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 20
ANOMALY_THRESHOLD = 2.5  # z-score > 2.5 = anomaly


class RollingStats:
    """Track rolling mean and std for a metric."""

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self._values: deque = deque(maxlen=window)

    def add(self, value: float):
        self._values.append(value)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._values) / (len(self._values) - 1)
        return math.sqrt(variance)

    @property
    def latest_z(self) -> float:
        """z-score of the newest value against the window before it.

        check_anomalies used to call z_score(self.mean), which is always ~0:
        every z-score branch was dead and the reported z was fake.
        """
        if len(self._values) < 2:
            return 0.0
        prior = list(self._values)[:-1]
        last = self._values[-1]
        prior_mean = sum(prior) / len(prior)
        denom = sum((v - prior_mean) ** 2 for v in prior)
        if denom <= 0:
            return 0.0
        return (last - prior_mean) / ((denom / (len(prior) - 1)) ** 0.5)

    def z_score(self, value: float) -> float:
        s = self.std
        if s == 0:
            return 0.0
        return (value - self.mean) / s

    @property
    def count(self) -> int:
        return len(self._values)


class AnomalyDetector:
    """Detect anomalies in trading metrics using rolling z-scores."""

    def __init__(self, window: int = DEFAULT_WINDOW, threshold: float = ANOMALY_THRESHOLD):
        self.window = window
        self.threshold = threshold
        self.win_rate = RollingStats(window)
        self.calibration_error = RollingStats(window)
        self.daily_pnl_pct = RollingStats(window)
        self.trade_frequency = RollingStats(window)
        self._anomalies: List[Dict[str, Any]] = []

    def record_trade_result(self, won: bool, pnl_pct: float, calibration_err: float = 0.0):
        """Record a trade result for anomaly monitoring."""
        self.win_rate.add(1.0 if won else 0.0)
        self.daily_pnl_pct.add(pnl_pct)
        self.calibration_error.add(calibration_err)

    def record_run(self, trades_count: int):
        """Record a run for frequency monitoring."""
        self.trade_frequency.add(float(trades_count))

    def check_anomalies(self) -> List[Dict[str, Any]]:
        """Check all metrics for anomalies. Returns list of detected anomalies."""
        anomalies = []

        if self.win_rate.count >= 10:
            # Low win rate is anomalous regardless of z-score
            if self.win_rate.mean < 0.4:
                severity = "high" if self.win_rate.mean < 0.3 else "medium"
                anomalies.append({
                    "metric": "win_rate",
                    "value": round(self.win_rate.mean, 3),
                    "z_score": round(self.win_rate.latest_z, 2),
                    "severity": severity,
                    "message": f"Win rate {self.win_rate.mean:.1%} is below 40% threshold",
                })

        if self.calibration_error.count >= 5:
            z = self.calibration_error.latest_z
            if abs(z) > self.threshold or self.calibration_error.mean > 0.15:
                anomalies.append({
                    "metric": "calibration_error",
                    "value": round(self.calibration_error.mean, 3),
                    "z_score": round(z, 2),
                    "severity": "high" if self.calibration_error.mean > 0.15 else "medium",
                    "message": f"Calibration error {self.calibration_error.mean:.1%} is abnormal (z={z:.2f})",
                })

        if self.daily_pnl_pct.count >= 5:
            z = self.daily_pnl_pct.latest_z
            if z < -self.threshold or self.daily_pnl_pct.mean < -2.0:
                anomalies.append({
                    "metric": "daily_pnl",
                    "value": round(self.daily_pnl_pct.mean, 3),
                    "z_score": round(z, 2),
                    "severity": "high" if self.daily_pnl_pct.mean < -2.0 else "medium",
                    "message": f"Daily P&L {self.daily_pnl_pct.mean:.2f}% is abnormally negative (z={z:.2f})",
                })

        if self.trade_frequency.count >= 5:
            z = self.trade_frequency.latest_z
            if z > self.threshold:
                anomalies.append({
                    "metric": "trade_frequency",
                    "value": round(self.trade_frequency.mean, 1),
                    "z_score": round(z, 2),
                    "severity": "medium",
                    "message": f"Trade frequency {self.trade_frequency.mean:.1f}/run is abnormally high (z={z:.2f})",
                })

        self._anomalies = anomalies
        return anomalies

    def get_summary(self) -> Dict[str, Any]:
        """Get current stats summary."""
        return {
            "win_rate": {"mean": round(self.win_rate.mean, 3), "std": round(self.win_rate.std, 3), "n": self.win_rate.count},
            "calibration_error": {"mean": round(self.calibration_error.mean, 3), "std": round(self.calibration_error.std, 3), "n": self.calibration_error.count},
            "daily_pnl_pct": {"mean": round(self.daily_pnl_pct.mean, 3), "std": round(self.daily_pnl_pct.std, 3), "n": self.daily_pnl_pct.count},
            "trade_frequency": {"mean": round(self.trade_frequency.mean, 1), "std": round(self.trade_frequency.std, 1), "n": self.trade_frequency.count},
            "active_anomalies": len(self._anomalies),
        }
