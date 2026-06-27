"""Market regime detection for AutFin.

Classifies market conditions into discrete regimes using a
volatility-targeting approach (simpler and more robust than HMM for
real-time trading):

  - LOW_VOL (risk_on):  VIX-equivalent < 14 or realized vol < 12%
  - NORMAL:             VIX 14-20 or realized vol 12-20%
  - HIGH_VOL (risk_off): VIX > 20 or realized vol > 20%
  - CRISIS:             VIX > 30 or realized vol > 30% or 5-day drawdown > 5%

The regime is used to:
  1. Gate Kelly criterion sizing (risk_off = half Kelly)
  2. Adjust conviction thresholds (risk_off = higher bar for entry)
  3. Trigger de-grossing (crisis = reduce all positions)

Fallback: if VIX data unavailable, uses Nifty 50 realized volatility.
All arithmetic is deterministic — no LLM involved.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regime thresholds
LOW_VOL_THRESHOLD = 14.0
HIGH_VOL_THRESHOLD = 20.0
CRISIS_VOL_THRESHOLD = 30.0
CRISIS_DRAWDOWN_THRESHOLD = -5.0  # 5% drawdown in 5 days

# Regime-to-Kelly multiplier
REGIME_KELLY_MULT = {
    "low_vol": 1.0,
    "normal": 0.75,
    "high_vol": 0.50,
    "crisis": 0.25,
}

# Regime-to-conviction threshold
REGIME_CONV_THRESHOLD = {
    "low_vol": 0.50,
    "normal": 0.55,
    "high_vol": 0.65,
    "crisis": 0.75,
}


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    regime: str
    vix: float
    realized_vol: float
    nifty_return_5d: float
    updated_at: str
    confidence: float  # 0.0-1.0, how confident in this regime

    @property
    def kelly_mult(self) -> float:
        return REGIME_KELLY_MULT.get(self.regime, 0.75)

    @property
    def conviction_threshold(self) -> float:
        return REGIME_CONV_THRESHOLD.get(self.regime, 0.55)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "vix": self.vix,
            "realized_vol": self.realized_vol,
            "nifty_return_5d": self.nifty_return_5d,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "kelly_mult": self.kelly_mult,
            "conviction_threshold": self.conviction_threshold,
        }


class RegimeDetector:
    """Detect market regime using VIX and realized volatility.

    Maintains a rolling window of daily returns for realized vol
    calculation. Uses VIX as primary signal, falls back to realized
    vol when VIX is unavailable.
    """

    def __init__(self, vol_window: int = 20):
        self.vol_window = vol_window
        self._returns: deque = deque(maxlen=vol_window + 5)
        self._last_regime: Optional[RegimeState] = None
        self._regime_history: List[Dict[str, Any]] = []

    def record_daily_return(self, daily_return: float):
        """Record a daily return for realized vol calculation."""
        self._returns.append(daily_return)

    def _realized_vol(self) -> float:
        """Annualized realized volatility from rolling returns."""
        if len(self._returns) < 5:
            return 0.0
        returns = list(self._returns)
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = math.sqrt(var)
        # Annualize: sqrt(252)
        return daily_vol * math.sqrt(252) * 100  # as percentage

    def _nifty_return_5d(self) -> float:
        """5-day cumulative return from recent returns."""
        if len(self._returns) < 5:
            return 0.0
        recent = list(self._returns)[-5:]
        cum = 1.0
        for r in recent:
            cum *= (1 + r)
        return (cum - 1) * 100

    def detect(
        self,
        vix: Optional[float] = None,
        nifty_close_today: Optional[float] = None,
        nifty_close_5d_ago: Optional[float] = None,
    ) -> RegimeState:
        """Detect current market regime.

        Args:
            vix: Current India VIX value (if available)
            nifty_close_today: Latest Nifty 50 close
            nifty_close_5d_ago: Nifty 50 close 5 days ago

        Returns:
            RegimeState with regime classification
        """
        # Primary: use VIX if available
        vol_signal = vix if vix and vix > 0 else self._realized_vol()

        # 5-day drawdown
        if nifty_close_today and nifty_close_5d_ago and nifty_close_5d_ago > 0:
            nifty_return_5d = ((nifty_close_today - nifty_close_5d_ago) / nifty_close_5d_ago) * 100
        else:
            nifty_return_5d = self._nifty_return_5d()

        # Classify regime
        if vol_signal >= CRISIS_VOL_THRESHOLD or nifty_return_5d <= CRISIS_DRAWDOWN_THRESHOLD:
            regime = "crisis"
            confidence = 0.9
        elif vol_signal >= HIGH_VOL_THRESHOLD:
            regime = "high_vol"
            confidence = 0.8
        elif vol_signal <= LOW_VOL_THRESHOLD:
            regime = "low_vol"
            confidence = 0.8
        else:
            regime = "normal"
            confidence = 0.7

        # Adjust confidence if we have limited data
        if len(self._returns) < self.vol_window:
            confidence *= 0.8

        now = datetime.now(timezone.utc).isoformat()
        state = RegimeState(
            regime=regime,
            vix=round(vix if vix else 0.0, 2),
            realized_vol=round(self._realized_vol(), 2),
            nifty_return_5d=round(nifty_return_5d, 2),
            updated_at=now,
            confidence=round(confidence, 2),
        )

        # Track regime changes
        if self._last_regime and self._last_regime.regime != regime:
            self._regime_history.append({
                "from": self._last_regime.regime,
                "to": regime,
                "vix": state.vix,
                "realized_vol": state.realized_vol,
                "timestamp": now,
            })
            logger.info(
                "Regime change: %s -> %s (VIX=%.1f, vol=%.1f%%)",
                self._last_regime.regime, regime, state.vix, state.realized_vol,
            )

        self._last_regime = state
        return state

    @property
    def current_regime(self) -> RegimeState:
        """Get current regime (or default to normal)."""
        if self._last_regime:
            return self._last_regime
        return RegimeState(
            regime="normal",
            vix=0.0,
            realized_vol=0.0,
            nifty_return_5d=0.0,
            updated_at=datetime.now(timezone.utc).isoformat(),
            confidence=0.5,
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get regime change history."""
        return list(self._regime_history)

    def should_de_gross(self) -> bool:
        """Check if current regime triggers de-grossing."""
        return self.current_regime.regime == "crisis"


# Module-level singleton
_detector: Optional[RegimeDetector] = None


def get_regime_detector() -> RegimeDetector:
    """Get the global regime detector instance."""
    global _detector
    if _detector is None:
        _detector = RegimeDetector()
    return _detector


def reset_regime_detector():
    """Reset the global detector (for testing)."""
    global _detector
    _detector = None
