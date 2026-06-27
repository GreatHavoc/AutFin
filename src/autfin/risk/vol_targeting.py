"""
Volatility-targeting overlay.

Scales gross exposure so portfolio vol stays near a target (default 10% annualized).
When realized vol is high, reduces position sizes. When low, allows full sizing.

QuantPedia validated: vol targeting improves Sharpe by 0.2-0.5 on equity portfolios.
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_ANNUAL_TARGET_VOL = 0.10        # 10% annualized
DEFAULT_LOOKBACK_DAYS = 60              # 60 trading days
DEFAULT_MIN_SCALING = 0.25              # Never go below 25% of original size
DEFAULT_MAX_SCALING = 1.50              # Allow up to 150% in low-vol regimes


def compute_realized_vol(
    returns: List[float],
    annualize: bool = True,
) -> Optional[float]:
    """
    Compute realized volatility from a list of daily returns.

    Args:
        returns: Daily log or simple returns (e.g., [0.01, -0.02, 0.005])
        annualize: If True, multiply by sqrt(252)

    Returns:
        Annualized volatility (float), or None if insufficient data
    """
    if not returns or len(returns) < 5:
        return None

    n = len(returns)
    mean_ret = sum(returns) / n
    variance = sum((r - mean_ret) ** 2 for r in returns) / (n - 1)
    std_dev = math.sqrt(variance)

    if annualize:
        std_dev *= math.sqrt(252)

    return std_dev


def compute_vol_scaling(
    realized_vol: float,
    target_vol: float = DEFAULT_ANNUAL_TARGET_VOL,
    min_scaling: float = DEFAULT_MIN_SCALING,
    max_scaling: float = DEFAULT_MAX_SCALING,
) -> float:
    """
    Compute the scaling factor based on realized vs target vol.

    scaling = target_vol / realized_vol, clamped to [min, max]

    Args:
        realized_vol: Current annualized volatility
        target_vol: Desired annualized volatility
        min_scaling: Floor (don't shrink below this)
        max_scaling: Ceiling (don't grow beyond this)

    Returns:
        Scaling factor (float)
    """
    if realized_vol <= 0:
        return max_scaling  # Zero vol = allow full sizing

    raw_scaling = target_vol / realized_vol
    return max(min_scaling, min(max_scaling, raw_scaling))


def vol_target_positions(
    positions: List[Dict[str, Any]],
    returns: List[float],
    target_vol: float = DEFAULT_ANNUAL_TARGET_VOL,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_scaling: float = DEFAULT_MIN_SCALING,
    max_scaling: float = DEFAULT_MAX_SCALING,
) -> Dict[str, Any]:
    """
    Apply vol-targeting to a set of positions.

    Args:
        positions: List of position dicts with at least 'qty' or 'weight' keys
        returns: Daily portfolio returns (most recent `lookback_days`)
        target_vol: Desired annualized vol
        lookback_days: Window for realized vol computation
        min_scaling: Floor scaling factor
        max_scaling: Ceiling scaling factor

    Returns:
        Dict with:
            - scaling: float (the scaling factor applied)
            - realized_vol: float (current annualized vol)
            - target_vol: float (target vol)
            - positions: list of positions with adjusted qty/weight
            - adjustment_pct: float (% change in total exposure)
    """
    if not positions:
        return {
            "scaling": 1.0,
            "realized_vol": 0.0,
            "target_vol": target_vol,
            "positions": [],
            "adjustment_pct": 0.0,
        }

    # Use last N returns for vol estimation
    recent_returns = returns[-lookback_days:] if len(returns) > lookback_days else returns
    realized_vol = compute_realized_vol(recent_returns) or 0.0

    # Compute scaling
    scaling = compute_vol_scaling(realized_vol, target_vol, min_scaling, max_scaling)

    # Apply scaling to positions
    adjusted = []
    total_original = 0.0
    total_adjusted = 0.0

    for pos in positions:
        new_pos = dict(pos)  # shallow copy

        if "qty" in pos:
            original_qty = pos["qty"]
            adjusted_qty = round(original_qty * scaling)
            new_pos["qty"] = max(0, adjusted_qty)
            total_original += abs(original_qty)
            total_adjusted += abs(new_pos["qty"])
        elif "weight" in pos:
            original_weight = pos["weight"]
            adjusted_weight = original_weight * scaling
            new_pos["weight"] = adjusted_weight
            total_original += abs(original_weight)
            total_adjusted += abs(adjusted_weight)

        adjusted.append(new_pos)

    adjustment_pct = (total_adjusted / total_original - 1.0) * 100 if total_original > 0 else 0.0

    logger.info(
        "Vol targeting: realized=%.1f%%, target=%.1f%%, scaling=%.2f, exposure %.1f%%",
        realized_vol * 100, target_vol * 100, scaling, adjustment_pct,
    )

    return {
        "scaling": scaling,
        "realized_vol": realized_vol,
        "target_vol": target_vol,
        "positions": adjusted,
        "adjustment_pct": adjustment_pct,
    }
