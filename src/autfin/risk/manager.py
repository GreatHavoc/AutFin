"""Trade risk assessment — the orchestrator `risk_agent.assess_trade` calls.

`agents/risk.py` imported `autfin.agents.risk_manager`, which never existed, so
the risk agent's core tool raised `ModuleNotFoundError` on every call. This
module supplies it from the deterministic pieces already in the repo
(`risk/kelly.py`, `risk/vol_targeting.py`), returning the `RiskAssessment` the
agent declares as its output.
"""
from __future__ import annotations

from autfin.types import RiskAssessment

# Conviction below this is not a trade. Mirrors the system reminder on the
# supervisor and the guardrail in agents/guardrails.py.
MIN_CONVICTION = 0.5

# Win-rate and win/loss-size priors used until the learning journal has enough
# completed trades to supply real ones. Deliberately conservative.
DEFAULT_WIN_RATE = 0.45
DEFAULT_AVG_WIN = 0.06
DEFAULT_AVG_LOSS = 0.04


async def assess_risk(
    *,
    symbol: str,
    action: str,
    entry_price: float,
    conviction: float,
    capital: float,
    portfolio: list,
    max_position_pct: float,
    sector_limit_pct: float,
    max_drawdown_pct: float,
    analysis_data: dict | None = None,
) -> RiskAssessment:
    """Size and gate one trade deterministically. No LLM involved."""
    from autfin.risk.kelly import kelly_fraction

    conditions: list[str] = []

    if conviction < MIN_CONVICTION:
        return RiskAssessment(
            symbol=symbol,
            verdict="reject",
            rationale=f"Conviction {conviction:.2f} below minimum {MIN_CONVICTION}",
            conditions=[f"conviction {conviction:.2f} < {MIN_CONVICTION}"],
        )

    # Existing exposure to this symbol and this sector.
    current_symbol_qty = sum(h.qty for h in portfolio if h.tradingsymbol == symbol)
    symbol_value = current_symbol_qty * entry_price
    symbol_pct = symbol_value / capital if capital else 0.0

    sector = (analysis_data or {}).get("sector") or (
        getattr(portfolio[0], "sector", "") if portfolio else ""
    )
    sector_value = sum(
        h.qty * entry_price
        for h in portfolio
        if getattr(h, "sector", "") == sector
    )
    sector_exposure = sector_value / capital if capital else 0.0

    # Kelly on the conservative prior, scaled by conviction.
    kelly = kelly_fraction(
        win_rate=DEFAULT_WIN_RATE,
        avg_win=DEFAULT_AVG_WIN,
        avg_loss=DEFAULT_AVG_LOSS,
        conviction=conviction,
    )
    target_pct = min(kelly["position_pct"], max_position_pct)
    target_value = target_pct * capital

    if symbol_pct >= max_position_pct:
        return RiskAssessment(
            symbol=symbol,
            verdict="reject",
            sector_exposure=round(sector_exposure, 4),
            rationale=(
                f"{symbol} already {symbol_pct:.1%} of capital, "
                f"limit {max_position_pct:.1%}"
            ),
            conditions=["symbol at position limit"],
        )

    if sector_exposure >= sector_limit_pct:
        return RiskAssessment(
            symbol=symbol,
            sector_exposure=round(sector_exposure, 4),
            verdict="conditional",
            rationale=f"{sector or 'sector'} exposure {sector_exposure:.1%} at limit",
            conditions=[f"{sector or 'sector'} at sector limit — reduce existing exposure first"],
        )

    if target_value <= 0:
        return RiskAssessment(
            symbol=symbol,
            verdict="reject",
            rationale="Kelly size is zero at the current prior — no edge to size against",
            conditions=["no positive expectancy at the current prior"],
        )

    position_size = int(target_value / entry_price) if entry_price > 0 else 0
    if position_size <= 0:
        return RiskAssessment(
            symbol=symbol,
            target_value=target_value,
            verdict="reject",
            rationale=f"Sized position is smaller than one share at ₹{entry_price:,.2f}",
        )

    stop_loss = entry_price * (1 - max_drawdown_pct / 2)
    max_loss = (entry_price - stop_loss) * position_size
    target = entry_price * 1.5
    risk_reward = (target - entry_price) / max(entry_price - stop_loss, 1e-9)

    if risk_reward < 2.0:
        conditions.append(f"risk/reward {risk_reward:.2f} below 2.0 — widen the target")
    if max_loss / max(capital, 1) > 0.02:
        conditions.append(f"single-trade loss ₹{max_loss:,.0f} exceeds 2% of capital")

    verdict = "conditional" if conditions else "approve"
    return RiskAssessment(
        symbol=symbol,
        position_size=position_size,
        max_loss=round(max_loss, 2),
        risk_reward=round(risk_reward, 2),
        sector_exposure=round(sector_exposure, 4),
        portfolio_heat=round(symbol_pct + sector_exposure, 4),
        verdict=verdict,
        conditions=conditions,
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        max_position_pct=round(position_size * entry_price / max(capital, 1), 4),
        rationale=(
            f"Kelly-sized at {target_pct:.1%} of ₹{capital:,.0f} "
            f"({kelly['rationale']}); conviction {conviction:.2f}"
        ),
    )
