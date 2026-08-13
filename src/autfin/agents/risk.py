"""AutFin v7.0 — Risk Agent

Position sizing, stop-loss, correlation checks, risk budgeting.
"""
from __future__ import annotations

# ponytail: removed Decimal — NIM xgrammar chokes on its regex pattern
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Thinking

from autfin.agents.base import create_agent, get_model
from autfin.agents.managed_prompts import risk_instructions
from autfin.agents.observability import create_debug_instrumentation
from autfin.tools import risk_toolset, technical_toolset
from autfin.types import (
    AgentDeps,
    RiskAssessment,
    TradeOrder,
    Holding,
)


risk_agent = create_agent(
    name="risk",
    output_type=RiskAssessment,
    model_type="reasoning",
    instructions=risk_instructions.default,
    capabilities=[
        Thinking(effort="high"),
        create_debug_instrumentation(),
    ],
    toolsets=[risk_toolset, technical_toolset],
)


@risk_agent.tool
async def assess_trade(
    ctx: RunContext[AgentDeps],
    symbol: str,
    action: str,
    entry_price: float,
    conviction: float,
    analysis_data: dict | None = None,
    portfolio_holdings: list[Holding] | None = None,
) -> RiskAssessment:
    """Comprehensive risk assessment for a trade."""
    from autfin.risk.kelly import calculate_kelly_size
    from autfin.risk.vol_targeting import calculate_vol_target_size
    from autfin.risk.regime import check_correlation_risk
    from autfin.agents.risk_manager import assess_risk

    deps = ctx.deps
    capital = deps.capital

    # Get current portfolio if not provided
    if portfolio_holdings is None:
        kite = deps.kite_client
        if kite:
            holdings_data = kite.holdings()
            portfolio_holdings = [
                Holding(
                    tradingsymbol=h["tradingsymbol"],
                    qty=h["quantity"],
                    avg_price=h["average_price"],
                    ltp=h["last_price"],
                    pnl=h["pnl"],
                    sector=h.get("sector", ""),
                )
                for h in holdings_data
            ]
        else:
            portfolio_holdings = []

    # Run risk assessment
    assessment = await assess_risk(
        symbol=symbol,
        action=action,
        entry_price=float(entry_price),
        conviction=conviction,
        capital=capital,
        portfolio=portfolio_holdings,
        max_position_pct=deps.max_position_pct,
        sector_limit_pct=deps.sector_limit_pct,
        max_drawdown_pct=deps.max_drawdown_pct,
        analysis_data=analysis_data,
    )

    return assessment


@risk_agent.tool
async def check_position_limits(
    ctx: RunContext[AgentDeps],
    symbol: str,
    proposed_qty: int,
    current_holdings: list[Holding],
) -> dict:
    """Check if proposed position exceeds limits."""
    deps = ctx.deps
    capital = deps.capital

    # Current position
    current_qty = sum(h.qty for h in current_holdings if h.tradingsymbol == symbol)
    new_qty = current_qty + proposed_qty if proposed_qty > 0 else current_qty - abs(proposed_qty)

    # Get LTP for position value
    ltp = next((h.ltp for h in current_holdings if h.tradingsymbol == symbol), None)
    if not ltp:
        return {"allowed": False, "reason": "No price data"}

    position_value = ltp * new_qty
    position_pct = float(position_value / capital)

    # Sector exposure
    sector = next((h.sector for h in current_holdings if h.tradingsymbol == symbol), "")
    sector_value = sum(h.market_value for h in current_holdings if h.sector == sector)
    sector_value += ltp * proposed_qty if proposed_qty > 0 else -ltp * abs(proposed_qty)
    sector_pct = float(sector_value / capital)

    return {
        "allowed": position_pct <= deps.max_position_pct and sector_pct <= deps.sector_limit_pct,
        "position_pct": position_pct,
        "sector_pct": sector_pct,
        "max_position_pct": deps.max_position_pct,
        "max_sector_pct": deps.sector_limit_pct,
        "new_qty": new_qty,
    }


@risk_agent.tool
async def calculate_stop_loss(
    ctx: RunContext[AgentDeps],
    symbol: str,
    entry_price: float,
    action: str,
    atr: float | None = None,
) -> dict:
    """Calculate appropriate stop loss level."""
    from autfin.agents.technical import get_atr

    if atr is None:
        atr = await get_atr(symbol)

    if action == "buy":
        # Long: stop below entry
        atr_stop = entry_price - (2.5 * atr)
        pct_stop = entry_price * 0.92  # 8% max
        stop = max(atr_stop, pct_stop)
    else:
        # Short: stop above entry
        atr_stop = entry_price + (2.5 * atr)
        pct_stop = entry_price * 1.08
        stop = min(atr_stop, pct_stop)

    return {
        "stop_loss": round(stop, 2),
        "atr_stop": round(atr_stop, 2),
        "pct_stop": round(pct_stop, 2),
        "atr": atr,
        "risk_pct": round(abs(entry_price - stop) / entry_price * 100, 2),
    }


@risk_agent.tool
async def calculate_position_size(
    ctx: RunContext[AgentDeps],
    symbol: str,
    entry_price: float,
    stop_loss: float,
    conviction: float,
    capital: float | None = None,
) -> dict:
    """Calculate position size using Kelly + volatility targeting."""
    from autfin.risk.kelly import calculate_kelly_size
    from autfin.risk.vol_targeting import calculate_vol_target_size

    cap = capital or float(ctx.deps.capital)

    # Risk per share
    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share == 0:
        return {"qty": 0, "reason": "Invalid stop loss"}

    # Kelly sizing
    kelly_qty = calculate_kelly_size(
        capital=cap,
        win_rate=0.55 + (conviction - 0.5) * 0.2,  # Adjust by conviction
        risk_reward=2.0,  # Minimum 2:1
        risk_per_share=risk_per_share,
        entry_price=entry_price,
    )

    # Volatility targeting
    vol_qty = calculate_vol_target_size(
        capital=cap,
        symbol=symbol,
        risk_per_share=risk_per_share,
        entry_price=entry_price,
    )

    # Max position limit
    max_qty = int((cap * ctx.deps.max_position_pct) / entry_price)

    # Final size (minimum of all, fractional Kelly)
    final_qty = min(int(kelly_qty * 0.25), vol_qty, max_qty)
    final_qty = max(0, final_qty)

    return {
        "qty": final_qty,
        "kelly_qty": int(kelly_qty),
        "vol_qty": vol_qty,
        "max_qty": max_qty,
        "risk_per_share": risk_per_share,
        "position_value": final_qty * entry_price,
        "position_pct": (final_qty * entry_price) / cap,
    }


@risk_agent.tool
async def check_portfolio_heat(
    ctx: RunContext[AgentDeps],
    portfolio_holdings: list[Holding],
    new_trade: TradeOrder | None = None,
) -> dict:
    """Check total portfolio risk (heat)."""
    total_risk = 0.0
    total_value = sum(h.market_value for h in portfolio_holdings)

    for h in portfolio_holdings:
        # Estimate risk per holding (8% stop loss)
        risk = h.market_value * 0.08
        total_risk += risk

    if new_trade:
        new_risk = float(new_trade.quantity) * float(new_trade.price or 0) * 0.08
        total_risk += new_risk
        total_value += new_risk

    heat_pct = total_risk / total_value if total_value > 0 else 0

    return {
        "total_risk": total_risk,
        "total_value": total_value,
        "heat_pct": heat_pct,
        "max_heat_pct": ctx.deps.max_drawdown_pct,
        "within_limit": heat_pct <= ctx.deps.max_drawdown_pct,
    }