"""
Portfolio Manager Agent - Capital allocation and deployment pacing.

Follows the universal AutFin agent pattern:
- async function signature
- reads from state only
- returns a partial state update dict
- no direct state mutation
"""

import logging
import sqlite3
import uuid
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.autfin.config.settings import AutFinSettings
from src.autfin.state.agent_state import AgentState

logger = logging.getLogger(__name__)

# Database path (same as memory/database.py)
_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"


# Minimal sector mapping for common Indian stocks. Prefer metadata from
# holdings/positions when available; this map is a fallback.
_SECTOR_MAP: dict[str, str] = {
    "RELIANCE": "Energy",
    "TCS": "IT",
    "INFY": "IT",
    "HCLTECH": "IT",
    "WIPRO": "IT",
    "TECHM": "IT",
    "HDFCBANK": "Bank",
    "ICICIBANK": "Bank",
    "SBIN": "Bank",
    "KOTAKBANK": "Bank",
    "AXISBANK": "Bank",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",
    "LARSEN": "Infrastructure",
    "SUNPHARMA": "Pharma",
    "DRREDDY": "Pharma",
    "TATAMOTORS": "Auto",
    "MARUTI": "Auto",
    "M&M": "Auto",
    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "POWERGRID": "Energy",
    "NTPC": "Energy",
    "COALINDIA": "Energy",
    "BAJFINANCE": "Finance",
    "HDFC": "Finance",
    "ASIANPAINT": "Consumer",
    "ADANIENT": "Conglomerate",
    "ULTRACEMCO": "Cement",
    "SHREECEM": "Cement",
    "ACC": "Cement",
    "AMBUJACEM": "Cement",
}


def _symbol_from_item(item: dict) -> str:
    """Return the trading symbol from a holding/position dict."""
    return (
        item.get("tradingsymbol")
        or item.get("symbol")
        or item.get("trading_symbol")
        or ""
    )


def _symbol_sector(symbol: str, holdings: list[dict], positions: list[dict]) -> str:
    """Resolve sector for a symbol using metadata or the fallback map."""
    for item in holdings + positions:
        if _symbol_from_item(item) == symbol:
            sector = item.get("sector")
            if sector:
                return str(sector)
    return _SECTOR_MAP.get(symbol, "Other")


def _item_value(item: dict) -> float:
    """Market value of a holding/position item."""
    qty = float(item.get("qty", 0) or 0)
    price = float(
        item.get("ltp")
        or item.get("avg_price")
        or item.get("close")
        or item.get("price")
        or 0
    )
    return qty * price


def _holding_qty(symbol: str, holdings: list[dict]) -> int:
    """Return current holding quantity for a symbol."""
    for item in holdings:
        if _symbol_from_item(item) == symbol:
            return int(float(item.get("qty", 0) or 0))
    return 0


def _ltp_for_symbol(symbol: str, state: AgentState) -> float:
    """Best-effort last traded price for a symbol."""
    market_snapshot = state.get("market_snapshot") or {}
    quotes = market_snapshot.get("quotes", {})
    quote = quotes.get(symbol, {})
    if quote:
        return float(quote.get("ltp", 0) or 0)

    for item in state.get("holdings", []):
        if _symbol_from_item(item) == symbol:
            return float(
                item.get("ltp", 0)
                or item.get("avg_price", 0)
                or item.get("price", 0)
                or 0
            )
    return 0.0


def _current_sector_value(
    sector: str, holdings: list[dict], positions: list[dict]
) -> float:
    """Total market value currently deployed in a sector."""
    total = 0.0
    for item in holdings + positions:
        item_symbol = _symbol_from_item(item)
        if _symbol_sector(item_symbol, holdings, positions) == sector:
            total += _item_value(item)
    return total


def _detect_top_up(current_cash: float, settings: AutFinSettings) -> tuple[bool, float]:
    """
    Detect if user has added money since last snapshot.

    Returns:
        (is_top_up, previous_cash) - True if top-up detected, previous cash balance.
    """
    if not _DB_PATH.exists():
        return False, 0.0

    try:
        conn = sqlite3.connect(_DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT available_cash FROM balance_history ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if not row:
                return False, 0.0

            previous_cash = float(row[0] or 0.0)
            increase = current_cash - previous_cash

            # Detect top-up: cash increased by more than min_trade_amount
            if increase >= settings.min_trade_amount:
                logger.info(
                    "Top-up detected: cash increased from ₹%.2f to ₹%.2f (Δ₹%.2f)",
                    previous_cash, current_cash, increase
                )
                return True, previous_cash

            return False, previous_cash
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to check balance history for top-up detection: %s", e)
        return False, 0.0


def _get_holding_period(symbol: str) -> tuple[int, str]:
    """
    Get holding period for a symbol from trade history.

    Returns:
        (days_held, tax_class) where tax_class is 'STCG' or 'LTCG'.
    """
    if not _DB_PATH.exists():
        return 0, "STCG"

    try:
        conn = sqlite3.connect(_DB_PATH)
        try:
            cursor = conn.cursor()
            # Find earliest BUY trade for this symbol — check both trades and paper_trades tables
            entry_date_str = None
            for table, ts_col in [("trades", "timestamp"), ("paper_trades", "filled_at")]:
                try:
                    cursor.execute(
                        f"""
                        SELECT {ts_col} FROM {table}
                        WHERE symbol = ? AND action = 'BUY'
                        ORDER BY {ts_col} ASC LIMIT 1
                        """,
                        (symbol,),
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        entry_date_str = row[0]
                        break
                except Exception:
                    continue

            if not entry_date_str:
                return 0, "STCG"

            # Parse ISO format timestamp
            entry_date = datetime.fromisoformat(entry_date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_held = (now - entry_date).days

            # Indian equity tax classification (FY 2026-27):
            # STCG (Sec 111A): held < 12 months (taxed at 20%)
            # LTCG (Sec 112A): held >= 12 months (exempt up to ₹1.25 lakh/year, then 12.5%)
            tax_class = "LTCG" if days_held >= 365 else "STCG"
            return days_held, tax_class
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to get holding period for %s: %s", symbol, e)
        return 0, "STCG"


def _trading_days_elapsed() -> int:
    """Approximate trading days elapsed in current month (weekdays only)."""
    today = date.today()
    start = today.replace(day=1)
    days = 0
    current = start
    while current <= today:
        if current.weekday() < 5:  # Monday-Friday
            days += 1
        current = date.fromordinal(current.toordinal() + 1)
    return max(1, days)


def _total_trading_days_in_month() -> int:
    """Approximate total trading days in current month (weekdays only)."""
    today = date.today()
    if today.month == 12:
        next_month = today.replace(year=today.year + 1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month + 1, day=1)

    start = today.replace(day=1)
    days = 0
    current = start
    while current < next_month:
        if current.weekday() < 5:
            days += 1
        current = date.fromordinal(current.toordinal() + 1)
    return max(1, days)


def _extract_signals(
    research_debates: Any,
    risk_assessments: Any,
    macro_signals: Any,
    settings: AutFinSettings,
) -> list[dict[str, Any]]:
    """
    Extract actionable signals per symbol from research debates and risk assessments.

    Handles both per-symbol debates (``{symbol: {recommendation, conviction}}``)
    and the aggregate debate format produced by the researcher agent.
    """
    signals: list[dict[str, Any]] = []

    # Normalize debates to a per-symbol mapping.
    per_symbol_debates: dict[str, dict] = {}
    if isinstance(research_debates, dict):
        # Aggregate format: {summary, bull, bear, neutral}
        if "summary" in research_debates and "bull" in research_debates:
            summary = research_debates.get("summary", {})
            conviction = float(summary.get("conviction_score", 0.0) or 0.0)
            recommendation = summary.get("recommendation", "hold")
            # Attach the summary recommendation to every symbol in risk assessments.
            if isinstance(risk_assessments, dict):
                for symbol in risk_assessments.keys():
                    per_symbol_debates[symbol] = {
                        "recommendation": recommendation,
                        "conviction": conviction,
                    }
        else:
            per_symbol_debates = {
                k: v for k, v in research_debates.items() if isinstance(v, dict)
            }

    if not isinstance(risk_assessments, dict):
        risk_assessments = {}

    for symbol, debate in per_symbol_debates.items():
        recommendation = debate.get("recommendation", "hold")
        conviction = float(
            debate.get("conviction", debate.get("conviction_score", 0.0)) or 0.0
        )
        risk = risk_assessments.get(symbol, {})
        if isinstance(risk, dict):
            approved = risk.get(
                "approved",
                risk.get("risk_approved", not risk.get("should_veto", False)),
            )
        else:
            approved = False

        action = "HOLD"
        if isinstance(recommendation, str):
            rec = recommendation.upper().replace("STRONG ", "")
            if rec == "BUY":
                action = "BUY"
            elif rec in ("SELL", "REDUCE"):
                action = "SELL"

        signals.append(
            {
                "symbol": symbol,
                "action": action,
                "conviction": conviction,
                "risk_approved": bool(approved),
                "debate": debate,
                "risk": risk if isinstance(risk, dict) else {},
            }
        )

    # Surface explicit SELL triggers from risk assessments even when no debate exists.
    for symbol, risk in risk_assessments.items():
        if not isinstance(risk, dict):
            continue
        if symbol in per_symbol_debates:
            continue
        reason = risk.get("reason", "")
        rec = risk.get("recommendation", "hold")
        if (
            risk.get("sell_signal")
            or rec in ("sell", "SELL", "REDUCE")
            or "stop" in str(reason).lower()
        ):
            signals.append(
                {
                    "symbol": symbol,
                    "action": "SELL",
                    "conviction": float(risk.get("conviction", 0.8) or 0.8),
                    "risk_approved": bool(
                        risk.get("approved", not risk.get("should_veto", False))
                    ),
                    "debate": {},
                    "risk": risk,
                }
            )

    return signals


async def portfolio_manager_agent(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """
    Capital allocation and deployment pacing agent.

    Reads research debates and risk assessments, then builds a prioritized
    trade queue respecting cash buffer, position limits, sector exposure,
    deployment pacing, and max daily trades.

    Args:
        state: Current agent state.
        config: Optional LangGraph config (unused but kept for pattern compatibility).

    Returns:
        Partial state update with ``trade_queue`` and ``messages``.
    """
    _ = config  # Reserved for future use (e.g. configurable overrides).
    settings = AutFinSettings()
    errors: list[str] = list(state.get("errors", []) or [])
    messages: list[Any] = []

    available_cash = float(state.get("available_cash", 0.0) or 0.0)
    holdings = list(state.get("holdings", []) or [])
    positions = list(state.get("positions", []) or [])
    research_debates = state.get("research_debates")
    risk_assessments = state.get("risk_assessments")
    macro_signals = state.get("macro_signals")

    try:
        # Detect top-up: compare current cash with last snapshot
        is_top_up, previous_cash = _detect_top_up(available_cash, settings)
        if is_top_up:
            top_up_amount = available_cash - previous_cash
            messages.append(
                AIMessage(
                    content=(
                        f"Top-up detected: ₹{top_up_amount:,.2f} added. "
                        f"Cash increased from ₹{previous_cash:,.2f} to ₹{available_cash:,.2f}. "
                        f"Deployment pacing reset."
                    )
                )
            )
            errors.append(f"TOP_UP: ₹{top_up_amount:,.2f} added to account")

        deployable_cash = available_cash * (1.0 - settings.cash_buffer_pct)
        if deployable_cash < settings.min_trade_amount:
            deployable_cash = 0.0

        portfolio_value = available_cash + sum(_item_value(item) for item in holdings)

        # Deployment pacing: guideline only.
        # If top-up detected, treat as fresh month (reset pacing).
        trading_days_elapsed = _trading_days_elapsed() if not is_top_up else 1
        total_trading_days = _total_trading_days_in_month()
        ideal_deployed = deployable_cash * (trading_days_elapsed / total_trading_days)
        already_deployed = portfolio_value - available_cash
        pace_ratio = (
            already_deployed / ideal_deployed if ideal_deployed > 0 else 1.0
        )

        base_conviction_threshold = 0.6
        if pace_ratio > 1.0:
            extra = min(0.25, (pace_ratio - 1.0) * 0.15)
            conviction_threshold = base_conviction_threshold + extra
        else:
            conviction_threshold = base_conviction_threshold

        # Macro adjustment: negative macro backdrop raises the bar.
        if isinstance(macro_signals, dict):
            # Accept both key names from different macro signal sources
            impact_str = (
                macro_signals.get("overall_sentiment")
                or macro_signals.get("impact_score")
                or "neutral"
            )
            macro_score_map = {"high_positive": 1.0, "medium_positive": 0.5, "neutral": 0.0, "medium_negative": -0.5, "high_negative": -1.0}
            macro_score = macro_score_map.get(impact_str, 0.0)
            if macro_score < -0.3:
                conviction_threshold += 0.05
            elif macro_score > 0.3:
                conviction_threshold -= 0.05

        signals = _extract_signals(
            research_debates, risk_assessments, macro_signals, settings
        )

        buy_signals = sorted(
            [s for s in signals if s["action"] == "BUY"],
            key=lambda x: x["conviction"],
            reverse=True,
        )
        sell_signals = [s for s in signals if s["action"] == "SELL"]

        trade_queue: list[dict[str, Any]] = []
        remaining_deployable = deployable_cash
        daily_trades = 0
        max_daily_trades = settings.max_daily_trades

        # Process SELL signals first (free cash, do not count against pacing).
        for signal in sell_signals:
            if daily_trades >= max_daily_trades:
                messages.append(
                    AIMessage(content="Max daily trades reached; remaining SELLs skipped.")
                )
                break

            symbol = signal["symbol"]
            qty = _holding_qty(symbol, holdings)
            if qty <= 0:
                continue

            price = _ltp_for_symbol(symbol, state)
            if price <= 0:
                price = float(
                    signal["risk"].get("price", 0)
                    or signal["debate"].get("price", 0)
                    or 0
                )
            if price <= 0:
                errors.append(f"Cannot price SELL for {symbol}; skipped.")
                continue

            position_size = qty * price

            # Tax tracking: compute holding period and classify
            days_held, tax_class = _get_holding_period(symbol)
            tax_note = (
                f" | Tax: {tax_class} (held {days_held} days)"
                if days_held > 0
                else ""
            )

            trade_queue.append(
                {
                    "symbol": symbol,
                    "action": "SELL",
                    "qty": qty,
                    "price": round(price, 2),
                    "position_size": round(position_size, 2),
                    "order_type": "MARKET",
                    "conviction": round(signal["conviction"], 4),
                    "rationale": (
                        f"SELL signal for {symbol}: "
                        f"risk_approved={signal['risk_approved']}, "
                        f"reason={signal['risk'].get('reason', 'thesis/risk trigger')}"
                        f"{tax_note}"
                    ),
                    "risk_approved": signal["risk_approved"],
                    "decision_id": str(uuid.uuid4()),
                    "days_held": days_held,
                    "tax_class": tax_class,
                }
            )
            daily_trades += 1

        # Process BUY signals in conviction order.
        for signal in buy_signals:
            if daily_trades >= max_daily_trades:
                messages.append(
                    AIMessage(content="Max daily trades reached; remaining BUYs skipped.")
                )
                break

            if not signal["risk_approved"]:
                continue

            if signal["conviction"] < conviction_threshold:
                continue

            if remaining_deployable < settings.min_trade_amount:
                messages.append(
                    AIMessage(content="Deployable cash exhausted; remaining BUYs skipped.")
                )
                break

            symbol = signal["symbol"]
            sector = _symbol_sector(symbol, holdings, positions)

            price = _ltp_for_symbol(symbol, state)
            if price <= 0:
                price = float(
                    signal["debate"].get("price", 0)
                    or signal["risk"].get("price", 0)
                    or 0
                )
            if price <= 0:
                errors.append(f"Cannot price BUY for {symbol}; skipped.")
                continue

            max_position_size = deployable_cash * settings.max_single_position_pct
            position_size = min(max_position_size, remaining_deployable)

            # Sector exposure check.
            sector_value = _current_sector_value(sector, holdings, positions)
            new_sector_value = sector_value + position_size
            max_sector_value = portfolio_value * settings.max_sector_exposure_pct
            if new_sector_value > max_sector_value:
                allowed = max(0.0, max_sector_value - sector_value)
                position_size = min(position_size, allowed)

            if position_size < settings.min_trade_amount:
                messages.append(
                    AIMessage(
                        content=f"BUY for {symbol} below min trade amount after sector check; skipped."
                    )
                )
                continue

            qty = int(position_size // price)
            if qty <= 0:
                errors.append(
                    f"Calculated zero quantity for {symbol} at price {price}; skipping."
                )
                continue

            actual_position_size = qty * price
            trade_queue.append(
                {
                    "symbol": symbol,
                    "action": "BUY",
                    "qty": qty,
                    "price": round(price, 2),
                    "position_size": round(actual_position_size, 2),
                    "order_type": "MARKET",
                    "conviction": round(signal["conviction"], 4),
                    "rationale": (
                        f"BUY {symbol}: conviction={signal['conviction']:.2f}, "
                        f"sector={sector}, pace_ratio={pace_ratio:.2f}"
                    ),
                    "risk_approved": True,
                    "decision_id": str(uuid.uuid4()),
                }
            )
            remaining_deployable -= actual_position_size
            daily_trades += 1

        buys = sum(1 for t in trade_queue if t["action"] == "BUY")
        sells = sum(1 for t in trade_queue if t["action"] == "SELL")
        top_up_note = f" | Top-up: ₹{available_cash - previous_cash:,.2f}" if is_top_up else ""
        summary = (
            f"Portfolio Manager: {buys} BUYs, {sells} SELLs queued. "
            f"Deployable=₹{deployable_cash:,.0f}, remaining=₹{remaining_deployable:,.0f}."
            f"{top_up_note}"
        )
        messages.append(AIMessage(content=summary))

        return {
            "trade_queue": trade_queue,
            "messages": messages,
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Portfolio manager failed: {str(e)}")
        return {
            "trade_queue": [],
            "messages": messages,
            "errors": errors,
        }
