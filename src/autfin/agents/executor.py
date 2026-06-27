"""Execution agent for AutFin.

Provides three execution modes:
- recommendation: generate human-readable recommendations, never place orders
- paper: simulate execution with fake order ids
- auto-small: place real orders via self-hosted Kite only when safe

Follows the universal AutFin agent pattern:
- async function signature
- reads from state only
- returns a partial state update dict
- no direct state mutation
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage

from langgraph.config import RunnableConfig

from src.autfin.config.settings import AutFinSettings
from src.autfin.state.agent_state import AgentState
from src.autfin.tools.kite_tools import KiteTools


_REQUIRED_KEYS = {"symbol", "action", "qty", "price", "order_type"}


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _get_available_cash(state: AgentState) -> float:
    return float(state.get("available_cash", 0.0) or 0.0)


def _get_holdings(state: AgentState) -> list[dict]:
    return list(state.get("holdings", []) or [])


def _holding_qty(symbol: str, holdings: list[dict]) -> int:
    """Return current holding quantity for a symbol."""
    for item in holdings:
        item_symbol = (
            item.get("tradingsymbol")
            or item.get("symbol")
            or item.get("trading_symbol")
            or ""
        )
        if item_symbol == symbol:
            qty = item.get("qty") or item.get("quantity") or 0
            return int(float(qty) if qty is not None else 0)
    return 0


def _validate_trade(trade: Any) -> tuple[bool, str]:
    """Validate a trade dict. Returns (is_valid, reason)."""
    if not isinstance(trade, dict):
        return False, "Trade is not a dict"

    missing = _REQUIRED_KEYS - set(trade.keys())
    if missing:
        return False, f"Missing required keys: {sorted(missing)}"

    try:
        qty = int(float(trade["qty"]))
    except (TypeError, ValueError):
        return False, "qty is not a number"
    if qty <= 0:
        return False, f"Invalid qty: {qty}"

    try:
        price = float(trade["price"])
    except (TypeError, ValueError):
        return False, "price is not a number"

    order_type = str(trade.get("order_type", "")).upper()
    if order_type != "MARKET" and price <= 0:
        return False, f"Invalid price for {order_type} order: {price}"

    action = str(trade.get("action", "")).upper()
    if action not in {"BUY", "SELL"}:
        return False, f"Invalid action: {action}"

    return True, ""


def _build_base_report(trade: dict[str, Any], mode: str) -> dict[str, Any]:
    """Initialize an execution report from a trade dict."""
    decision_id = trade.get("decision_id") or str(uuid.uuid4())
    qty = int(float(trade["qty"]))
    price = float(trade["price"])
    trade_value = round(qty * price, 2)
    # SEBI Strategy ID — required by Apr 2026 algo framework
    strategy_id = trade.get("strategy_id") or f"AF-{trade.get('strategy', 'CORE')[:8].upper()}"
    return {
        "decision_id": decision_id,
        "strategy_id": strategy_id,
        "symbol": str(trade["symbol"]),
        "action": str(trade["action"]).upper(),
        "qty": qty,
        "price": price,
        "order_type": str(trade["order_type"]).upper(),
        "trade_value": trade_value,
        "mode": mode,
        "status": "",
        "order_id": None,
        "simulated_order_id": None,
        "message": "",
        "timestamp": _now_iso(),
    }


def _recommendation_message(report: dict[str, Any]) -> str:
    """Generate a human-readable recommendation message."""
    action = report["action"]
    symbol = report["symbol"]
    qty = report["qty"]
    order_type = report["order_type"]
    if order_type == "MARKET":
        return (
            f"Recommend {action} {qty} shares of {symbol} at MARKET price "
            f"(est. value ₹{report['trade_value']:,.2f})."
        )
    return (
        f"Recommend {action} {qty} shares of {symbol} at ₹{report['price']:,.2f} "
        f"({order_type}, value ₹{report['trade_value']:,.2f})."
    )


def _kelly_sized_qty(
    trade: dict[str, Any],
    available_cash: float,
    current_drawdown: float = 0.0,
) -> dict[str, Any]:
    """Recompute trade qty using Kelly criterion if opportunity data is available.

    If the trade dict contains 'win_rate', 'avg_win', 'avg_loss', and 'conviction',
    this function recalculates the optimal position size using Kelly with regime gates.
    Otherwise, returns the trade unchanged.

    Args:
        trade: Trade dict with optional Kelly opportunity data
        available_cash: Available cash for BUY orders
        current_drawdown: Current portfolio drawdown (0.0 = at peak)

    Returns:
        Trade dict with potentially updated qty and kelly metadata
    """
    win_rate = trade.get("win_rate")
    avg_win = trade.get("avg_win")
    avg_loss = trade.get("avg_loss")
    conviction = trade.get("conviction")

    # Only apply Kelly if all opportunity data is present
    if win_rate is None or avg_win is None or avg_loss is None or conviction is None:
        return trade

    try:
        from src.autfin.risk.kelly import kelly_fraction
        from src.autfin.risk.regime import get_regime_detector

        regime = get_regime_detector().current_regime
        kelly = kelly_fraction(
            win_rate=float(win_rate),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            regime=regime.regime,
            drawdown_pct=current_drawdown,
            conviction=float(conviction),
        )

        if kelly["position_pct"] <= 0:
            # Kelly says don't trade — reduce qty to 0
            trade = dict(trade)
            trade["qty"] = 0
            trade["kelly_skip"] = True
            trade["kelly_rationale"] = kelly["rationale"]
            return trade

        price = float(trade.get("price", 0))
        if price <= 0:
            return trade

        # Compute optimal allocation
        alloc_value = available_cash * kelly["position_pct"]
        optimal_qty = max(1, int(alloc_value / price))

        # Only override if Kelly suggests fewer shares (never increase)
        current_qty = int(float(trade.get("qty", 0)))
        if optimal_qty < current_qty:
            trade = dict(trade)
            trade["qty"] = optimal_qty
            trade["kelly_sized"] = True
            trade["kelly_position_pct"] = kelly["position_pct"]
            trade["kelly_rationale"] = kelly["rationale"]

        return trade
    except Exception:
        return trade


async def executor_agent(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """
    Execute or simulate trades from the trade_queue.

    Also handles order modifications and cancellations if requested.

    Args:
        state: Current agent state.
        config: Reserved for future configurable overrides.

    Returns:
        Partial state update with ``execution_reports``, ``messages``, and ``errors``.
    """

    settings = AutFinSettings()
    execution_mode = settings.execution_mode
    auto_execute_max_amount = float(settings.auto_execute_max_amount or 0.0)
    kite_auto_login_configured = bool(settings.kite_user_id and settings.kite_password and settings.kite_totp_secret)

    # Market hours check — NSE: 9:15 AM to 3:30 PM IST, Mon-Fri
    # Skipped in paper/recommendation mode, or when state["skip_market_hours_check"] = True
    from datetime import datetime, timezone, timedelta
    _ist = timezone(timedelta(hours=5, minutes=30))
    _now_ist = datetime.now(_ist)
    _market_open = _now_ist.hour == 9 and _now_ist.minute >= 15 or 10 <= _now_ist.hour <= 14 or _now_ist.hour == 15 and _now_ist.minute <= 30
    _is_weekday = _now_ist.weekday() < 5  # Mon=0..Fri=4
    _market_hours = _market_open and _is_weekday
    _skip_market_check = state.get("skip_market_hours_check", False)

    execution_reports: list[dict[str, Any]] = []
    messages: list[Any] = []
    errors: list[str] = list(state.get("errors", []) or [])
    trade_queue = list(state.get("trade_queue", []) or [])
    order_modifications = list(state.get("order_modifications", []) or [])
    order_cancellations = list(state.get("order_cancellations", []) or [])

    # Create KiteTools once for all operations (reuse connection)
    kite = KiteTools() if kite_auto_login_configured else None

    # Process order modifications
    for mod in order_modifications:
        order_id = mod.get("order_id")
        if not order_id:
            errors.append("Order modification skipped: missing order_id")
            continue

        if not kite_auto_login_configured or not kite:
            errors.append(f"Cannot modify order {order_id}: Kite not configured")
            execution_reports.append({
                "type": "order_modify",
                "order_id": order_id,
                "status": "error",
                "message": "Kite auto-login not configured",
            })
            continue

        try:
            new_qty = mod.get("new_qty")
            new_price = mod.get("new_price")
            result = await kite.modify_order(
                order_id=order_id,
                quantity=int(new_qty) if new_qty else None,
                price=float(new_price) if new_price else None,
            )
            if result.get("status") == "success":
                execution_reports.append({
                    "type": "order_modify",
                    "order_id": order_id,
                    "status": "modified",
                    "message": f"Order {order_id} modified successfully",
                })
                messages.append(AIMessage(content=f"Modified order {order_id}"))
            else:
                msg = result.get("message", "Unknown error")
                execution_reports.append({
                    "type": "order_modify",
                    "order_id": order_id,
                    "status": "error",
                    "message": f"Failed to modify order {order_id}: {msg}",
                })
                errors.append(f"Order modify failed for {order_id}: {msg}")
        except Exception as e:
            execution_reports.append({
                "type": "order_modify",
                "order_id": order_id,
                "status": "error",
                "message": f"Error modifying order {order_id}: {str(e)}",
            })
            errors.append(f"Order modify error for {order_id}: {str(e)}")

    # Process order cancellations
    for cancel in order_cancellations:
        order_id = cancel.get("order_id")
        if not order_id:
            errors.append("Order cancellation skipped: missing order_id")
            continue

        if not kite_auto_login_configured or not kite:
            errors.append(f"Cannot cancel order {order_id}: Kite not configured")
            execution_reports.append({
                "type": "order_cancel",
                "order_id": order_id,
                "status": "error",
                "message": "Kite auto-login not configured",
            })
            continue

        try:
            result = await kite.cancel_order(order_id=order_id)
            if result.get("status") == "success":
                execution_reports.append({
                    "type": "order_cancel",
                    "order_id": order_id,
                    "status": "cancelled",
                    "message": f"Order {order_id} cancelled successfully",
                })
                messages.append(AIMessage(content=f"Cancelled order {order_id}"))
            else:
                msg = result.get("message", "Unknown error")
                execution_reports.append({
                    "type": "order_cancel",
                    "order_id": order_id,
                    "status": "error",
                    "message": f"Failed to cancel order {order_id}: {msg}",
                })
                errors.append(f"Order cancel failed for {order_id}: {msg}")
        except Exception as e:
            execution_reports.append({
                "type": "order_cancel",
                "order_id": order_id,
                "status": "error",
                "message": f"Error cancelling order {order_id}: {str(e)}",
            })
            errors.append(f"Order cancel error for {order_id}: {str(e)}")

    # Process trade queue (existing logic)
    if not trade_queue and not order_modifications and not order_cancellations:
        messages.append(AIMessage(content="Executor: trade queue is empty; nothing to do."))
        return {
            "execution_reports": execution_reports,
            "messages": messages,
            "errors": errors,
        }

    if not trade_queue:
        # Order modifications/cancellations were already processed above
        pass
    else:
        available_cash = _get_available_cash(state)
        holdings = _get_holdings(state)

        for trade in trade_queue:
            valid, reason = _validate_trade(trade)
            if not valid:
                errors.append(f"Executor skipped invalid trade: {reason}")
                continue

            # Apply Kelly sizing if opportunity data is present
            current_drawdown = float(state.get("current_drawdown", 0.0) or 0.0)
            trade = _kelly_sized_qty(trade, available_cash, current_drawdown)

            # Skip trades Kelly says don't take
            if trade.get("kelly_skip"):
                report = _build_base_report(trade, execution_mode)
                report["status"] = "kelly_skipped"
                report["message"] = (
                    f"Kelly criterion says skip: {trade.get('kelly_rationale', 'negative edge')}"
                )
                execution_reports.append(report)
                messages.append(AIMessage(content=f"Executor: {report['message']}"))
                continue

            # Re-validate after Kelly sizing
            if int(float(trade.get("qty", 0))) <= 0:
                continue

            report = _build_base_report(trade, execution_mode)
            action = report["action"]
            symbol = report["symbol"]
            qty = report["qty"]
            trade_value = report["trade_value"]

            # Block real orders outside market hours (paper/recommendation always allowed)
            # Only enforced when state["enforce_market_hours"] is explicitly True
            if execution_mode == "auto-small" and not _market_hours and state.get("enforce_market_hours", False):
                report["status"] = "rejected"
                report["message"] = (
                    f"Order rejected: market closed (NSE hours: 9:15-15:30 IST). "
                    f"Current time: {_now_ist.strftime('%H:%M IST %A')}."
                )
                execution_reports.append(report)
                messages.append(AIMessage(content=f"Executor: {report['message']}"))
                continue

            if execution_mode == "recommendation":
                report["status"] = "recommended"
                report["message"] = _recommendation_message(report)
                execution_reports.append(report)
                messages.append(AIMessage(content=f"Executor recommendation: {report['message']}"))

            elif execution_mode == "paper":
                try:
                    from src.autfin.paper_trading.engine import get_paper_engine

                    paper = get_paper_engine()
                    order = paper.place_order(
                        symbol=symbol,
                        action=action,
                        quantity=qty,
                        order_type=report["order_type"],
                        price=report["price"],
                        product="CNC",
                    )
                    report["status"] = "paper_executed"
                    report["simulated_order_id"] = order.order_id
                    report["filled_price"] = order.filled_price
                    report["message"] = (
                        f"Paper {action} {order.filled_quantity} shares of {symbol} at "
                        f"₹{order.filled_price:,.2f} (order_id={order.order_id})."
                    )
                except Exception as e:
                    report["status"] = "paper_executed"
                    report["simulated_order_id"] = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
                    report["message"] = (
                        f"Paper {action} {qty} shares of {symbol} at "
                        f"₹{report['price']:,.2f} (order_id={report['simulated_order_id']}). "
                        f"Engine unavailable: {e}"
                    )
                execution_reports.append(report)
                messages.append(AIMessage(content=f"Executor paper trade: {report['message']}"))

            elif execution_mode == "auto-small":
                should_execute = True
                downgrade_reason = ""

                if not kite_auto_login_configured:
                    should_execute = False
                    downgrade_reason = "Kite auto-login not configured; downgraded to recommendation."

                if should_execute and trade_value > auto_execute_max_amount:
                    should_execute = False
                    downgrade_reason = (
                        f"Trade value ₹{trade_value:,.2f} exceeds auto-execute limit "
                        f"₹{auto_execute_max_amount:,.2f}; downgraded to recommendation."
                    )

                if should_execute:
                    if action == "BUY" and available_cash < trade_value:
                        should_execute = False
                        downgrade_reason = (
                            f"Insufficient funds: need ₹{trade_value:,.2f}, "
                            f"have ₹{available_cash:,.2f}; downgraded to recommendation."
                        )
                    elif action == "SELL":
                        held_qty = _holding_qty(symbol, holdings)
                        if held_qty < qty:
                            should_execute = False
                            downgrade_reason = (
                                f"Insufficient holdings: need {qty} shares of {symbol}, "
                                f"have {held_qty}; downgraded to recommendation."
                            )

                if should_execute:
                    try:
                        order_price = 0.0 if report["order_type"] == "MARKET" else report["price"]
                        exchanges = ["NSE", "BSE"]
                        last_result: dict[str, Any] | None = None
                        for exchange in exchanges:
                            last_result = await kite.place_order(
                                variety="regular",
                                exchange=exchange,
                                tradingsymbol=symbol,
                                transaction_type=action,
                                quantity=qty,
                                product="CNC",
                                order_type=report["order_type"],
                                price=order_price,
                                tag="AutFin",
                            )
                            if last_result.get("status") == "success":
                                break

                        if last_result and last_result.get("status") == "success":
                            report["status"] = "executed"
                            report["order_id"] = str(last_result.get("order_id"))
                            report["message"] = (
                                f"Executed {action} {qty} shares of {symbol} on {exchange}: "
                                f"order_id={report['order_id']}."
                            )
                            if action == "BUY":
                                available_cash -= trade_value
                        else:
                            msg = (
                                last_result.get("message", "Unknown error")
                                if last_result
                                else "No response from KiteTools"
                            )
                            report["status"] = "error"
                            report["message"] = f"Order placement failed: {msg}"
                            errors.append(f"Executor order failed for {symbol}: {msg}")
                    except Exception as e:
                        report["status"] = "error"
                        report["message"] = f"KiteTools error: {str(e)}"
                        errors.append(f"Executor KiteTools error for {symbol}: {str(e)}")
                else:
                    report["status"] = "recommended"
                    report["message"] = f"{_recommendation_message(report)} {downgrade_reason}"

                execution_reports.append(report)
                messages.append(AIMessage(content=f"Executor auto-small: {report['message']}"))

            else:
                report["status"] = "error"
                report["message"] = f"Unknown execution mode: {execution_mode}"
                execution_reports.append(report)
                errors.append(f"Unknown execution mode: {execution_mode}")

    summary = (
        f"Executor: processed {len(trade_queue)} trade(s), "
        f"{len(order_modifications)} modify(s), {len(order_cancellations)} cancel(s). "
        f"Generated {len(execution_reports)} report(s) in mode '{execution_mode}'."
    )
    messages.append(AIMessage(content=summary))

    return {
        "execution_reports": execution_reports,
        "messages": messages,
        "errors": errors,
    }
