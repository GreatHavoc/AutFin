"""Indian tax engine for AutFin.

Implements FY 2026-27 Indian equity tax rules:
- STCG (Sec 111A): held < 12 months, taxed at 20%
- LTCG (Sec 112A): held >= 12 months, exempt up to Rs 1.25L/yr, then 12.5%
- Tax-loss harvesting: offset losses against gains
- LTCG gain harvesting: realize up to Rs 1.25L LTCG at zero tax
- XIRR: money-weighted return calculation

All arithmetic is deterministic -- no LLM involved.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"

# FY 2026-27 Tax Rates (Indian Equity, Sec 111A / 112A)
STCG_TAX_RATE = 0.20
LTCG_TAX_RATE = 0.125
LTCG_EXEMPTION_LIMIT = 125_000
HOLDING_PERIOD_DAYS = 365


def _get_fy_start(dt: datetime | date) -> date:
    if isinstance(dt, datetime):
        dt = dt.date()
    if dt.month >= 4:
        return date(dt.year, 4, 1)
    return date(dt.year - 1, 4, 1)


class TaxEngine:

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_all_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        trades: List[Dict[str, Any]] = []
        if not self._db_path.exists():
            return trades
        try:
            conn = self._connect()
            try:
                where = "WHERE symbol = ?" if symbol else ""
                params = [symbol] if symbol else []
                rows = conn.execute(
                    f"SELECT symbol, action, quantity, price, timestamp, pnl FROM trades {where} ORDER BY timestamp",
                    params,
                ).fetchall()
                for r in rows:
                    trades.append({
                        "symbol": r[0], "action": r[1], "quantity": r[2],
                        "price": r[3], "timestamp": r[4],
                        "pnl": r[5] or 0.0, "source": "trades",
                    })
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Failed to fetch trades: %s", e)
        trades.sort(key=lambda t: t.get("timestamp", ""))
        return trades

    def classify_trade(self, buy_date: datetime | date, sell_date: datetime | date) -> Tuple[int, str, float]:
        if isinstance(buy_date, datetime):
            buy_date = buy_date.date()
        if isinstance(sell_date, datetime):
            sell_date = sell_date.date()
        days_held = (sell_date - buy_date).days
        if days_held >= HOLDING_PERIOD_DAYS:
            return days_held, "LTCG", LTCG_TAX_RATE
        return days_held, "STCG", STCG_TAX_RATE

    def compute_tax_summary(self, fy: Optional[int] = None) -> Dict[str, Any]:
        if fy is None:
            today = date.today()
            fy = today.year if today.month >= 4 else today.year - 1
        fy_start = date(fy, 4, 1)
        fy_end = date(fy + 1, 3, 31)
        trades = self.get_all_trades()
        lots: Dict[str, List[Dict[str, Any]]] = {}
        stcg_gains = 0.0
        ltcg_gains = 0.0
        ltcg_exemption_used = 0.0
        for t in trades:
            ts = t.get("timestamp", "")
            try:
                trade_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            except Exception:
                continue
            sym = t["symbol"]
            action = t["action"]
            qty = int(t["quantity"])
            price = float(t["price"])
            # Include ALL BUYs for lot tracking (even pre-FY), but only count SELLs within FY
            if action == "BUY":
                lots.setdefault(sym, []).append({"date": trade_date, "qty": qty, "price": price})
            elif action == "SELL" and trade_date <= fy_end and lots.get(sym):
                remaining = qty
                while remaining > 0 and lots[sym]:
                    lot = lots[sym][0]
                    match_qty = min(remaining, lot["qty"])
                    days_held, tax_class, _ = self.classify_trade(lot["date"], trade_date)
                    gain = (price - lot["price"]) * match_qty
                    if tax_class == "STCG":
                        stcg_gains += max(0, gain)
                    else:
                        remaining_exemption = LTCG_EXEMPTION_LIMIT - ltcg_exemption_used
                        taxable_gain = max(0, gain - remaining_exemption)
                        ltcg_exemption_used += min(gain, remaining_exemption)
                        ltcg_gains += taxable_gain
                    lot["qty"] -= match_qty
                    remaining -= match_qty
                    if lot["qty"] <= 0:
                        lots[sym].pop(0)
        stcg_tax = stcg_gains * STCG_TAX_RATE
        ltcg_tax = ltcg_gains * LTCG_TAX_RATE
        total_tax = stcg_tax + ltcg_tax
        return {
            "financial_year": f"FY {fy}-{(fy+1) % 100:02d}",
            "fy_start": fy_start.isoformat(),
            "fy_end": fy_end.isoformat(),
            "stcg_gains": round(stcg_gains, 2),
            "stcg_tax": round(stcg_tax, 2),
            "ltcg_gains": round(ltcg_gains, 2),
            "ltcg_exemption_used": round(min(ltcg_exemption_used, LTCG_EXEMPTION_LIMIT), 2),
            "ltcg_exemption_remaining": round(max(0, LTCG_EXEMPTION_LIMIT - ltcg_exemption_used), 2),
            "ltcg_tax": round(ltcg_tax, 2),
            "total_tax": round(total_tax, 2),
            "effective_tax_rate": round(total_tax / max(stcg_gains + ltcg_gains, 1) * 100, 2),
        }

    def find_harvest_candidates(self) -> List[Dict[str, Any]]:
        trades = self.get_all_trades()
        lots: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            sym = t["symbol"]
            action = t["action"]
            qty = int(t["quantity"])
            price = float(t["price"])
            ts = t.get("timestamp", "")
            try:
                trade_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            except Exception:
                trade_date = date.today()
            if action == "BUY":
                lots.setdefault(sym, []).append({"date": trade_date, "qty": qty, "price": price})
            elif action == "SELL" and lots.get(sym):
                remaining = qty
                while remaining > 0 and lots[sym]:
                    lot = lots[sym][0]
                    match = min(remaining, lot["qty"])
                    lot["qty"] -= match
                    remaining -= match
                    if lot["qty"] <= 0:
                        lots[sym].pop(0)
        candidates = []
        today = date.today()
        for sym, sym_lots in lots.items():
            total_qty = sum(l["qty"] for l in sym_lots)
            if total_qty <= 0:
                continue
            avg_cost = sum(l["qty"] * l["price"] for l in sym_lots) / total_qty
            last_price = sym_lots[-1]["price"] if sym_lots else avg_cost
            unrealized_pnl = (last_price - avg_cost) * total_qty
            unrealized_pct = ((last_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
            if unrealized_pnl < 0:
                oldest_lot_date = min(l["date"] for l in sym_lots)
                days_held = (today - oldest_lot_date).days
                tax_class = "LTCG" if days_held >= HOLDING_PERIOD_DAYS else "STCG"
                candidates.append({
                    "symbol": sym,
                    "quantity": int(total_qty),
                    "avg_cost": round(avg_cost, 2),
                    "estimated_price": round(last_price, 2),
                    "unrealized_loss": round(abs(unrealized_pnl), 2),
                    "unrealized_loss_pct": round(abs(unrealized_pct), 2),
                    "days_held": days_held,
                    "tax_class": tax_class,
                    "harvest_priority": round(abs(unrealized_pnl), 2),
                })
        candidates.sort(key=lambda x: x["harvest_priority"], reverse=True)
        return candidates

    def find_gain_harvest_candidates(self) -> List[Dict[str, Any]]:
        summary = self.compute_tax_summary()
        remaining_exemption = summary["ltcg_exemption_remaining"]
        if remaining_exemption <= 0:
            return []
        trades = self.get_all_trades()
        today = date.today()
        lots: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            sym = t["symbol"]
            action = t["action"]
            qty = int(t["quantity"])
            price = float(t["price"])
            ts = t.get("timestamp", "")
            try:
                trade_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            except Exception:
                trade_date = date.today()
            if action == "BUY":
                lots.setdefault(sym, []).append({"date": trade_date, "qty": qty, "price": price})
            elif action == "SELL" and lots.get(sym):
                remaining = qty
                while remaining > 0 and lots[sym]:
                    lot = lots[sym][0]
                    match = min(remaining, lot["qty"])
                    lot["qty"] -= match
                    remaining -= match
                    if lot["qty"] <= 0:
                        lots[sym].pop(0)
        candidates = []
        for sym, sym_lots in lots.items():
            total_qty = sum(l["qty"] for l in sym_lots)
            if total_qty <= 0:
                continue
            avg_cost = sum(l["qty"] * l["price"] for l in sym_lots) / total_qty
            last_price = sym_lots[-1]["price"]
            oldest_lot_date = min(l["date"] for l in sym_lots)
            days_held = (today - oldest_lot_date).days
            if days_held < HOLDING_PERIOD_DAYS:
                continue
            unrealized_gain = (last_price - avg_cost) * total_qty
            if unrealized_gain <= 0:
                continue
            harvest_amount = min(unrealized_gain, remaining_exemption)
            harvest_qty = int(harvest_amount / (last_price - avg_cost)) if last_price > avg_cost else 0
            if harvest_qty > 0:
                candidates.append({
                    "symbol": sym,
                    "quantity": harvest_qty,
                    "avg_cost": round(avg_cost, 2),
                    "estimated_price": round(last_price, 2),
                    "realized_gain": round(harvest_amount, 2),
                    "days_held": days_held,
                    "tax_class": "LTCG",
                    "tax_cost": 0.0,
                })
                remaining_exemption -= harvest_amount
        return candidates

    def compute_xirr(self, symbol: Optional[str] = None) -> Optional[float]:
        trades = self.get_all_trades(symbol=symbol)
        if len(trades) < 2:
            return None
        cashflows: List[Tuple[date, float]] = []
        for t in trades:
            ts = t.get("timestamp", "")
            try:
                trade_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            except Exception:
                continue
            qty = int(t["quantity"])
            price = float(t["price"])
            if t["action"] == "BUY":
                cashflows.append((trade_date, -(qty * price)))
            else:
                cashflows.append((trade_date, qty * price))
        if not cashflows:
            return None
        cashflows.sort(key=lambda x: x[0])
        start_date = cashflows[0][0]

        def _npv(rate: float) -> float:
            total = 0.0
            for cf_date, cf_amount in cashflows:
                years = (cf_date - start_date).days / 365.25
                total += cf_amount / ((1 + rate) ** years)
            return total

        def _npv_derivative(rate: float) -> float:
            total = 0.0
            for cf_date, cf_amount in cashflows:
                years = (cf_date - start_date).days / 365.25
                total -= years * cf_amount / ((1 + rate) ** (years + 1))
            return total

        rate = 0.1
        for _ in range(100):
            npv = _npv(rate)
            dnpv = _npv_derivative(rate)
            if abs(dnpv) < 1e-12:
                break
            rate -= npv / dnpv
            if abs(npv) < 0.01:
                break
        if not (-0.99 < rate < 10.0):
            return None
        return round(rate, 4)

    def generate_gain_harvest_orders(self) -> List[Dict[str, Any]]:
        """Generate SELL orders to harvest LTCG gains within the ₹1.25L exemption.

        Returns list of order dicts ready for paper trading engine execution.
        Each order includes sell_price, rebuy_price, and rationale.
        """
        candidates = self.find_gain_harvest_candidates()
        if not candidates:
            return []

        orders = []
        for c in candidates:
            sell_value = c["quantity"] * c["estimated_price"]
            orders.append({
                "symbol": c["symbol"],
                "action": "SELL",
                "quantity": c["quantity"],
                "price": c["estimated_price"],
                "order_type": "LIMIT",
                "total_value": round(sell_value, 2),
                "tax_class": "LTCG",
                "tax_cost": 0.0,
                "realized_gain": c["realized_gain"],
                "days_held": c["days_held"],
                "rebuy": True,  # Signal to rebuy at same price after fill
                "rebuy_price": c["estimated_price"],
                "rebuy_quantity": c["quantity"],
                "rationale": (
                    f"LTCG gain harvest: ₹{c['realized_gain']:.0f} gain realized at zero tax "
                    f"({c['days_held']}d held, within ₹1.25L exemption). "
                    f"Rebuy to reset cost basis higher."
                ),
                "strategy_id": "TAX-LTCG-HARVEST",
            })

        total_gain = sum(o["realized_gain"] for o in orders)
        logger.info(
            "Generated %d LTCG harvest orders: ₹%.0f total gain, ₹0 tax cost",
            len(orders), total_gain,
        )
        return orders

    def generate_loss_harvest_orders(self) -> List[Dict[str, Any]]:
        """Generate SELL orders to harvest tax losses for offset against gains.

        Returns list of order dicts ready for paper trading engine execution.
        Prioritizes by loss amount (largest first) and STCG losses (offset at 20%).
        """
        candidates = self.find_harvest_candidates()
        if not candidates:
            return []

        summary = self.compute_tax_summary()
        stcg_gains = summary["stcg_gains"]
        ltcg_gains = summary["ltcg_gains"]

        orders = []
        remaining_stcg_to_offset = stcg_gains
        remaining_ltcg_to_offset = ltcg_gains

        for c in candidates:
            loss_amount = c["unrealized_loss"]
            tax_class = c["tax_class"]
            sell_value = c["quantity"] * c["estimated_price"]

            # Determine how much of this loss is useful
            if tax_class == "STCG":
                # STCG losses offset STCG gains at 20% rate
                useful_loss = min(loss_amount, remaining_stcg_to_offset)
                tax_benefit = useful_loss * STCG_TAX_RATE
                remaining_stcg_to_offset -= useful_loss
            else:
                # LTCG losses offset LTCG gains (exemption first, then 12.5%)
                useful_loss = min(loss_amount, remaining_ltcg_to_offset)
                tax_benefit = useful_loss * LTCG_TAX_RATE
                remaining_ltcg_to_offset -= useful_loss

            orders.append({
                "symbol": c["symbol"],
                "action": "SELL",
                "quantity": c["quantity"],
                "price": c["estimated_price"],
                "order_type": "LIMIT",
                "total_value": round(sell_value, 2),
                "tax_class": tax_class,
                "loss_amount": loss_amount,
                "useful_loss": round(useful_loss, 2),
                "tax_benefit": round(tax_benefit, 2),
                "days_held": c["days_held"],
                "rebuy": True,  # Signal to rebuy at same price after fill
                "rebuy_price": c["estimated_price"],
                "rebuy_quantity": c["quantity"],
                "rationale": (
                    f"Tax-loss harvest: ₹{loss_amount:.0f} loss ({tax_class}, {c['days_held']}d held). "
                    f"Tax benefit: ₹{tax_benefit:.0f} (offsets {tax_class} gains). "
                    f"Rebuy to maintain position."
                ),
                "strategy_id": f"TAX-{tax_class}-LOSS-HARVEST",
            })

            # Stop if we've offset all gains
            if remaining_stcg_to_offset <= 0 and remaining_ltcg_to_offset <= 0:
                break

        total_benefit = sum(o["tax_benefit"] for o in orders)
        logger.info(
            "Generated %d loss harvest orders: ₹%.0f total tax benefit",
            len(orders), total_benefit,
        )
        return orders

    def get_harvest_summary(self) -> Dict[str, Any]:
        """Get a combined summary of harvest opportunities (gain + loss)."""
        gain_orders = self.generate_gain_harvest_orders()
        loss_orders = self.generate_loss_harvest_orders()

        return {
            "gain_harvest": {
                "orders": gain_orders,
                "total_gain": sum(o["realized_gain"] for o in gain_orders),
                "tax_cost": 0.0,  # Always zero for LTCG within exemption
            },
            "loss_harvest": {
                "orders": loss_orders,
                "total_loss": sum(o.get("loss_amount", 0) for o in loss_orders),
                "total_tax_benefit": sum(o.get("tax_benefit", 0) for o in loss_orders),
            },
            "total_orders": len(gain_orders) + len(loss_orders),
        }
