"""Paper trading engine — simulated order fills with P&L tracking."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PaperPosition:
    symbol: str
    exchange: str = "NSE"
    quantity: int = 0
    avg_price: float = 0.0
    last_price: float = 0.0
    product: str = "CNC"
    created_at: str = field(default_factory=_now_iso)

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    exchange: str
    action: str
    quantity: int
    order_type: str
    price: float
    trigger_price: float = 0.0
    product: str = "CNC"
    status: str = "PENDING"
    filled_quantity: int = 0
    filled_price: float = 0.0
    placed_at: str = field(default_factory=_now_iso)
    filled_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    tag: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in ("COMPLETE", "CANCELLED", "REJECTED")

    @property
    def trade_value(self) -> float:
        fp = self.filled_price or self.price
        return self.filled_quantity * fp


class PaperTradingEngine:
    """Simulated trading engine for paper/backtest mode.

    Tracks orders, positions, cash, and P&L in-memory and persists to SQLite.
    Fills market orders instantly at the provided price. Limit orders are
    filled only if the limit condition is met.
    """

    def __init__(self, initial_cash: float = 100000.0, db_path: Optional[str] = None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.order_history: List[PaperOrder] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.total_pnl: float = 0.0
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        if not self._db_path:
            return
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS paper_orders (
                    order_id TEXT PRIMARY KEY, symbol TEXT, exchange TEXT,
                    action TEXT, quantity INTEGER, order_type TEXT,
                    price REAL, trigger_price REAL, product TEXT,
                    status TEXT, filled_quantity INTEGER, filled_price REAL,
                    placed_at TEXT, filled_at TEXT, cancelled_at TEXT,
                    tag TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS paper_positions (
                    symbol TEXT PRIMARY KEY, exchange TEXT,
                    quantity INTEGER, avg_price REAL, last_price REAL,
                    product TEXT, created_at TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS paper_trades (
                    id TEXT PRIMARY KEY, order_id TEXT, symbol TEXT,
                    action TEXT, quantity INTEGER, price REAL,
                    value REAL, pnl REAL, pnl_pct REAL, filled_at TEXT
                )"""
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Paper trading DB init failed: {e}")

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0.0,
        trigger_price: float = 0.0,
        product: str = "CNC",
        exchange: str = "NSE",
        tag: Optional[str] = None,
    ) -> PaperOrder:
        order_id = f"PAPER-{uuid.uuid4().hex[:12].upper()}"
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            exchange=exchange,
            action=action.upper(),
            quantity=quantity,
            order_type=order_type.upper(),
            price=price,
            trigger_price=trigger_price,
            product=product,
            tag=tag,
        )
        self.orders[order_id] = order

        if order_type.upper() == "MARKET" and price > 0:
            self._fill_order(order, price)

        return order

    def _fill_order(self, order: PaperOrder, fill_price: float):
        order.status = "COMPLETE"
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.filled_at = _now_iso()

        key = f"{order.exchange}:{order.symbol}"
        sell_pnl = 0.0
        sell_pnl_pct = 0.0
        if order.action == "BUY":
            cost = order.filled_quantity * fill_price
            if self.cash < cost:
                order.status = "REJECTED"
                order.filled_quantity = 0
                return
            self.cash -= cost

            if key in self.positions:
                pos = self.positions[key]
                total_qty = pos.quantity + order.filled_quantity
                total_cost = pos.cost_basis + cost
                pos.quantity = total_qty
                pos.avg_price = total_cost / total_qty if total_qty > 0 else 0
                pos.last_price = fill_price
            else:
                self.positions[key] = PaperPosition(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    quantity=order.filled_quantity,
                    avg_price=fill_price,
                    last_price=fill_price,
                    product=order.product,
                )
        else:
            if key not in self.positions:
                order.status = "REJECTED"
                return
            pos = self.positions[key]
            if pos.quantity < order.filled_quantity:
                order.status = "REJECTED"
                return

            revenue = order.filled_quantity * fill_price
            self.cash += revenue

            sell_pnl = (fill_price - pos.avg_price) * order.filled_quantity
            self.total_pnl += sell_pnl

            pos.quantity -= order.filled_quantity
            pos.last_price = fill_price
            if pos.quantity == 0:
                del self.positions[key]

            sell_pnl_pct = (
                ((fill_price - pos.avg_price) / pos.avg_price) * 100
                if pos.avg_price > 0
                else 0
            )

        trade = {
            "id": f"TR-{uuid.uuid4().hex[:8].upper()}",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "action": order.action,
            "quantity": order.filled_quantity,
            "price": fill_price,
            "value": order.trade_value,
            "pnl": sell_pnl if order.action == "SELL" else 0.0,
            "pnl_pct": sell_pnl_pct if order.action == "SELL" else 0.0,
            "filled_at": order.filled_at,
        }
        self.trade_history.append(trade)
        self.order_history.append(order)

        self._persist_order(order)
        self._persist_position(key)

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
    ) -> Optional[PaperOrder]:
        order = self.orders.get(order_id)
        if not order or order.is_complete:
            return None
        if quantity is not None:
            order.quantity = quantity
        if price is not None:
            order.price = price
        if trigger_price is not None:
            order.trigger_price = trigger_price
        return order

    def cancel_order(self, order_id: str) -> Optional[PaperOrder]:
        order = self.orders.get(order_id)
        if not order or order.is_complete:
            return None
        order.status = "CANCELLED"
        order.cancelled_at = _now_iso()
        self.order_history.append(order)
        return order

    def update_prices(self, prices: Dict[str, float]):
        for key, pos in self.positions.items():
            _, symbol = key.split(":", 1) if ":" in key else ("NSE", key)
            if symbol in prices:
                pos.last_price = prices[symbol]
            elif key in prices:
                pos.last_price = prices[key]

    def get_status(self) -> Dict[str, Any]:
        total_market_value = sum(p.market_value for p in self.positions.values())
        total_cost = sum(p.cost_basis for p in self.positions.values())
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return {
            "initial_cash": self.initial_cash,
            "available_cash": self.cash,
            "market_value": round(total_market_value, 2),
            "total_cost": round(total_cost, 2),
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl": round(self.total_pnl, 2),
            "total_equity": round(self.cash + total_market_value, 2),
            "total_return_pct": round(
                ((self.cash + total_market_value - self.initial_cash) / self.initial_cash) * 100, 2
            ),
            "open_positions": len(self.positions),
            "open_orders": len([o for o in self.orders.values() if not o.is_complete]),
            "total_trades": len(self.trade_history),
            "positions": {
                k: {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_price": round(p.avg_price, 2),
                    "last_price": round(p.last_price, 2),
                    "market_value": round(p.market_value, 2),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 2),
                }
                for k, p in self.positions.items()
            },
        }

    def get_position(self, symbol: str, exchange: str = "NSE") -> Optional[PaperPosition]:
        return self.positions.get(f"{exchange}:{symbol}")

    def _persist_order(self, order: PaperOrder):
        if not self._db_path:
            return
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """INSERT OR REPLACE INTO paper_orders VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    order.order_id, order.symbol, order.exchange,
                    order.action, order.quantity, order.order_type,
                    order.price, order.trigger_price, order.product,
                    order.status, order.filled_quantity, order.filled_price,
                    order.placed_at, order.filled_at, order.cancelled_at,
                    order.tag,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _persist_position(self, key: str):
        if not self._db_path:
            return
        pos = self.positions.get(key)
        try:
            conn = sqlite3.connect(self._db_path)
            if pos:
                conn.execute(
                    """INSERT OR REPLACE INTO paper_positions VALUES (?,?,?,?,?,?)""",
                    (
                        pos.symbol, pos.exchange, pos.quantity,
                        pos.avg_price, pos.last_price, pos.product,
                        pos.created_at,
                    ),
                )
            else:
                symbol = key.split(":", 1)[1] if ":" in key else key
                conn.execute("DELETE FROM paper_positions WHERE symbol=?", (symbol,))
            conn.commit()
            conn.close()
        except Exception:
            pass


_paper_engine: Optional[PaperTradingEngine] = None


def get_paper_engine(initial_cash: float = 100000.0) -> PaperTradingEngine:
    global _paper_engine
    if _paper_engine is None:
        from pathlib import Path

        db = Path(__file__).parent.parent.parent.parent / "data" / "paper_trading.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        _paper_engine = PaperTradingEngine(
            initial_cash=initial_cash, db_path=str(db)
        )
    return _paper_engine


def reset_paper_engine():
    global _paper_engine
    _paper_engine = None
