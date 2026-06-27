"""Paper trading engine — simulated order fills with P&L tracking."""

from __future__ import annotations

import logging
import random
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
class SlippageModel:
    """Simulates realistic market slippage for paper trading.

    Components:
      - Base slippage: random component (0 to base_slippage_pct)
      - Market impact: scales with order value (larger orders = more impact)
      - Max cap: prevents extreme slippage for illiquid stocks
    """

    base_slippage_pct: float = 0.05
    max_slippage_pct: float = 0.3
    impact_factor: float = 0.0001

    def calculate_slippage(self, price: float, quantity: int, action: str) -> float:
        """Calculate slippage-adjusted fill price.

        Args:
            price: Market price of the stock
            quantity: Number of shares
            action: BUY or SELL

        Returns:
            Adjusted fill price with slippage applied
        """
        order_value = price * quantity

        # Base slippage (random component)
        base = self.base_slippage_pct / 100.0
        random_slippage = random.uniform(0, base)

        # Market impact component (per ₹1L of order value)
        impact = self.impact_factor * (order_value / 100000)

        # Total slippage percentage, capped at max
        total_slippage = min(random_slippage + impact, self.max_slippage_pct / 100.0)

        # Apply direction: BUY pays more, SELL receives less
        if action == "BUY":
            return price * (1 + total_slippage)
        else:
            return price * (1 - total_slippage)


# Indian market cap category thresholds (market cap in ₹Cr)
LARGECAP_THRESHOLD = 20000    # >₹20K Cr = largecap
MIDCAP_THRESHOLD = 5000       # ₹5K-20K Cr = midcap
# <₹5K Cr = smallcap

# Base spread estimates by category (% of price)
SPREAD_LARGE_CAP = 0.08       # 0.08% — tight spreads (RELIANCE, TCS, INFY)
SPREAD_MID_CAP = 0.30         # 0.30% — moderate spreads
SPREAD_SMALL_CAP = 1.00       # 1.00% — wide spreads (illiquid)

# Time-of-day multipliers (market hours 9:15-15:30 IST)
# Spreads widen at open/close, narrow mid-day
TIME_OPEN_MULTIPLIER = 1.5    # First 30 min (9:15-9:45)
TIME_CLOSE_MULTIPLIER = 1.3   # Last 30 min (15:00-15:30)
TIME_MIDDAY_MULTIPLIER = 1.0  # Normal hours

# VIX multiplier: spread widens with volatility
VIX_LOW = 12.0
VIX_HIGH = 25.0
VIX_SPREAD_MULTIPLIER_RANGE = (1.0, 2.0)  # (low vol, high vol)


@dataclass
class SpreadModel:
    """Estimates bid-ask spread for Indian equities.

    Spread is the dominant cost for mid/smallcap stocks, often larger
    than slippage. This model estimates spread based on:
      - Market cap category (largecap/midcap/smallcap)
      - Time of day (wider at open/close)
      - VIX level (wider in high volatility)
      - Order value (larger orders consume more of the book)
    """

    # Override base spreads if needed
    largecap_spread_pct: float = SPREAD_LARGE_CAP
    midcap_spread_pct: float = SPREAD_MID_CAP
    smallcap_spread_pct: float = SPREAD_SMALL_CAP

    def estimate_spread_pct(
        self,
        market_cap_cr: Optional[float] = None,
        vix: Optional[float] = None,
        hour: Optional[int] = None,
        minute: Optional[int] = None,
    ) -> float:
        """Estimate the bid-ask spread as a percentage of price.

        Args:
            market_cap_cr: Market cap in ₹Cr (determines category)
            vix: Current India VIX level
            hour: Current hour (IST, 0-23)
            minute: Current minute (0-59)

        Returns:
            Spread percentage (e.g., 0.30 for 0.30%)
        """
        # 1. Base spread by market cap category
        if market_cap_cr is not None:
            if market_cap_cr >= LARGECAP_THRESHOLD:
                base = self.largecap_spread_pct
            elif market_cap_cr >= MIDCAP_THRESHOLD:
                base = self.midcap_spread_pct
            else:
                base = self.smallcap_spread_pct
        else:
            # Default to midcap if unknown
            base = self.midcap_spread_pct

        # 2. Time-of-day multiplier
        time_mult = TIME_MIDDAY_MULTIPLIER
        if hour is not None:
            if hour == 9 and (minute or 0) < 45:
                time_mult = TIME_OPEN_MULTIPLIER
            elif hour == 15 and (minute or 0) >= 0:
                time_mult = TIME_CLOSE_MULTIPLIER
            elif hour < 9 or hour > 15:
                time_mult = TIME_CLOSE_MULTIPLIER  # After-hours = close spread

        # 3. VIX multiplier
        vix_mult = 1.0
        if vix is not None:
            vix_clamped = max(VIX_LOW, min(VIX_HIGH, vix))
            vix_range = VIX_HIGH - VIX_LOW
            vix_pct = (vix_clamped - VIX_LOW) / vix_range if vix_range > 0 else 0.5
            low_mult, high_mult = VIX_SPREAD_MULTIPLIER_RANGE
            vix_mult = low_mult + vix_pct * (high_mult - low_mult)

        return base * time_mult * vix_mult

    def calculate_spread_cost(
        self,
        price: float,
        quantity: int,
        action: str,
        market_cap_cr: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> float:
        """Calculate spread-adjusted fill price.

        Args:
            price: Mid-market price
            quantity: Number of shares
            action: BUY or SELL
            market_cap_cr: Market cap in ₹Cr
            vix: Current India VIX

        Returns:
            Price adjusted for bid-ask spread
        """
        now = datetime.now()
        spread_pct = self.estimate_spread_pct(
            market_cap_cr=market_cap_cr,
            vix=vix,
            hour=now.hour,
            minute=now.minute,
        )

        # Spread is half on each side — midpoint to best bid/ask
        spread_half = spread_pct / 200.0  # Convert % to fraction, then half

        # BUY hits the ask (pay more), SELL hits the bid (receive less)
        if action == "BUY":
            return price * (1 + spread_half)
        else:
            return price * (1 - spread_half)


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

    def __init__(self, initial_cash: float = 100000.0, db_path: Optional[str] = None, slippage_model: Optional[SlippageModel] = None, spread_model: Optional[SpreadModel] = None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trade_history: List[Dict] = []
        self.order_history: List[PaperOrder] = []
        self.db_path = db_path
        self.slippage = slippage_model or SlippageModel()
        self.spread = spread_model or SpreadModel()
        self._init_db()

    def _init_db(self):
        if not self.db_path:
            return
        try:
            conn = sqlite3.connect(self.db_path)
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

    async def _fetch_live_price(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """Fetch live price for a symbol via yfinance."""
        try:
            import yfinance as yf
            ticker = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
            t = yf.Ticker(ticker)
            data = t.history(period="1d")
            if data is not None and not data.empty:
                return float(data["Close"].iloc[-1])
            return None
        except Exception as e:
            logger.warning("Live price fetch failed for %s: %s", symbol, e)
            return None

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

        if order_type.upper() == "MARKET":
            if price > 0:
                self._fill_order(order, price)
            else:
                # Fetch live price for MARKET orders submitted without price
                import asyncio
                try:
                    live_price = asyncio.get_event_loop().run_until_complete(
                        self._fetch_live_price(symbol, exchange)
                    )
                    if live_price and live_price > 0:
                        self._fill_order(order, live_price)
                    else:
                        # If live price unavailable, use 0.05% slippage from last known
                        logger.warning("MARKET order for %s: no live price available, using last known", symbol)
                        order.status = "REJECTED"
                        order.notes = "MARKET order rejected: live price unavailable"
                except Exception as e:
                    logger.warning("MARKET order fetch failed for %s: %s", symbol, e)
                    order.status = "REJECTED"
                    order.notes = f"MARKET order rejected: {e}"

        return order

    def _fill_order(self, order: PaperOrder, fill_price: float):
        # Apply slippage to fill price
        adjusted_price = self.slippage.calculate_slippage(
            fill_price, order.quantity, order.action
        )
        # Apply spread on top of slippage
        adjusted_price = self.spread.calculate_spread_cost(
            adjusted_price, order.quantity, order.action
        )

        order.status = "COMPLETE"
        order.filled_quantity = order.quantity
        order.filled_price = adjusted_price
        order.filled_at = _now_iso()

        key = f"{order.exchange}:{order.symbol}"
        sell_pnl = 0.0
        sell_pnl_pct = 0.0
        if order.action == "BUY":
            cost = order.filled_quantity * adjusted_price
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
                pos.last_price = adjusted_price
            else:
                self.positions[key] = PaperPosition(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    quantity=order.filled_quantity,
                    avg_price=adjusted_price,
                    last_price=adjusted_price,
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

            revenue = order.filled_quantity * adjusted_price
            self.cash += revenue

            sell_pnl = (adjusted_price - pos.avg_price) * order.filled_quantity
            self.total_pnl += sell_pnl

            pos.quantity -= order.filled_quantity
            pos.last_price = adjusted_price
            if pos.quantity == 0:
                del self.positions[key]

            sell_pnl_pct = (
                ((adjusted_price - pos.avg_price) / pos.avg_price) * 100
                if pos.avg_price > 0
                else 0
            )

        trade = {
            "id": f"TR-{uuid.uuid4().hex[:8].upper()}",
            "order_id": order.order_id,
            "symbol": order.symbol,
            "action": order.action,
            "quantity": order.filled_quantity,
            "price": adjusted_price,
            "value": order.trade_value,
            "pnl": sell_pnl if order.action == "SELL" else 0.0,
            "pnl_pct": sell_pnl_pct if order.action == "SELL" else 0.0,
            "has_pnl": order.action == "SELL",  # Only SELL trades have meaningful P&L
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
        if not self.db_path:
            return
        try:
            conn = sqlite3.connect(self.db_path)
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
        if not self.db_path:
            return
        pos = self.positions.get(key)
        try:
            conn = sqlite3.connect(self.db_path)
            if pos:
                conn.execute(
                    """INSERT OR REPLACE INTO paper_positions VALUES (?,?,?,?,?,?,?)""",
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
