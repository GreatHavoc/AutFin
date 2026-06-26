"""
Tests for PaperTradingEngine — simulated order fills with P&L tracking.
All tests run locally with in-memory or temp SQLite databases.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.autfin.paper_trading.engine import (
    PaperPosition,
    PaperOrder,
    PaperTradingEngine,
    get_paper_engine,
    reset_paper_engine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Fresh PaperTradingEngine with default 100k cash, no DB."""
    return PaperTradingEngine(initial_cash=100000.0)


@pytest.fixture
def engine_with_db(tmp_path):
    """PaperTradingEngine with SQLite persistence."""
    db_path = str(tmp_path / "test_paper.db")
    return PaperTradingEngine(initial_cash=100000.0, db_path=db_path)


# ---------------------------------------------------------------------------
# PaperPosition tests
# ---------------------------------------------------------------------------

class TestPaperPosition:
    def test_market_value(self):
        pos = PaperPosition(symbol="RELIANCE", quantity=10, last_price=2500.0)
        assert pos.market_value == 25000.0

    def test_cost_basis(self):
        pos = PaperPosition(symbol="RELIANCE", quantity=10, avg_price=2400.0)
        assert pos.cost_basis == 24000.0

    def test_unrealized_pnl_profit(self):
        pos = PaperPosition(symbol="RELIANCE", quantity=10, avg_price=2400.0, last_price=2500.0)
        assert pos.unrealized_pnl == 1000.0

    def test_unrealized_pnl_loss(self):
        pos = PaperPosition(symbol="RELIANCE", quantity=10, avg_price=2600.0, last_price=2500.0)
        assert pos.unrealized_pnl == -1000.0

    def test_unrealized_pnl_pct(self):
        pos = PaperPosition(symbol="RELIANCE", quantity=10, avg_price=200.0, last_price=220.0)
        assert pos.unrealized_pnl_pct == pytest.approx(10.0)

    def test_unrealized_pnl_pct_zero_cost(self):
        pos = PaperPosition(symbol="RELIANCE", quantity=0, avg_price=0.0, last_price=2500.0)
        assert pos.unrealized_pnl_pct == 0.0


# ---------------------------------------------------------------------------
# PaperOrder tests
# ---------------------------------------------------------------------------

class TestPaperOrder:
    def test_is_complete_status_complete(self):
        order = PaperOrder(order_id="P-1", symbol="RELIANCE", exchange="NSE",
                           action="BUY", quantity=10, order_type="MARKET", price=2500.0,
                           status="COMPLETE")
        assert order.is_complete is True

    def test_is_complete_status_pending(self):
        order = PaperOrder(order_id="P-1", symbol="RELIANCE", exchange="NSE",
                           action="BUY", quantity=10, order_type="LIMIT", price=2500.0,
                           status="PENDING")
        assert order.is_complete is False

    def test_trade_value_with_fill(self):
        order = PaperOrder(order_id="P-1", symbol="RELIANCE", exchange="NSE",
                           action="BUY", quantity=10, order_type="MARKET", price=2500.0,
                           filled_quantity=10, filled_price=2500.0)
        assert order.trade_value == 25000.0

    def test_trade_value_no_fill(self):
        order = PaperOrder(order_id="P-1", symbol="RELIANCE", exchange="NSE",
                           action="BUY", quantity=10, order_type="LIMIT", price=2500.0)
        assert order.trade_value == 0.0


# ---------------------------------------------------------------------------
# PaperTradingEngine — order placement
# ---------------------------------------------------------------------------

class TestOrderPlacement:
    def test_place_market_order_buy_fills(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        assert order.status == "COMPLETE"
        assert order.filled_quantity == 10
        assert order.filled_price == 2500.0
        assert order.filled_at is not None

    def test_place_limit_order_no_fill(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "LIMIT", price=2400.0)
        assert order.status == "PENDING"
        assert order.filled_quantity == 0

    def test_order_id_format(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        assert order.order_id.startswith("PAPER-")
        assert len(order.order_id) == 18  # PAPER- + 12 hex chars

    def test_order_stored_in_orders_dict(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        assert order.order_id in engine.orders

    def test_order_in_order_history(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        assert order in engine.order_history


# ---------------------------------------------------------------------------
# PaperTradingEngine — BUY flow
# ---------------------------------------------------------------------------

class TestBuyFlow:
    def test_buy_reduces_cash(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        assert engine.cash == 100000.0 - 25000.0

    def test_buy_creates_position(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        pos = engine.get_position("RELIANCE")
        assert pos is not None
        assert pos.quantity == 10
        assert pos.avg_price == 2500.0

    def test_buy_multiple_updates_avg_price(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2600.0)
        pos = engine.get_position("RELIANCE")
        assert pos.quantity == 20
        assert pos.avg_price == 2550.0

    def test_buy_insufficient_cash_rejects(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 100, "MARKET", price=2500.0)
        assert order.status == "REJECTED"
        assert engine.cash == 100000.0  # unchanged

    def test_buy_creates_trade_record(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        assert len(engine.trade_history) == 1
        trade = engine.trade_history[0]
        assert trade["action"] == "BUY"
        assert trade["pnl"] == 0.0  # no P&L on buy


# ---------------------------------------------------------------------------
# PaperTradingEngine — SELL flow
# ---------------------------------------------------------------------------

class TestSellFlow:
    def test_sell_increases_cash(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine.place_order("RELIANCE", "SELL", 10, "MARKET", price=2600.0)
        # Cash: 100k - 25k + 26k = 101k
        assert engine.cash == 101000.0

    def test_sell_records_pnl(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine.place_order("RELIANCE", "SELL", 10, "MARKET", price=2600.0)
        assert engine.total_pnl == 1000.0

    def test_sell_partial_reduces_position(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine.place_order("RELIANCE", "SELL", 5, "MARKET", price=2600.0)
        pos = engine.get_position("RELIANCE")
        assert pos.quantity == 5

    def test_sell_full_removes_position(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine.place_order("RELIANCE", "SELL", 10, "MARKET", price=2600.0)
        pos = engine.get_position("RELIANCE")
        assert pos is None

    def test_sell_no_position_rejects(self, engine):
        order = engine.place_order("RELIANCE", "SELL", 10, "MARKET", price=2600.0)
        assert order.status == "REJECTED"

    def test_sell_insufficient_quantity_rejects(self, engine):
        engine.place_order("RELIANCE", "BUY", 5, "MARKET", price=2500.0)
        order = engine.place_order("RELIANCE", "SELL", 10, "MARKET", price=2600.0)
        assert order.status == "REJECTED"


# ---------------------------------------------------------------------------
# PaperTradingEngine — modify & cancel
# ---------------------------------------------------------------------------

class TestModifyCancel:
    def test_modify_pending_order(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "LIMIT", price=2400.0)
        modified = engine.modify_order(order.order_id, quantity=20, price=2350.0)
        assert modified.quantity == 20
        assert modified.price == 2350.0

    def test_modify_complete_order_returns_none(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        modified = engine.modify_order(order.order_id, quantity=20)
        assert modified is None

    def test_modify_nonexistent_returns_none(self, engine):
        modified = engine.modify_order("PAPER-NONEXISTENT", quantity=20)
        assert modified is None

    def test_cancel_pending_order(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "LIMIT", price=2400.0)
        cancelled = engine.cancel_order(order.order_id)
        assert cancelled.status == "CANCELLED"
        assert cancelled.cancelled_at is not None

    def test_cancel_complete_order_returns_none(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        cancelled = engine.cancel_order(order.order_id)
        assert cancelled is None


# ---------------------------------------------------------------------------
# PaperTradingEngine — status & prices
# ---------------------------------------------------------------------------

class TestStatusPrices:
    def test_get_status_initial(self, engine):
        status = engine.get_status()
        assert status["initial_cash"] == 100000.0
        assert status["available_cash"] == 100000.0
        assert status["total_equity"] == 100000.0
        assert status["total_return_pct"] == 0.0
        assert status["open_positions"] == 0
        assert status["total_trades"] == 0

    def test_get_status_after_trade(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        status = engine.get_status()
        assert status["available_cash"] == 75000.0
        assert status["market_value"] == 25000.0
        assert status["total_equity"] == 100000.0
        assert status["open_positions"] == 1

    def test_update_prices(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine.update_prices({"RELIANCE": 2700.0})
        pos = engine.get_position("RELIANCE")
        assert pos.last_price == 2700.0

    def test_update_prices_with_exchange_key(self, engine):
        engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine.update_prices({"NSE:RELIANCE": 2700.0})
        pos = engine.get_position("RELIANCE")
        assert pos.last_price == 2700.0


# ---------------------------------------------------------------------------
# PaperTradingEngine — singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_paper_engine_returns_singleton(self):
        reset_paper_engine()
        e1 = get_paper_engine()
        e2 = get_paper_engine()
        assert e1 is e2

    def test_reset_paper_engine(self):
        e1 = get_paper_engine()
        reset_paper_engine()
        e2 = get_paper_engine()
        assert e1 is not e2


# ---------------------------------------------------------------------------
# PaperTradingEngine — SQLite persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_order_persisted_to_db(self, engine_with_db):
        engine_with_db.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        import sqlite3
        conn = sqlite3.connect(engine_with_db._db_path)
        row = conn.execute("SELECT * FROM paper_orders WHERE symbol='RELIANCE'").fetchone()
        conn.close()
        assert row is not None

    def test_position_persisted_to_db(self, engine_with_db):
        engine_with_db.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        import sqlite3
        conn = sqlite3.connect(engine_with_db._db_path)
        row = conn.execute("SELECT * FROM paper_positions WHERE symbol='RELIANCE'").fetchone()
        conn.close()
        assert row is not None
        assert row[2] == 10  # quantity

    def test_position_deleted_on_full_sell(self, engine_with_db):
        engine_with_db.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        engine_with_db.place_order("RELIANCE", "SELL", 10, "MARKET", price=2600.0)
        import sqlite3
        conn = sqlite3.connect(engine_with_db._db_path)
        row = conn.execute("SELECT * FROM paper_positions WHERE symbol='RELIANCE'").fetchone()
        conn.close()
        assert row is None


# ---------------------------------------------------------------------------
# PaperTradingEngine — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_quantity_buy(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 0, "MARKET", price=2500.0)
        assert order.status == "COMPLETE"
        assert engine.cash == 100000.0

    def test_no_tag_order(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0)
        assert order.tag is None

    def test_order_with_tag(self, engine):
        order = engine.place_order("RELIANCE", "BUY", 10, "MARKET", price=2500.0, tag="research_buy")
        assert order.tag == "research_buy"
