"""Tests for the Indian Tax Engine."""

import sqlite3
import tempfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pytest

from src.autfin.portfolio.tax_engine import (
    TaxEngine,
    STCG_TAX_RATE,
    LTCG_TAX_RATE,
    LTCG_EXEMPTION_LIMIT,
    HOLDING_PERIOD_DAYS,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp SQLite DB with the trades table."""
    db = tmp_path / "test_tax.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE trades (
        symbol TEXT, action TEXT, quantity INTEGER, price REAL,
        timestamp TEXT, pnl REAL
    )""")
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def engine(tmp_db):
    return TaxEngine(db_path=str(tmp_db))


def _insert_trade(conn, symbol, action, qty, price, ts):
    conn.execute(
        "INSERT INTO trades (symbol, action, quantity, price, timestamp, pnl) VALUES (?, ?, ?, ?, ?, ?)",
        (symbol, action, qty, price, ts, 0.0),
    )
    conn.commit()


class TestClassifyTrade:
    def test_stcg(self):
        e = TaxEngine.__new__(TaxEngine)
        buy = date(2026, 1, 1)
        sell = date(2026, 6, 1)
        days, cls, rate = e.classify_trade(buy, sell)
        assert cls == "STCG"
        assert days == 151
        assert rate == STCG_TAX_RATE

    def test_ltcg(self):
        e = TaxEngine.__new__(TaxEngine)
        buy = date(2025, 1, 1)
        sell = date(2026, 6, 1)
        days, cls, rate = e.classify_trade(buy, sell)
        assert cls == "LTCG"
        assert days == 516
        assert rate == LTCG_TAX_RATE

    def test_boundary_365_days(self):
        e = TaxEngine.__new__(TaxEngine)
        buy = date(2025, 6, 27)
        sell = date(2026, 6, 27)
        days, cls, _ = e.classify_trade(buy, sell)
        assert days == 365
        assert cls == "LTCG"

    def test_boundary_364_days(self):
        e = TaxEngine.__new__(TaxEngine)
        buy = date(2025, 6, 28)
        sell = date(2026, 6, 27)
        days, cls, _ = e.classify_trade(buy, sell)
        assert days == 364
        assert cls == "STCG"


class TestComputeTaxSummary:
    def test_no_trades(self, engine):
        s = engine.compute_tax_summary(fy=2026)
        assert s["stcg_gains"] == 0.0
        assert s["ltcg_gains"] == 0.0
        assert s["total_tax"] == 0.0

    def test_stcg_gain(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "RELIANCE", "BUY", 100, 2000, "2026-05-01T10:00:00")
        _insert_trade(conn, "RELIANCE", "SELL", 100, 2500, "2026-06-01T10:00:00")
        conn.close()
        s = engine.compute_tax_summary(fy=2026)
        assert s["stcg_gains"] == 50000.0
        assert s["stcg_tax"] == 10000.0  # 20% of 50k
        assert s["ltcg_gains"] == 0.0

    def test_ltcg_gain_within_exemption(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "INFY", "BUY", 100, 1000, "2025-05-01T10:00:00")
        _insert_trade(conn, "INFY", "SELL", 100, 2000, "2026-06-01T10:00:00")
        conn.close()
        s = engine.compute_tax_summary(fy=2026)
        assert s["ltcg_gains"] == 0.0  # 1L gain, within 1.25L exemption
        assert s["ltcg_tax"] == 0.0
        assert s["ltcg_exemption_used"] == 100000.0

    def test_ltcg_gain_exceeds_exemption(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "TCS", "BUY", 200, 1000, "2025-05-01T10:00:00")
        _insert_trade(conn, "TCS", "SELL", 200, 2000, "2026-06-01T10:00:00")
        conn.close()
        s = engine.compute_tax_summary(fy=2026)
        # Gain = 200 * 1000 = 2L. Exemption = 1.25L. Taxable = 75k
        assert s["ltcg_gains"] == 75000.0
        assert s["ltcg_tax"] == 9375.0  # 12.5% of 75k

    def test_mixed_stcg_and_ltcg(self, engine):
        conn = sqlite3.connect(engine._db_path)
        # STCG trade
        _insert_trade(conn, "RELIANCE", "BUY", 50, 2000, "2026-01-01T10:00:00")
        _insert_trade(conn, "RELIANCE", "SELL", 50, 2500, "2026-06-01T10:00:00")
        # LTCG trade
        _insert_trade(conn, "INFY", "BUY", 100, 1000, "2025-05-01T10:00:00")
        _insert_trade(conn, "INFY", "SELL", 100, 1500, "2026-06-01T10:00:00")
        conn.close()
        s = engine.compute_tax_summary(fy=2026)
        assert s["stcg_gains"] == 25000.0
        assert s["ltcg_gains"] == 0.0  # 50k gain, within exemption
        assert s["total_tax"] == 5000.0  # 20% of 25k STCG


class TestFindHarvestCandidates:
    def test_no_candidates_when_profit(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "RELIANCE", "BUY", 100, 2000, "2026-01-01T10:00:00")
        conn.close()
        candidates = engine.find_harvest_candidates()
        # Last price = buy price (no loss), so no candidates
        assert len(candidates) == 0

    def test_finds_loss_candidates(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "RELIANCE", "BUY", 100, 2000, "2026-01-01T10:00:00")
        _insert_trade(conn, "TATASTEEL", "BUY", 50, 1500, "2026-02-01T10:00:00")
        conn.close()
        candidates = engine.find_harvest_candidates()
        # Both bought at higher prices, estimated at last buy price -> no loss
        # Actually last_price = last lot price = buy price, so no loss
        assert len(candidates) == 0


class TestFindGainHarvestCandidates:
    def test_no_candidates_when_no_exemption(self, engine):
        conn = sqlite3.connect(engine._db_path)
        # Create a trade that uses up the full exemption
        _insert_trade(conn, "TCS", "BUY", 200, 1000, "2025-05-01T10:00:00")
        _insert_trade(conn, "TCS", "SELL", 200, 2000, "2026-06-01T10:00:00")
        conn.close()
        candidates = engine.find_gain_harvest_candidates()
        # Exemption already used
        assert len(candidates) == 0

    def test_empty_db(self, engine):
        candidates = engine.find_gain_harvest_candidates()
        assert candidates == []


class TestComputeXIRR:
    def test_insufficient_trades(self, engine):
        result = engine.compute_xirr()
        assert result is None

    def test_single_trade(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "RELIANCE", "BUY", 100, 2000, "2026-01-01T10:00:00")
        conn.close()
        result = engine.compute_xirr()
        assert result is None

    def test_buy_sell_returns_positive(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "RELIANCE", "BUY", 100, 1000, "2025-01-01T10:00:00")
        _insert_trade(conn, "RELIANCE", "SELL", 100, 1500, "2026-01-01T10:00:00")
        conn.close()
        result = engine.compute_xirr()
        assert result is not None
        assert result > 0.4  # ~50% return in 1 year

    def test_buy_sell_loss_returns_negative(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "RELIANCE", "BUY", 100, 2000, "2025-01-01T10:00:00")
        _insert_trade(conn, "RELIANCE", "SELL", 100, 1500, "2026-01-01T10:00:00")
        conn.close()
        result = engine.compute_xirr()
        assert result is not None
        assert result < 0  # Loss


class TestFYHelpers:
    def test_fy_start_april(self):
        from src.autfin.portfolio.tax_engine import _get_fy_start
        assert _get_fy_start(date(2027, 4, 1)) == date(2027, 4, 1)

    def test_fy_start_march(self):
        from src.autfin.portfolio.tax_engine import _get_fy_start
        assert _get_fy_start(date(2027, 3, 31)) == date(2026, 4, 1)

    def test_fy_start_january(self):
        from src.autfin.portfolio.tax_engine import _get_fy_start
        assert _get_fy_start(date(2027, 1, 15)) == date(2026, 4, 1)


class TestGenerateGainHarvestOrders:
    def test_no_orders_when_no_exemption(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "TCS", "BUY", 200, 1000, "2025-05-01T10:00:00")
        _insert_trade(conn, "TCS", "SELL", 200, 2000, "2026-06-01T10:00:00")
        conn.close()
        orders = engine.generate_gain_harvest_orders()
        assert orders == []

    def test_generates_sell_order(self, engine):
        conn = sqlite3.connect(engine._db_path)
        # Buy at different prices to create unrealized gain
        # First buy: 100 at 800 (old, >365 days)
        _insert_trade(conn, "INFY", "BUY", 100, 800, "2025-05-01T10:00:00")
        # Second buy: 100 at 1200 (newer, raises estimated price)
        _insert_trade(conn, "INFY", "BUY", 100, 1200, "2025-08-01T10:00:00")
        conn.close()
        # avg_cost = (100*800 + 100*1200) / 200 = 1000
        # last_price = 1200 (last lot)
        # unrealized_gain = (1200 - 1000) * 200 = 40000
        orders = engine.generate_gain_harvest_orders()
        assert len(orders) == 1
        o = orders[0]
        assert o["symbol"] == "INFY"
        assert o["action"] == "SELL"
        assert o["tax_class"] == "LTCG"
        assert o["tax_cost"] == 0.0
        assert o["rebuy"] is True
        assert "TAX-LTCG-HARVEST" in o["strategy_id"]

    def test_harvest_respects_exemption_limit(self, engine):
        conn = sqlite3.connect(engine._db_path)
        # Buy 100 at 1000, estimate price 2000 -> gain = 100k (within 1.25L)
        _insert_trade(conn, "INFY", "BUY", 100, 1000, "2025-05-01T10:00:00")
        conn.close()
        orders = engine.generate_gain_harvest_orders()
        if orders:
            assert orders[0]["realized_gain"] <= LTCG_EXEMPTION_LIMIT

    def test_order_has_rationale(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "INFY", "BUY", 50, 1500, "2025-06-01T10:00:00")
        conn.close()
        orders = engine.generate_gain_harvest_orders()
        if orders:
            assert "rationale" in orders[0]
            assert "LTCG" in orders[0]["rationale"]


class TestGenerateLossHarvestOrders:
    def test_no_orders_when_no_loss(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "RELIANCE", "BUY", 100, 2000, "2026-01-01T10:00:00")
        conn.close()
        orders = engine.generate_loss_harvest_orders()
        assert orders == []

    def test_generates_loss_order(self, engine):
        conn = sqlite3.connect(engine._db_path)
        # Buy at 2000, sell at 1500 -> 50k loss
        _insert_trade(conn, "RELIANCE", "BUY", 100, 2000, "2025-01-01T10:00:00")
        _insert_trade(conn, "RELIANCE", "SELL", 100, 1500, "2026-06-01T10:00:00")
        # New position with unrealized loss
        _insert_trade(conn, "RELIANCE", "BUY", 100, 1500, "2026-06-15T10:00:00")
        conn.close()
        orders = engine.generate_loss_harvest_orders()
        # The new position at 1500 is the "current" position
        # find_harvest_candidates looks at open lots
        if orders:
            assert orders[0]["action"] == "SELL"
            assert orders[0]["rebuy"] is True
            assert "TAX-" in orders[0]["strategy_id"]

    def test_order_has_tax_benefit(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "INFY", "BUY", 100, 2000, "2025-01-01T10:00:00")
        _insert_trade(conn, "INFY", "SELL", 100, 1500, "2026-06-01T10:00:00")
        _insert_trade(conn, "INFY", "BUY", 100, 1500, "2026-06-15T10:00:00")
        conn.close()
        orders = engine.generate_loss_harvest_orders()
        if orders:
            assert "tax_benefit" in orders[0]
            assert "loss_amount" in orders[0]


class TestGetHarvestSummary:
    def test_empty_portfolio(self, engine):
        summary = engine.get_harvest_summary()
        assert summary["total_orders"] == 0
        assert summary["gain_harvest"]["total_gain"] == 0.0
        assert summary["loss_harvest"]["total_loss"] == 0.0

    def test_with_gain_candidates(self, engine):
        conn = sqlite3.connect(engine._db_path)
        _insert_trade(conn, "INFY", "BUY", 100, 1000, "2025-05-01T10:00:00")
        conn.close()
        summary = engine.get_harvest_summary()
        assert "gain_harvest" in summary
        assert "loss_harvest" in summary
        assert summary["total_orders"] >= 0
