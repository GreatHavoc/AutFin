"""Tests for the tax engine (STCG/LTCG, harvesting, XIRR).

Trades are written to a throwaway DB with the real schema, so the engine reads
the same columns the app writes.
"""
from __future__ import annotations

from datetime import date

import pytest


@pytest.fixture
def engine(tmp_path, monkeypatch):
    from autfin.memory import database
    from autfin.portfolio import TaxEngine

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "autfin.db")
    database.init_database()
    eng = TaxEngine(str(tmp_path / "autfin.db"))
    yield eng
    eng._connect().close()


def _trade(engine, symbol, action, qty, price, ts):
    conn = engine._connect()
    try:
        conn.execute(
            "INSERT INTO trades (timestamp, symbol, exchange, action, quantity, price, total_value) "
            "VALUES (?, ?, 'NSE', ?, ?, ?, ?)",
            (ts, symbol, action, qty, price, qty * price),
        )
        # TaxEngine._connect() is not autocommit — without this the row is
        # rolled back on close and every engine method sees an empty table.
        conn.commit()
    finally:
        conn.close()


def test_reads_trades_from_the_real_schema(engine):
    """The old SELECT asked for a `pnl` column that does not exist; the whole
    engine returned empty and silently produced zero tax."""
    _trade(engine, "RELIANCE", "BUY", 10, 2500.0, "2024-04-05")
    _trade(engine, "RELIANCE", "SELL", 10, 2600.0, "2024-06-06")

    trades = engine.get_all_trades("RELIANCE")
    assert len(trades) == 2
    assert trades[0]["action"] == "BUY"
    assert trades[0]["price"] == 2500.0


def test_stcg_classification(engine):
    days, tax_class, rate = engine.classify_trade(date(2024, 4, 5), date(2025, 1, 5))
    assert tax_class == "STCG"
    assert rate == 0.20


def test_ltcg_classification(engine):
    days, tax_class, rate = engine.classify_trade(date(2023, 4, 5), date(2024, 4, 5))
    assert tax_class == "LTCG"
    assert rate == 0.125


def test_short_term_gain_taxed_at_20(engine):
    _trade(engine, "TCS", "BUY", 10, 3000.0, "2024-04-05")
    _trade(engine, "TCS", "SELL", 10, 3600.0, "2024-06-06")  # +6000, held <12m

    summary = engine.compute_tax_summary(2024)
    assert summary["stcg_gains"] == 6000.0
    assert summary["stcg_tax"] == 1200.0
    assert summary["total_tax"] == 1200.0


def test_long_term_gain_uses_the_exemption_then_12_5pct(engine):
    # Two separate LTCG lots, both held over 12 months, totalling 150000.
    _trade(engine, "TCS", "BUY", 100, 1000.0, "2023-04-05")
    _trade(engine, "TCS", "SELL", 100, 2000.0, "2024-06-06")  # +100000
    _trade(engine, "INFY", "BUY", 100, 1000.0, "2023-05-05")
    _trade(engine, "INFY", "SELL", 100, 1500.0, "2024-06-06")  # +50000

    summary = engine.compute_tax_summary(2024)
    assert summary["ltcg_exemption_used"] == 125000.0
    assert summary["ltcg_gains"] == 25000.0
    assert summary["ltcg_tax"] == 3125.0
    assert summary["ltcg_exemption_remaining"] == 0.0
    assert summary["stcg_gains"] == 0.0
    assert summary["total_tax"] == 3125.0


def test_sells_outside_the_fy_are_not_taxed_in_it(engine):
    """Sells were only checked against fy_end, so a prior-FY sell leaked into
    the current year."""
    _trade(engine, "INFY", "BUY", 10, 1500.0, "2023-04-05")
    _trade(engine, "INFY", "SELL", 10, 1800.0, "2024-02-06")  # FY 2023-24

    inside = engine.compute_tax_summary(2023)
    outside = engine.compute_tax_summary(2024)

    assert inside["stcg_gains"] == 3000.0
    assert outside["stcg_gains"] == 0.0
    assert outside["total_tax"] == 0.0


def test_loss_harvest_sizes_off_the_loss_percentage(engine):
    _trade(engine, "SBIN", "BUY", 100, 600.0, "2024-06-05")
    _trade(engine, "SBIN", "BUY", 100, 560.0, "2024-07-05")  # last recorded price

    candidates = engine.find_harvest_candidates()
    assert len(candidates) == 1
    c = candidates[0]
    assert c["symbol"] == "SBIN"
    assert c["quantity"] == 200
    assert c["avg_cost"] == 580.0
    # Reported as a percentage, matching the field name: |560-580|/580 = 3.45%.
    assert c["unrealized_loss_pct"] == pytest.approx(20.0 / 580.0 * 100, abs=0.01)
    assert c["unrealized_loss"] == 4000.0


def test_no_harvest_candidates_without_losses(engine):
    _trade(engine, "SBIN", "BUY", 100, 600.0, "2023-04-05")
    _trade(engine, "SBIN", "SELL", 100, 700.0, "2024-06-05")
    assert engine.find_harvest_candidates() == []


def test_xirr_matches_a_known_20pct_year(engine):
    _trade(engine, "ITC", "BUY", 100, 100.0, "2024-01-01")
    _trade(engine, "ITC", "SELL", 100, 120.0, "2025-01-01")

    assert engine.compute_xirr("ITC") == pytest.approx(0.20, abs=1e-3)


def test_xirr_none_without_a_round_trip(engine):
    _trade(engine, "ITC", "BUY", 100, 100.0, "2024-01-01")
    assert engine.compute_xirr("ITC") is None


def test_gain_harvest_stays_inside_the_exemption(engine):
    _trade(engine, "HUL", "BUY", 100, 2000.0, "2023-06-05")
    _trade(engine, "HUL", "BUY", 100, 2400.0, "2023-07-05")  # +40000 unrealised

    orders = engine.generate_gain_harvest_orders()
    assert len(orders) == 1
    order = orders[0]
    assert order["action"] == "SELL"
    assert order["strategy_id"] == "TAX-LTCG-HARVEST"
    assert order["realized_gain"] == 40000.0
    assert order["tax_cost"] == 0.0
    assert order["rebuy"] is True
    # Selling all 200 shares at 2400 realises (2400-2000)*100 + (2400-2400)*100.
    assert order["quantity"] == 200
