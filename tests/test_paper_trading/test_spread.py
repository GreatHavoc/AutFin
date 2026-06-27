"""Tests for SpreadModel (paper_trading/engine.py)."""

import pytest
from unittest.mock import patch
from datetime import datetime

from src.autfin.paper_trading.engine import (
    SpreadModel,
    SlippageModel,
    PaperTradingEngine,
    SPREAD_LARGE_CAP,
    SPREAD_MID_CAP,
    SPREAD_SMALL_CAP,
    LARGECAP_THRESHOLD,
    MIDCAP_THRESHOLD,
)


class TestSpreadModelEstimateSpread:
    def test_largecap_base(self):
        sm = SpreadModel()
        spread = sm.estimate_spread_pct(market_cap_cr=50000)
        assert abs(spread - SPREAD_LARGE_CAP) < 0.01

    def test_midcap_base(self):
        sm = SpreadModel()
        spread = sm.estimate_spread_pct(market_cap_cr=10000)
        assert abs(spread - SPREAD_MID_CAP) < 0.01

    def test_smallcap_base(self):
        sm = SpreadModel()
        spread = sm.estimate_spread_pct(market_cap_cr=2000)
        assert abs(spread - SPREAD_SMALL_CAP) < 0.01

    def test_unknown_cap_defaults_to_midcap(self):
        sm = SpreadModel()
        spread = sm.estimate_spread_pct(market_cap_cr=None)
        assert abs(spread - SPREAD_MID_CAP) < 0.01

    def test_open_hour_widens_spread(self):
        sm = SpreadModel()
        spread_open = sm.estimate_spread_pct(market_cap_cr=50000, hour=9, minute=30)
        spread_midday = sm.estimate_spread_pct(market_cap_cr=50000, hour=12, minute=0)
        assert spread_open > spread_midday

    def test_close_hour_widens_spread(self):
        sm = SpreadModel()
        spread_close = sm.estimate_spread_pct(market_cap_cr=50000, hour=15, minute=15)
        spread_midday = sm.estimate_spread_pct(market_cap_cr=50000, hour=12, minute=0)
        assert spread_close > spread_midday

    def test_vix_widens_spread(self):
        sm = SpreadModel()
        spread_low_vix = sm.estimate_spread_pct(market_cap_cr=50000, vix=12)
        spread_high_vix = sm.estimate_spread_pct(market_cap_cr=50000, vix=25)
        assert spread_high_vix > spread_low_vix

    def test_vix_clamped(self):
        sm = SpreadModel()
        # Very high VIX should be clamped
        spread_extreme = sm.estimate_spread_pct(market_cap_cr=50000, vix=100)
        spread_high = sm.estimate_spread_pct(market_cap_cr=50000, vix=25)
        assert abs(spread_extreme - spread_high) < 0.01

    def test_all_factors_combined(self):
        sm = SpreadModel()
        # Smallcap + high VIX + open hour = widest spread
        worst = sm.estimate_spread_pct(market_cap_cr=2000, vix=25, hour=9, minute=30)
        # Largecap + low VIX + midday = tightest spread
        best = sm.estimate_spread_pct(market_cap_cr=50000, vix=12, hour=12, minute=0)
        assert worst > best * 3  # Should be significantly wider


class TestSpreadModelCalculateCost:
    def test_buy_pays_more(self):
        sm = SpreadModel()
        adjusted = sm.calculate_spread_cost(1000.0, 10, "BUY", market_cap_cr=50000)
        assert adjusted > 1000.0

    def test_sell_receives_less(self):
        sm = SpreadModel()
        adjusted = sm.calculate_spread_cost(1000.0, 10, "SELL", market_cap_cr=50000)
        assert adjusted < 1000.0

    def test_smallcap_spread_larger_than_largecap(self):
        sm = SpreadModel()
        adj_large = sm.calculate_spread_cost(1000.0, 10, "BUY", market_cap_cr=50000)
        adj_small = sm.calculate_spread_cost(1000.0, 10, "BUY", market_cap_cr=2000)
        assert adj_small > adj_large

    def test_spread_is_symmetric(self):
        sm = SpreadModel()
        adj_buy = sm.calculate_spread_cost(1000.0, 10, "BUY", market_cap_cr=50000)
        adj_sell = sm.calculate_spread_cost(1000.0, 10, "SELL", market_cap_cr=50000)
        # BUY premium ≈ SELL discount (symmetric around mid)
        buy_premium = adj_buy - 1000.0
        sell_discount = 1000.0 - adj_sell
        assert abs(buy_premium - sell_discount) < 0.01


class TestPaperTradingEngineSpread:
    def test_engine_has_spread_model(self, tmp_path):
        db = tmp_path / "test.db"
        engine = PaperTradingEngine(initial_cash=100000, db_path=str(db))
        assert isinstance(engine.spread, SpreadModel)

    def test_engine_accepts_custom_spread(self, tmp_path):
        db = tmp_path / "test.db"
        custom = SpreadModel(largecap_spread_pct=0.05)
        engine = PaperTradingEngine(initial_cash=100000, db_path=str(db), spread_model=custom)
        assert engine.spread.largecap_spread_pct == 0.05

    def test_fill_applies_both_slippage_and_spread(self, tmp_path):
        db = tmp_path / "test.db"
        engine = PaperTradingEngine(initial_cash=1000000, db_path=str(db))
        order = engine.place_order(
            symbol="RELIANCE",
            exchange="NSE",
            action="BUY",
            quantity=10,
            order_type="MARKET",
            price=2500.0,
        )
        # Fill price should be higher than market price (slippage + spread)
        assert order.filled_price > 2500.0
        assert order.status == "COMPLETE"
