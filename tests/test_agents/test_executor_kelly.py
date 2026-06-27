"""Tests for Kelly integration in executor."""

import pytest

from src.autfin.agents.executor import _kelly_sized_qty


class TestKellySizedQty:
    def test_no_kelly_data_unchanged(self):
        trade = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "qty": 50,
            "price": 2500.0,
            "order_type": "MARKET",
        }
        result = _kelly_sized_qty(trade, available_cash=500000.0)
        assert result["qty"] == 50
        assert "kelly_skip" not in result

    def test_positive_edge_sizing(self):
        trade = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "qty": 100,
            "price": 2500.0,
            "order_type": "MARKET",
            "win_rate": 0.6,
            "avg_win": 0.05,
            "avg_loss": 0.03,
            "conviction": 0.7,
        }
        result = _kelly_sized_qty(trade, available_cash=500000.0)
        # Kelly should size appropriately
        assert result["qty"] > 0
        assert result["qty"] <= 100  # Never increase

    def test_negative_edge_skip(self):
        trade = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "qty": 50,
            "price": 2500.0,
            "order_type": "MARKET",
            "win_rate": 0.2,  # Very low win rate
            "avg_win": 0.02,
            "avg_loss": 0.10,  # High loss
            "conviction": 0.3,
        }
        result = _kelly_sized_qty(trade, available_cash=500000.0)
        assert result.get("kelly_skip") is True

    def test_drawdown_reduces_size(self):
        trade = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "qty": 100,
            "price": 2500.0,
            "order_type": "MARKET",
            "win_rate": 0.6,
            "avg_win": 0.05,
            "avg_loss": 0.03,
            "conviction": 0.7,
        }
        result_no_dd = _kelly_sized_qty(trade, available_cash=500000.0, current_drawdown=0.0)
        result_with_dd = _kelly_sized_qty(trade, available_cash=500000.0, current_drawdown=0.03)
        assert result_with_dd["qty"] <= result_no_dd["qty"]

    def test_zero_price_unchanged(self):
        trade = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "qty": 50,
            "price": 0.0,
            "order_type": "MARKET",
            "win_rate": 0.6,
            "avg_win": 0.05,
            "avg_loss": 0.03,
            "conviction": 0.7,
        }
        result = _kelly_sized_qty(trade, available_cash=500000.0)
        assert result["qty"] == 50

    def test_does_not_mutate_original(self):
        trade = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "qty": 100,
            "price": 2500.0,
            "order_type": "MARKET",
            "win_rate": 0.6,
            "avg_win": 0.05,
            "avg_loss": 0.03,
            "conviction": 0.7,
        }
        original_qty = trade["qty"]
        _kelly_sized_qty(trade, available_cash=500000.0)
        assert trade["qty"] == original_qty  # Original unchanged
