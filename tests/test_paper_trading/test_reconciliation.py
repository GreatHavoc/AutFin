"""Tests for paper-live reconciliation."""

import pytest

from src.autfin.paper_trading.reconciliation import reconcile


class TestReconcile:
    def test_empty_both(self):
        result = reconcile([], [])
        assert result["matches"] == 0
        assert result["discrepancies"] == 0

    def test_perfect_match(self):
        paper = [{"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0}]
        live = [{"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0}]
        result = reconcile(paper, live)
        assert result["matches"] == 1
        assert result["discrepancies"] == 0
        assert result["match_rate"] == 100.0

    def test_quantity_mismatch(self):
        paper = [{"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0}]
        live = [{"tradingsymbol": "RELIANCE", "qty": 50, "avg_price": 2500.0}]
        result = reconcile(paper, live)
        assert result["matches"] == 0
        assert result["discrepancies"] == 1
        assert result["discrepancy_details"][0]["type"] == "quantity_mismatch"
        assert result["needs_attention"] is True

    def test_price_drift(self):
        paper = [{"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0}]
        live = [{"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2600.0}]
        result = reconcile(paper, live, tolerance_pct=0.01)
        assert result["discrepancies"] == 1
        assert result["discrepancy_details"][0]["type"] == "price_drift"

    def test_paper_only(self):
        paper = [{"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0}]
        result = reconcile(paper, [])
        assert result["discrepancies"] == 1
        assert result["discrepancy_details"][0]["type"] == "paper_only"

    def test_live_only(self):
        live = [{"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0}]
        result = reconcile([], live)
        assert result["discrepancies"] == 1
        assert result["discrepancy_details"][0]["type"] == "live_only"

    def test_mixed_scenario(self):
        paper = [
            {"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0},
            {"tradingsymbol": "INFY", "qty": 50, "avg_price": 1500.0},
        ]
        live = [
            {"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500.0},
            {"tradingsymbol": "TCS", "qty": 30, "avg_price": 3500.0},
        ]
        result = reconcile(paper, live)
        assert result["matches"] == 1  # RELIANCE
        assert result["discrepancies"] == 2  # INFY (paper only), TCS (live only)
