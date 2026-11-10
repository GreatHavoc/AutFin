"""Tests for screening_toolset.py — pure functions (no network)."""
from __future__ import annotations

import pytest

from autfin.tools.screening import (
    NIFTY_500_SYMBOLS,
    liquidity_filter,
    indian_red_flags,
    magic_formula_rank,
    apply_sector_rotation,
)


# ── NIFTY_500_SYMBOLS ────────────────────────────────────────────────


def test_nifty500_is_nonempty():
    assert len(NIFTY_500_SYMBOLS) > 50


def test_nifty500_contains_reliance():
    assert "RELIANCE" in NIFTY_500_SYMBOLS


# ── liquidity_filter ──────────────────────────────────────────────────


def test_liquidity_filter_removes_low_volume():
    stocks = [
        {"market_cap": 10_000_000_000, "avg_volume": 50_000_000, "symbol": "A"},
        {"market_cap": 100_000_000, "avg_volume": 100_000, "symbol": "B"},  # too small
    ]
    result = liquidity_filter(stocks)
    assert len(result) == 1
    assert result[0]["symbol"] == "A"


def test_liquidity_filter_empty():
    assert liquidity_filter([]) == []


# ── indian_red_flags ──────────────────────────────────────────────────


def test_red_flag_high_pledge():
    flags = indian_red_flags({"promoter_pledge_pct": 80})
    assert len(flags) > 0
    assert "pledge" in flags[0].lower()


def test_red_flag_no_pledge():
    flags = indian_red_flags({"promoter_pledge_pct": 5})
    assert len(flags) == 0


def test_red_flag_missing_fields():
    flags = indian_red_flags({})
    assert len(flags) == 0


# ── magic_formula_rank ────────────────────────────────────────────────


def test_magic_formula_rank_basic():
    stocks = [
        {"symbol": "A", "earnings_yield": 0.15, "roic": 0.20},
        {"symbol": "B", "earnings_yield": 0.05, "roic": 0.10},
    ]
    result = magic_formula_rank(stocks)
    assert len(result) == 2
    # A should rank higher
    assert result[0]["symbol"] == "A"


def test_magic_formula_rank_empty():
    assert magic_formula_rank([]) == []


# ── apply_sector_rotation ────────────────────────────────────────────


def test_sector_rotation_preserves_count():
    stocks = [
        {"symbol": "RELIANCE", "composite_score": 0.8},
        {"symbol": "TCS", "composite_score": 0.3},
    ]
    biases = {"Energy": 0.5, "Technology": -0.2}
    result = apply_sector_rotation(stocks, biases)
    assert len(result) == 2
    # Reliance (Energy, bullish bias) should rank higher
    assert result[0]["symbol"] == "RELIANCE"


def test_sector_rotation_empty():
    assert apply_sector_rotation([], {}) == []


# ── Screening toolset creation ────────────────────────────────────────


def test_screening_toolset_creation():
    from autfin.tools.screening_toolset import create_screening_toolset
    ts = create_screening_toolset()
    assert ts is not None
