"""Tests for the Nifty 500 universe module and sector rotation."""

import pytest
from src.autfin.tools.universe import (
    NIFTY_500,
    SECTOR_MAP,
    get_sector,
    get_sector_symbols,
    get_sector_weight_map,
)
from src.autfin.tools.screening import (
    apply_sector_rotation,
    _get_symbol_sector,
)


# ---------------------------------------------------------------------------
# NIFTY_500 sanity checks
# ---------------------------------------------------------------------------

def test_nifty_500_has_major_stocks():
    for sym in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"]:
        assert sym in NIFTY_500


def test_nifty_500_has_at_least_100_symbols():
    assert len(NIFTY_500) >= 100


def test_nifty_500_no_duplicates():
    assert len(NIFTY_500) == len(set(NIFTY_500))


# ---------------------------------------------------------------------------
# SECTOR_MAP tests
# ---------------------------------------------------------------------------

def test_sector_map_has_it_sector():
    assert "IT" in SECTOR_MAP
    assert "TCS" in SECTOR_MAP["IT"]
    assert "INFY" in SECTOR_MAP["IT"]


def test_sector_map_has_banking_sector():
    assert "BANKING" in SECTOR_MAP
    assert "HDFCBANK" in SECTOR_MAP["BANKING"]


def test_sector_map_covers_most_nifty_500():
    """Most NIFTY_500 symbols should be in at least one sector."""
    all_sector_symbols = set()
    for symbols in SECTOR_MAP.values():
        all_sector_symbols.update(symbols)
    uncovered = [s for s in NIFTY_500 if s not in all_sector_symbols]
    # Allow uncovered symbols (new listings, niche companies)
    assert len(uncovered) <= 20, f"Too many uncovered symbols: {uncovered}"


# ---------------------------------------------------------------------------
# get_sector / get_sector_symbols tests
# ---------------------------------------------------------------------------

def test_get_sector_returns_correct_sector():
    assert get_sector("TCS") == "IT"
    assert get_sector("HDFCBANK") == "BANKING"
    assert get_sector("SUNPHARMA") == "PHARMA"


def test_get_sector_returns_unknown_for_invalid():
    assert get_sector("FAKESTOCK") == "UNKNOWN"


def test_get_sector_is_case_insensitive():
    assert get_sector("tcs") == "IT"
    assert get_sector("hdfcbank") == "BANKING"


def test_get_sector_symbols_returns_list():
    symbols = get_sector_symbols("IT")
    assert isinstance(symbols, list)
    assert "TCS" in symbols
    assert len(symbols) >= 5


def test_get_sector_symbols_empty_for_unknown():
    symbols = get_sector_symbols("FAKESECTOR")
    assert symbols == []


# ---------------------------------------------------------------------------
# get_sector_weight_map tests
# ---------------------------------------------------------------------------

def test_sector_weight_map_returns_all_symbols():
    from src.autfin.tools.universe import NIFTY_500
    weights = get_sector_weight_map({"IT": 0.5, "BANKING": -0.3})
    assert len(weights) == len(NIFTY_500)


def test_sector_weight_map_bullish_sector_gets_higher_weight():
    weights = get_sector_weight_map({"IT": 1.0})
    tcs_weight = weights.get("TCS", 0)
    reliance_weight = weights.get("RELIANCE", 0)
    # TCS is IT, RELIANCE is ENERGY — IT should be higher with +1.0 bias
    assert tcs_weight > reliance_weight


def test_sector_weight_map_bearish_sector_gets_lower_weight():
    weights = get_sector_weight_map({"IT": -1.0})
    tcs_weight = weights.get("TCS", 0)
    reliance_weight = weights.get("RELIANCE", 0)
    assert tcs_weight < reliance_weight


def test_sector_weight_map_neutral_has_no_effect():
    weights = get_sector_weight_map({})
    # With no biases, all weights should be 1.0 (base weight)
    for sym in NIFTY_500:
        assert weights[sym] == pytest.approx(1.0, abs=0.01)


def test_sector_weight_map_floor_at_0_1():
    weights = get_sector_weight_map({"IT": -10.0})  # Extreme bearish
    for sym in NIFTY_500:
        assert weights[sym] >= 0.1


# ---------------------------------------------------------------------------
# _get_symbol_sector (screening module) tests
# ---------------------------------------------------------------------------

def test_get_symbol_sector_screening():
    assert _get_symbol_sector("TCS") == "IT"
    assert _get_symbol_sector("HDFCBANK") == "BANKING"
    assert _get_symbol_sector("FAKE") == "OTHER"


# ---------------------------------------------------------------------------
# apply_sector_rotation tests
# ---------------------------------------------------------------------------

def test_apply_sector_rotation_bullish_it_boosts_tcs():
    candidates = [
        {"symbol": "TCS", "composite_score": 0.5},
        {"symbol": "RELIANCE", "composite_score": 0.5},
    ]
    rotated = apply_sector_rotation(candidates, {"IT": 0.8})
    # TCS (IT) should get boosted, RELIANCE (ENERGY) should stay neutral
    tcs = next(c for c in rotated if c["symbol"] == "TCS")
    reliance = next(c for c in rotated if c["symbol"] == "RELIANCE")
    assert tcs["adjusted_score"] > reliance["adjusted_score"]


def test_apply_sector_rotation_bearish_it_penalizes_tcs():
    candidates = [
        {"symbol": "TCS", "composite_score": 0.5},
        {"symbol": "RELIANCE", "composite_score": 0.5},
    ]
    rotated = apply_sector_rotation(candidates, {"IT": -0.8})
    tcs = next(c for c in rotated if c["symbol"] == "TCS")
    reliance = next(c for c in rotated if c["symbol"] == "RELIANCE")
    assert tcs["adjusted_score"] < reliance["adjusted_score"]


def test_apply_sector_rotation_empty_candidates():
    result = apply_sector_rotation([], {"IT": 0.5})
    assert result == []


def test_apply_sector_rotation_adds_sector_metadata():
    candidates = [{"symbol": "TCS", "composite_score": 0.5}]
    rotated = apply_sector_rotation(candidates, {"IT": 0.5})
    assert rotated[0]["sector"] == "IT"
    assert "sector_bias" in rotated[0]
    assert "adjusted_score" in rotated[0]


def test_apply_sector_rotation_sorts_by_adjusted_score():
    candidates = [
        {"symbol": "TCS", "composite_score": 0.5},
        {"symbol": "RELIANCE", "composite_score": 0.5},
    ]
    # IT is bullish, ENERGY is neutral — TCS should overtake RELIANCE
    rotated = apply_sector_rotation(candidates, {"IT": 1.0})
    assert rotated[0]["symbol"] == "TCS"
    assert rotated[1]["symbol"] == "RELIANCE"
