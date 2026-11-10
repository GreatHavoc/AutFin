"""Tests for the watchlist curator and the universe symbol/sector tables."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def curator():
    from autfin.universe import WatchlistCurator

    return WatchlistCurator()
@pytest.fixture(autouse=True)
def no_movers_scan(curator):
    # curate() always runs _find_top_movers, which scans NIFTY_500[:50] over
    # live yfinance; patch it away unless a test is about movers.
    with patch.object(curator, "_find_top_movers", return_value=[]):
        yield



@pytest.mark.asyncio
async def test_watchlist_from_holdings_and_manual(curator):
    """`curate()` raised AttributeError on a numeric impact_score, so the whole
    builder died before returning anything."""
    state = {
        "holdings": [{"tradingsymbol": "RELIANCE"}, {"symbol": "tcs.NS"}],
        "candidates": ["INFY"],
    }

    assert await curator.curate(state) == ["INFY", "RELIANCE", "TCS"]


@pytest.mark.asyncio
async def test_watchlist_survives_numeric_macro_signal(curator):
    state = {"holdings": [{"tradingsymbol": "SBIN"}], "macro_signals": {"impact_score": 0.9}}

    assert await curator.curate(state) == ["SBIN"]


@pytest.mark.asyncio
async def test_watchlist_survives_string_macro_signal(curator):
    state = {
        "holdings": [{"tradingsymbol": "SBIN"}],
        "macro_signals": {"impact_score": "high_negative"},
    }

    assert await curator.curate(state) == ["SBIN"]


@pytest.mark.asyncio
async def test_alert_symbol_joins_the_watchlist(curator):
    assert await curator.curate({"holdings": [], "alert_symbol": "hdfcbank"}) == ["HDFCBANK"]


@pytest.mark.asyncio
async def test_top_movers_are_merged_when_they_scan(curator):
    with patch.object(curator, "_find_top_movers", return_value=["VEDL"]):
        result = await curator.curate({"holdings": []})

    assert result == ["VEDL"]


@pytest.mark.asyncio
async def test_pre_earnings_is_called_with_the_watchlist(curator):
    """The call passed one argument to a zero-arg method, so the pre-earnings
    step raised TypeError every run and was swallowed by its own except."""
    with patch.object(
        curator,
        "_find_pre_earnings",
        side_effect=lambda watchlist: watchlist[:1],
    ) as spy:
        result = await curator.curate({"holdings": [], "candidates": ["MARUTI", "TITAN"]})

    assert spy.await_count == 1
    # placeholder pre-earnings returns everything it is handed, unchanged
    assert result == ["MARUTI", "TITAN"]


@pytest.mark.asyncio
async def test_empty_state_yields_empty_watchlist(curator):
    assert await curator.curate({}) == []


def test_universe_tables_are_consistent():
    from autfin.tools.universe import NIFTY_500, get_sector, get_sector_symbols

    assert NIFTY_500, "NIFTY_500 must not be empty"
    assert len(NIFTY_500) == len(set(NIFTY_500)), "no duplicates"
    assert NIFTY_500 == sorted(NIFTY_500)

    assert get_sector("TCS") == "IT"
    assert get_sector("NOTASYMBOL") == "UNKNOWN"
    assert "TCS" in get_sector_symbols("IT")
    assert get_sector_symbols("nope") == []


def test_sector_map_only_references_known_symbols():
    from autfin.tools.universe import NIFTY_500, SECTOR_MAP

    known = set(NIFTY_500)
    for sector, symbols in SECTOR_MAP.items():
        # A sector entry pointing at a symbol outside the investable universe
        # would silently drop out of every rotation overlay.
        assert symbols, f"{sector} is empty"
        assert set(symbols) <= known, f"{sector} references unknown symbols"


def test_sector_weight_map_respects_bias_and_floor():
    from autfin.tools.universe import get_sector_weight_map

    weights = get_sector_weight_map({"IT": 1.0, "BANKING": -1.0})

    assert weights["TCS"] == pytest.approx(1.3)  # +30% max adjustment
    assert weights["HDFCBANK"] == pytest.approx(0.7)
    assert all(w >= 0.1 for w in weights.values())


def test_universe_package_exports():
    import autfin.universe as universe

    assert set(universe.__all__) == {"WatchlistCurator", "curate_watchlist"}
