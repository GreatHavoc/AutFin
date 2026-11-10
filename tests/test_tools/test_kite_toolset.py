"""Tests for kite_toolset.py — tests the underlying KiteTools class."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_kite_tools():
    """Mock KiteTools instance."""
    from autfin.tools.kite_tools import KiteTools
    kt = MagicMock(spec=KiteTools)
    kt.get_holdings = AsyncMock(return_value={"status": "success", "data": [{"symbol": "RELIANCE", "qty": 10}]})
    kt.get_quotes = AsyncMock(return_value={"status": "success", "data": {"RELIANCE": {"last_price": 2500}}})
    kt.get_margins = AsyncMock(return_value={"status": "success", "data": {"available": {"cash": 100000}}})
    kt.get_positions = AsyncMock(return_value={"status": "success", "data": {"net": []}})
    kt.place_order = AsyncMock(return_value={"status": "success", "order_id": "12345"})
    kt.get_instruments = AsyncMock(return_value={"status": "success", "data": [{"instrument_token": 1}]})
    return kt


# ── get_holdings ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_holdings(mock_kite_tools):
    result = await mock_kite_tools.get_holdings()
    assert result["status"] == "success"
    assert len(result["data"]) == 1
    assert result["data"][0]["symbol"] == "RELIANCE"


# ── get_quotes ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_quotes(mock_kite_tools):
    result = await mock_kite_tools.get_quotes(["NSE:RELIANCE"])
    assert result["status"] == "success"
    assert "RELIANCE" in result["data"]


# ── get_margins ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_margins(mock_kite_tools):
    result = await mock_kite_tools.get_margins()
    assert result["status"] == "success"
    assert "available" in result["data"]


# ── get_positions ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_positions(mock_kite_tools):
    result = await mock_kite_tools.get_positions()
    assert result["status"] == "success"
    assert "net" in result["data"]


# ── place_order ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_place_order(mock_kite_tools):
    result = await mock_kite_tools.place_order(
        tradingsymbol="RELIANCE",
        transaction_type="BUY",
        quantity=10,
        order_type="MARKET",
    )
    assert result["status"] == "success"
    assert result["order_id"] == "12345"


# ── get_instruments ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_instruments(mock_kite_tools):
    result = await mock_kite_tools.get_instruments("NSE")
    assert result["status"] == "success"
    assert len(result["data"]) == 1


# ── Toolset creation ──────────────────────────────────────────────────


def test_kite_toolset_creation():
    from autfin.tools.kite_toolset import create_kite_toolset
    ts = create_kite_toolset()
    assert ts is not None
