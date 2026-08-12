"""Tests for Kite Tools integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import src.autfin.tools.kite_tools as _kt


@pytest.fixture(autouse=True)
def _patch_kiteconnect_module():
    """Ensure KITECONNECT_AVAILABLE is True regardless of test ordering."""
    _kt.KITECONNECT_AVAILABLE = True
    if not hasattr(_kt, "KiteConnect") or _kt.KiteConnect is None:
        _kt.KiteConnect = MagicMock
    yield



@pytest.fixture
def kite_tools(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "test_key")
    monkeypatch.setenv("KITE_API_SECRET", "test_secret")
    from src.autfin.tools.kite_tools import KiteTools

    kt = KiteTools(api_key="test_key", api_secret="test_secret")
    return kt


def _mock_ensure_auth(monkeypatch, kite_tools_inst):
    """Helper to mock _ensure_auth to return a MagicMock kite instance."""
    mock_kite = MagicMock()
    monkeypatch.setattr(kite_tools_inst, "_ensure_auth", lambda: mock_kite)
    return mock_kite


@pytest.mark.asyncio
async def test_get_holdings_returns_data(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.holdings.return_value = [
        {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2400.0}
    ]

    result = await kite_tools.get_holdings()

    assert result["status"] == "success"
    assert len(result["data"]) == 1
    assert result["data"][0]["tradingsymbol"] == "RELIANCE"
    mock_kite.holdings.assert_called_once()


@pytest.mark.asyncio
async def test_get_positions_returns_data(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.positions.return_value = {
        "day": [{"tradingsymbol": "TCS", "quantity": 5}],
        "net": [{"tradingsymbol": "TCS", "quantity": 5}],
    }

    result = await kite_tools.get_positions()

    assert result["status"] == "success"
    assert "day" in result["data"]
    assert result["data"]["day"][0]["tradingsymbol"] == "TCS"
    mock_kite.positions.assert_called_once()


@pytest.mark.asyncio
async def test_get_margins_returns_data(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.margins.return_value = {
        "equity": {"available": {"live_balance": 100000}},
        "commodity": {"available": {"live_balance": 0}},
    }

    result = await kite_tools.get_margins()

    assert result["status"] == "success"
    assert "equity" in result["data"]
    assert result["data"]["equity"]["available"]["live_balance"] == 100000
    mock_kite.margins.assert_called_once()


@pytest.mark.asyncio
async def test_place_order_success(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.place_order.return_value = "order_abc123"

    result = await kite_tools.place_order(
        variety="regular",
        exchange="NSE",
        tradingsymbol="RELIANCE",
        transaction_type="BUY",
        quantity=1,
        product="MIS",
        order_type="MARKET",
    )

    assert result["status"] == "success"
    assert result["order_id"] == "order_abc123"
    mock_kite.place_order.assert_called_once_with(
        variety="regular",
        exchange="NSE",
        tradingsymbol="RELIANCE",
        transaction_type="BUY",
        quantity=1,
        product="MIS",
        order_type="MARKET",
        price=None,
        trigger_price=None,
        validity="DAY",
        disclosed_quantity=None,
        tag="",
    )


@pytest.mark.asyncio
async def test_place_order_error(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.place_order.side_effect = Exception("Insufficient margin")

    result = await kite_tools.place_order(
        variety="regular",
        exchange="NSE",
        tradingsymbol="RELIANCE",
        transaction_type="BUY",
        quantity=10000,
    )

    assert result["status"] == "error"
    assert "Insufficient margin" in result["message"]


@pytest.mark.asyncio
async def test_cancel_order_success(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.cancel_order.return_value = "order_xyz789"

    result = await kite_tools.cancel_order(variety="regular", order_id="order_abc123")

    assert result["status"] == "success"
    assert result["order_id"] == "order_xyz789"
    mock_kite.cancel_order.assert_called_once_with(variety="regular", order_id="order_abc123")


@pytest.mark.asyncio
async def test_cancel_order_error(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.cancel_order.side_effect = Exception("Order not found")

    result = await kite_tools.cancel_order(variety="regular", order_id="nonexistent")

    assert result["status"] == "error"
    assert "Order not found" in result["message"]


@pytest.mark.asyncio
async def test_get_ltp_returns_data(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.ltp.return_value = {
        "NSE:RELIANCE": {"last_price": 2500.50, "instrument_token": 738561}
    }

    result = await kite_tools.get_ltp(["NSE:RELIANCE"])

    assert result["status"] == "success"
    assert result["data"]["NSE:RELIANCE"]["last_price"] == 2500.50
    mock_kite.ltp.assert_called_once_with(["NSE:RELIANCE"])


@pytest.mark.asyncio
async def test_place_gtt_returns_trigger_id(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.place_gtt.return_value = {"trigger_id": 12345}

    result = await kite_tools.place_gtt(
        trigger_type="single",
        tradingsymbol="RELIANCE",
        exchange="NSE",
        trigger_values=[2400.0],
        last_price=2500.0,
        orders=[{"transaction_type": "SELL", "quantity": 1, "product": "CNC", "order_type": "LIMIT", "price": 2390}],
    )

    assert result["status"] == "success"
    assert result["trigger_id"] == 12345
    mock_kite.place_gtt.assert_called_once()


@pytest.mark.asyncio
async def test_get_gtts_returns_list(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.get_gtts.return_value = [
        {"id": 1, "status": "active", "trigger_type": "single"},
        {"id": 2, "status": "triggered", "trigger_type": "two-leg"},
    ]

    result = await kite_tools.get_gtts()

    assert result["status"] == "success"
    assert len(result["data"]) == 2
    assert result["data"][0]["status"] == "active"
    mock_kite.get_gtts.assert_called_once()


@pytest.mark.asyncio
async def test_delete_gtt_success(kite_tools, monkeypatch):
    mock_kite = _mock_ensure_auth(monkeypatch, kite_tools)
    mock_kite.delete_gtt.return_value = {"status": "cancelled"}

    result = await kite_tools.delete_gtt(12345)

    assert result["status"] == "success"
    assert result["data"]["status"] == "cancelled"
    mock_kite.delete_gtt.assert_called_once_with(12345)


@pytest.mark.asyncio
async def test_kite_not_available_raises_import_error(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "test_key")
    monkeypatch.setenv("KITE_API_SECRET", "test_secret")
    monkeypatch.setattr("src.autfin.tools.kite_tools.KITECONNECT_AVAILABLE", False)
    from src.autfin.tools.kite_tools import KiteTools

    kt = KiteTools(api_key="test_key", api_secret="test_secret")
    monkeypatch.setattr(kt, "_load_token", lambda: None)

    with pytest.raises(ImportError, match="kiteconnect package not installed"):
        kt._ensure_auth()


@pytest.mark.asyncio
async def test_auth_failure_returns_error(monkeypatch):
    monkeypatch.setenv("KITE_API_KEY", "test_key")
    monkeypatch.setenv("KITE_API_SECRET", "test_secret")
    monkeypatch.setattr("src.autfin.tools.kite_tools.KITECONNECT_AVAILABLE", True)
    from src.autfin.tools.kite_tools import KiteTools

    kt = KiteTools(api_key="test_key", api_secret="test_secret")
    monkeypatch.setattr(kt, "_load_token", lambda: None)

    with pytest.raises(PermissionError, match="Auto-login failed"):
        kt._ensure_auth()
