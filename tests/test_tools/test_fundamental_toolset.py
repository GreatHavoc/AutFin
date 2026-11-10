"""Tests for fundamental_toolset.py — live yfinance calls, no mocks."""
from __future__ import annotations

import pytest

from autfin.tools.fundamental_tools import FundamentalTools


@pytest.mark.asyncio
async def test_get_financials_reliance():
    ft = FundamentalTools()
    result = await ft.get_financials("RELIANCE")

    assert result["symbol"] == "RELIANCE"
    assert result["pe"] is not None
    assert result["market_cap"] is not None
    assert result["market_cap"] > 0


@pytest.mark.asyncio
async def test_get_financials_bad_symbol():
    ft = FundamentalTools()
    result = await ft.get_financials("ZZZZZZZZ")

    assert "error" in result or result.get("pe") is None


@pytest.mark.asyncio
async def test_get_ratios_reliance():
    ft = FundamentalTools()
    result = await ft.get_ratios("RELIANCE")

    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_health_score_reliance():
    ft = FundamentalTools()
    result = await ft.compute_health_score("RELIANCE")

    assert "health_score" in result
    assert 0 <= result["health_score"] <= 100
    assert result["rating"] in ("STRONG", "GOOD", "FAIR", "WEAK")


@pytest.mark.asyncio
async def test_compare_peers():
    ft = FundamentalTools()
    result = await ft.compare_peers("RELIANCE", ["ONGC"])

    assert "RELIANCE" in result["peers"]
    assert "ONGC" in result["peers"]
    assert result["peers"]["ONGC"]["pe"] is not None


def test_toolset_creation():
    from autfin.tools.fundamental_toolset import create_fundamental_toolset
    assert create_fundamental_toolset() is not None
