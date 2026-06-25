"""Tests for learning/feedback_loop.py — orchestrator functions."""

import pytest
from unittest.mock import AsyncMock, patch
from src.autfin.learning.feedback_loop import (
    pre_trade_context,
    post_trade_learn,
    get_learning_summary,
    _gather,
)


# ---------------------------------------------------------------------------
# _gather
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gather_returns_results_in_order():
    async def a1():
        return "first"

    async def a2():
        return "second"

    result = await _gather(a1(), a2())
    assert result == ["first", "second"]


@pytest.mark.asyncio
async def test_gather_handles_exceptions():
    async def ok():
        return "ok"

    async def fail():
        raise ValueError("boom")

    result = await _gather(ok(), fail())
    assert result[0] == "ok"
    assert result[1] is None


# ---------------------------------------------------------------------------
# pre_trade_context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_trade_context_returns_structure():
    with patch("src.autfin.learning.retriever.search_similar_setups", new_callable=AsyncMock) as mock_search, \
         patch("src.autfin.learning.retriever.retrieve_learnings", new_callable=AsyncMock) as mock_ret, \
         patch("src.autfin.learning.calibrator.CalibrationTracker") as MockTracker:
        mock_search.return_value = [{"symbol": "RELIANCE"}]
        mock_ret.return_value = {"similar_setups": [{"symbol": "RELIANCE"}], "patterns": []}
        instance = AsyncMock()
        instance.get_calibration.return_value = {"high_conv": {"delta": -0.1}}
        instance.suggested_conviction_adjustment.return_value = {"adjusted_conviction": 0.7}
        MockTracker.return_value = instance

        result = await pre_trade_context("RELIANCE", "BUY", "ENERGY", 0.8)
        assert "similar_setups" in result
        assert "calibration" in result
        assert "conviction_adjustment" in result


@pytest.mark.asyncio
async def test_pre_trade_context_handles_error():
    with patch("src.autfin.learning.retriever.search_similar_setups", new_callable=AsyncMock, side_effect=Exception("fail")):
        result = await pre_trade_context("RELIANCE", "BUY")
        assert "error" in result
        assert result["similar_setups"] == []


# ---------------------------------------------------------------------------
# post_trade_learn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_trade_learn_records_and_returns():
    with patch("src.autfin.learning.calibrator.record_learning", new_callable=AsyncMock, return_value=True) as mock_rec, \
         patch("src.autfin.learning.calibrator.CalibrationTracker") as MockTracker, \
         patch("src.autfin.learning.calibrator.detect_patterns", new_callable=AsyncMock, return_value=[]) as mock_dp:
        instance = AsyncMock()
        instance.get_calibration.return_value = {"win_rate": 0.6}
        MockTracker.return_value = instance

        result = await post_trade_learn(
            symbol="TCS", action="SELL", conviction=0.6,
            pnl_pct=0.10, thesis_summary="profit target hit",
        )
        assert result["recorded"] is True
        assert "updated_calibration" in result
        mock_rec.assert_called_once()


@pytest.mark.asyncio
async def test_post_trade_learn_handles_error():
    with patch("src.autfin.learning.calibrator.record_learning", new_callable=AsyncMock, side_effect=Exception("fail")):
        result = await post_trade_learn(
            symbol="TCS", action="SELL", conviction=0.5,
            pnl_pct=-0.05, thesis_summary="stop loss",
        )
        assert result["recorded"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# get_learning_summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_learning_summary_returns_structure():
    with patch("src.autfin.learning.calibrator.CalibrationTracker") as MockTracker, \
         patch("src.autfin.learning.calibrator.detect_patterns", new_callable=AsyncMock, return_value=[]) as mock_dp, \
         patch("src.autfin.learning.retriever.retrieve_learnings", new_callable=AsyncMock) as mock_ret:
        instance = AsyncMock()
        instance.get_calibration.return_value = {"win_rate": 0.55}
        MockTracker.return_value = instance
        mock_ret.return_value = {"similar_setups": [{"symbol": "INFY"}]}

        result = await get_learning_summary()
        assert "calibration" in result
        assert "patterns" in result
        assert "recent_trades" in result
        assert len(result["recent_trades"]) == 1


@pytest.mark.asyncio
async def test_get_learning_summary_handles_error():
    with patch("src.autfin.learning.calibrator.CalibrationTracker", side_effect=Exception("fail")):
        result = await get_learning_summary()
        assert "error" in result
        assert result["calibration"] == {}
