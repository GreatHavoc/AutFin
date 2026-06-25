"""Tests for learning/calibrator.py — conviction calibration, pattern detection, record_learning."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.autfin.learning.calibrator import (
    CalibrationTracker,
    calibrate_conviction,
    detect_patterns,
    record_learning,
)


# ---------------------------------------------------------------------------
# record_learning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_learning_inserts_into_db():
    mock_conn = MagicMock()
    with patch("src.autfin.memory.database.get_connection", return_value=mock_conn):
        result = await record_learning(
            symbol="RELIANCE",
            action="BUY",
            conviction=0.8,
            pnl_pct=0.05,
            thesis_summary="Strong momentum",
            outcome="completed",
            sector="ENERGY",
        )
        assert result is True
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


@pytest.mark.asyncio
async def test_record_learning_handles_db_error():
    with patch("src.autfin.memory.database.get_connection", side_effect=Exception("db fail")):
        result = await record_learning(
            symbol="TCS", action="BUY", conviction=0.6,
            pnl_pct=-0.02, thesis_summary="test",
        )
        assert result is False


# ---------------------------------------------------------------------------
# CalibrationTracker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_calibration_returns_empty_on_error():
    tracker = CalibrationTracker()
    with patch("src.autfin.memory.database.get_connection", side_effect=Exception("no db")):
        result = await tracker.get_calibration()
        assert result == {}


@pytest.mark.asyncio
async def test_suggested_conviction_adjustment_returns_valid_structure():
    tracker = CalibrationTracker()
    with patch.object(tracker, "get_calibration", new_callable=AsyncMock) as mock_cal:
        mock_cal.return_value = {
            "high_conv": {"delta": -0.1, "status": "overconfident", "sample_size": 20},
        }
        result = await tracker.suggested_conviction_adjustment(0.8, "IT")
        assert "raw_conviction" in result
        assert "adjusted_conviction" in result
        assert result["tier"] == "high_conv"
        assert result["calibration_delta"] == -0.1


@pytest.mark.asyncio
async def test_suggested_conviction_adjustment_no_data():
    tracker = CalibrationTracker()
    with patch.object(tracker, "get_calibration", new_callable=AsyncMock, return_value={}):
        result = await tracker.suggested_conviction_adjustment(0.5)
        assert result["adjusted_conviction"] == 0.5
        assert result["calibration_delta"] == 0


# ---------------------------------------------------------------------------
# calibrate_conviction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calibrate_conviction_returns_float():
    with patch("src.autfin.learning.calibrator.CalibrationTracker") as MockTracker:
        instance = AsyncMock()
        instance.suggested_conviction_adjustment.return_value = {
            "adjusted_conviction": 0.65,
        }
        MockTracker.return_value = instance
        result = await calibrate_conviction(0.7, "IT")
        assert result == 0.65


# ---------------------------------------------------------------------------
# detect_patterns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_patterns_returns_list():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    with patch("src.autfin.memory.database.get_connection", return_value=mock_conn):
        result = await detect_patterns(min_samples=5)
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_detect_patterns_handles_db_error():
    with patch("src.autfin.memory.database.get_connection", side_effect=Exception("db fail")):
        result = await detect_patterns()
        assert result == []


@pytest.mark.asyncio
async def test_detect_patterns_sector_performance():
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = [
        MagicMock(fetchall=MagicMock(return_value=[("IT", 15, 10)])),  # sector rows
        MagicMock(fetchall=MagicMock(return_value=[])),  # dow rows
    ]
    with patch("src.autfin.memory.database.get_connection", return_value=mock_conn):
        result = await detect_patterns(min_samples=5)
        assert len(result) == 1
        assert result[0]["type"] == "sector_performance"
        assert result[0]["sector"] == "IT"
        assert result[0]["win_rate"] == pytest.approx(0.667, abs=0.01)
        assert result[0]["assessment"] == "STRONG"
