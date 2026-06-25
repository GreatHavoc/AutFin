"""Tests for learning/retriever.py — learning retrieval and search."""

import pytest
from unittest.mock import MagicMock, patch
from src.autfin.learning.retriever import (
    retrieve_learnings,
    search_similar_setups,
    _compute_calibration,
    _calibration_status,
    _detect_recent_patterns,
)


# ---------------------------------------------------------------------------
# _calibration_status
# ---------------------------------------------------------------------------

def test_calibration_status_overconfident():
    assert _calibration_status(0.3, 0.75) == "overconfident"


def test_calibration_status_calibrated():
    assert _calibration_status(0.7, 0.75) == "calibrated"


def test_calibration_status_underconfident():
    assert _calibration_status(0.9, 0.75) == "underconfident"


# ---------------------------------------------------------------------------
# _compute_calibration
# ---------------------------------------------------------------------------

def test_compute_calibration_empty_db():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (0, 0)
    result = _compute_calibration(mock_conn)
    assert result == {}


def test_compute_calibration_with_data():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.side_effect = [
        (10, 7),  # high_conv: 7 wins out of 10
        (0, 0),   # medium_conv: no data
        (0, 0),   # low_conv: no data
    ]
    result = _compute_calibration(mock_conn)
    assert "high_conv" in result
    assert result["high_conv"]["sample_size"] == 10
    assert result["high_conv"]["actual_win_rate"] == 0.7
    assert result["high_conv"]["status"] == "calibrated"


# ---------------------------------------------------------------------------
# _detect_recent_patterns
# ---------------------------------------------------------------------------

def test_detect_patterns_empty_db():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = (0,)
    result = _detect_recent_patterns(mock_conn)
    assert result == []


def test_detect_patterns_sector_concentration():
    mock_conn = MagicMock()
    loss_rows = [
        ("TCS", "IT", 0.8, -5.0, "thesis"),
        ("INFY", "IT", 0.7, -3.0, "thesis"),
        ("WIPRO", "IT", 0.6, -8.0, "thesis"),
        ("RELIANCE", "ENERGY", 0.5, -2.0, "thesis"),
    ]
    call_count = [0]
    def mock_execute(sql, *args):
        mock_result = MagicMock()
        if "FROM learning_journal WHERE pnl_pct < 0" in sql:
            mock_result.fetchall.return_value = loss_rows
        elif "COUNT(*) FROM learning_journal WHERE outcome" in sql and "pnl_pct" not in sql:
            mock_result.fetchone.return_value = (20,)
        elif "pnl_pct > 0" in sql:
            mock_result.fetchone.return_value = (12,)
        else:
            mock_result.fetchall.return_value = []
            mock_result.fetchone.return_value = (0,)
        return mock_result
    mock_conn.execute = mock_execute
    result = _detect_recent_patterns(mock_conn)
    patterns = [p for p in result if p.get("severity") in ("HIGH", "CRITICAL")]
    assert any("IT" in p["pattern"] for p in patterns)


def test_detect_patterns_high_conviction_streak():
    mock_conn = MagicMock()
    loss_rows = [
        ("A", "SECT", 0.8, -5.0, "t"),
        ("B", "SECT", 0.9, -3.0, "t"),
        ("C", "SECT", 0.75, -8.0, "t"),
        ("D", "OTHER", 0.3, -2.0, "t"),
    ]
    def mock_execute(sql, *args):
        mock_result = MagicMock()
        if "FROM learning_journal WHERE pnl_pct < 0" in sql:
            mock_result.fetchall.return_value = loss_rows
        elif "COUNT(*) FROM learning_journal WHERE outcome" in sql and "pnl_pct" not in sql:
            mock_result.fetchone.return_value = (20,)
        elif "pnl_pct > 0" in sql:
            mock_result.fetchone.return_value = (12,)
        else:
            mock_result.fetchall.return_value = []
            mock_result.fetchone.return_value = (0,)
        return mock_result
    mock_conn.execute = mock_execute
    result = _detect_recent_patterns(mock_conn)
    assert any("High-conviction" in p["pattern"] for p in result)


# ---------------------------------------------------------------------------
# retrieve_learnings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_learnings_returns_structure():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = (0,)

    with patch("src.autfin.memory.database.get_connection", return_value=mock_conn):
        result = await retrieve_learnings("RELIANCE buy")
        assert "similar_setups" in result
        assert "calibration" in result
        assert "patterns" in result


@pytest.mark.asyncio
async def test_retrieve_learnings_handles_error():
    with patch("src.autfin.memory.database.get_connection", side_effect=Exception("db fail")):
        result = await retrieve_learnings("test query")
        assert result["similar_setups"] == []
        assert "error" in result


# ---------------------------------------------------------------------------
# search_similar_setups
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_similar_setups_delegates():
    async def fake_ret(*a, **kw):
        return {"similar_setups": [{"symbol": "RELIANCE"}]}
    with patch("src.autfin.learning.retriever.retrieve_learnings", side_effect=fake_ret):
        result = await search_similar_setups("RELIANCE", "BUY")
        assert len(result) == 1
        assert result[0]["symbol"] == "RELIANCE"
