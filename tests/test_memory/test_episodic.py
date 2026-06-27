"""Tests for episodic memory."""

import sqlite3
from pathlib import Path

import pytest

from src.autfin.memory.episodic import EpisodicMemory


@pytest.fixture
def mem(tmp_path):
    db = tmp_path / "test_episodic.db"
    return EpisodicMemory(db_path=str(db))


class TestRecordDecision:
    def test_records_decision(self, mem):
        did = mem.record_decision("RELIANCE", "BUY", 2500.0, 0.8, thesis="Bullish")
        assert did.startswith("EP-")
        results = mem.retrieve_similar("RELIANCE")
        assert len(results) == 1
        assert results[0]["action"] == "BUY"
        assert results[0]["entry_price"] == 2500.0

    def test_records_multiple(self, mem):
        mem.record_decision("RELIANCE", "BUY", 2500.0, 0.8)
        mem.record_decision("INFY", "BUY", 1500.0, 0.7)
        mem.record_decision("RELIANCE", "SELL", 2800.0, 0.6)
        assert len(mem.retrieve_similar("RELIANCE")) == 2
        assert len(mem.retrieve_similar("INFY")) == 1


class TestCloseDecision:
    def test_close_profit(self, mem):
        did = mem.record_decision("RELIANCE", "BUY", 2000.0, 0.8)
        success = mem.close_decision(did, 2500.0, lesson="Good entry")
        assert success is True
        results = mem.retrieve_similar("RELIANCE")
        assert results[0]["outcome"] == "profit"
        assert results[0]["pnl_pct"] == 25.0
        assert results[0]["lesson"] == "Good entry"

    def test_close_loss(self, mem):
        did = mem.record_decision("RELIANCE", "BUY", 2000.0, 0.8)
        success = mem.close_decision(did, 1500.0, lesson="Bad timing")
        assert success is True
        results = mem.retrieve_similar("RELIANCE")
        assert results[0]["outcome"] == "loss"
        assert results[0]["pnl_pct"] == -25.0

    def test_close_nonexistent(self, mem):
        success = mem.close_decision("EP-FAKE", 2500.0)
        assert success is False


class TestRetrieveSimilar:
    def test_filter_by_action(self, mem):
        mem.record_decision("RELIANCE", "BUY", 2000.0, 0.8)
        mem.record_decision("RELIANCE", "SELL", 2500.0, 0.6)
        buys = mem.retrieve_similar("RELIANCE", action="BUY")
        assert len(buys) == 1
        assert buys[0]["action"] == "BUY"

    def test_limit(self, mem):
        for i in range(10):
            mem.record_decision("RELIANCE", "BUY", 2000.0 + i, 0.5)
        results = mem.retrieve_similar("RELIANCE", limit=3)
        assert len(results) == 3


class TestGetLessons:
    def test_lessons_only(self, mem):
        did1 = mem.record_decision("RELIANCE", "BUY", 2000.0, 0.8)
        mem.close_decision(did1, 2500.0, lesson="Good entry")
        did2 = mem.record_decision("INFY", "BUY", 1500.0, 0.7)
        mem.close_decision(did2, 1200.0)  # No lesson
        lessons = mem.get_lessons()
        assert len(lessons) == 1
        assert lessons[0]["lesson"] == "Good entry"


class TestGetStats:
    def test_empty(self, mem):
        stats = mem.get_stats()
        assert stats["total"] == 0
        assert stats["win_rate"] == 0

    def test_with_data(self, mem):
        did1 = mem.record_decision("RELIANCE", "BUY", 2000.0, 0.8)
        mem.close_decision(did1, 2500.0)
        did2 = mem.record_decision("INFY", "BUY", 1500.0, 0.7)
        mem.close_decision(did2, 1200.0)
        stats = mem.get_stats()
        assert stats["total"] == 2
        assert stats["profit"] == 1
        assert stats["loss"] == 1
        assert stats["win_rate"] == 50.0
