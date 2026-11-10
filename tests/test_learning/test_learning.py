"""Tests for the learning engine (calibration, retrieval, feedback loop).

All DB work is redirected to tmp_path so the real data/autfin.db is untouched.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def journal_db(tmp_path, monkeypatch):
    """Point the learning journal at a throwaway SQLite file."""
    from autfin.memory import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "autfin.db")
    database.init_database()
    return database.DB_PATH


async def _seed(db, rows: list[tuple]) -> None:
    from autfin.memory.database import get_connection

    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, "
            "thesis_summary, outcome, sector) VALUES (?, ?, ?, ?, ?, 'completed', ?)",
            rows,
        )
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_record_then_calibrate_round_trip(journal_db):
    from autfin.learning import CalibrationTracker, record_learning

    assert await record_learning(
        "RELIANCE", "BUY", 0.85, 0.12, "Margin expansion", sector="Energy"
    )
    await record_learning("TCS", "BUY", 0.85, -0.03, "Deal flow slowing", sector="IT")

    calibration = await CalibrationTracker().get_calibration()
    assert set(calibration) == {"high_conv"}
    assert calibration["high_conv"]["sample_size"] == 2
    assert calibration["high_conv"]["actual_win_rate"] == 0.5
    assert calibration["high_conv"]["delta"] == pytest.approx(0.5 - 0.75, abs=1e-3)


@pytest.mark.asyncio
async def test_calibration_can_make_the_agent_more_cautious(journal_db):
    """A tier that historically underperforms its conviction must be pulled down."""
    from autfin.learning import CalibrationTracker, record_learning

    for _ in range(3):
        await record_learning("X", "BUY", 0.85, -0.05, "Thesis", sector="IT")

    result = await CalibrationTracker().suggested_conviction_adjustment(0.85)
    assert result["tier"] == "high_conv"
    assert result["calibration_delta"] < 0
    assert result["adjusted_conviction"] < 0.85


@pytest.mark.asyncio
async def test_patterns_and_retrieval_return_rows(journal_db):
    from autfin.learning import detect_patterns, record_learning, retrieve_learnings

    await record_learning("RELIANCE", "BUY", 0.8, 0.10, "Margin expansion in petchem", sector="Energy")
    await record_learning("INFY", "BUY", 0.8, -0.04, "Margin pressure from wage hikes", sector="IT")

    patterns = await detect_patterns(min_samples=1)
    assert any(p["type"] == "sector_performance" for p in patterns)

    result = await retrieve_learnings("margin issues")
    assert len(result["similar_setups"]) == 2
    assert result["calibration"]["high_conv"]["sample_size"] == 2


@pytest.mark.asyncio
async def test_pre_trade_and_post_trade_share_one_written_view(journal_db):
    from autfin.learning import post_trade_learn, pre_trade_context

    before = await pre_trade_context("RELIANCE", "BUY", sector="Energy", raw_conviction=0.5)
    assert before["calibration"] == {}
    assert before["conviction_adjustment"]["sample_size"] == 0

    after = await post_trade_learn(
        "RELIANCE", "SELL", 0.5, -0.06, "Thesis invalidated", sector="Energy"
    )
    assert after["recorded"] is True
    assert after["updated_calibration"]["medium_conv"]["actual_win_rate"] == 0.0

    again = await pre_trade_context("RELIANCE", "BUY", sector="Energy", raw_conviction=0.5)
    assert again["conviction_adjustment"]["sample_size"] == 1
    assert again["conviction_adjustment"]["adjusted_conviction"] < 0.5


@pytest.mark.asyncio
async def test_corrupt_database_is_quarantined_not_handed_out(journal_db, monkeypatch, tmp_path):
    """The learning engine silently returned nothing when the DB file was bad."""
    from autfin.learning import record_learning
    from autfin.memory import database

    journal_db.write_bytes(b"this is not a sqlite database")

    conn = database.get_connection()
    try:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute(
            "SELECT COUNT(*) FROM learning_journal"
        ).fetchone()[0] == 0
    finally:
        conn.close()

    # The unreadable file is renamed, never deleted — it is the only copy.
    quarantined = list(tmp_path.glob("autfin.db.malformed-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"this is not a sqlite database"
    assert journal_db.exists()

    # And writes keep working after recovery.
    assert await record_learning("A", "BUY", 0.5, 0.01, "t")


def test_brier_score_is_the_mean_squared_error():
    from autfin.learning.online import OnlineLearner

    learner = OnlineLearner()
    learner.update(conviction=0.9, outcome=0.0)
    learner.update(conviction=0.9, outcome=0.0)

    bucket = learner.get_bucket(0.9)
    # mean((0.9-0.0)^2) = 0.81. The old formula divided sum(p*o) by n and
    # returned 0.0 — a perfectly miscalibrated strategy looked flawless.
    assert bucket.to_dict()["brier_score"] == pytest.approx(0.81)
    assert bucket.to_dict()["calibration_error"] == pytest.approx(0.9)


def test_online_learner_pulls_conviction_toward_reality():
    from autfin.learning.online import OnlineLearner

    learner = OnlineLearner()
    for _ in range(10):
        learner.update(conviction=0.9, outcome=0.0)

    assert learner.get_adjustment(0.9) < 0.9
    report = learner.get_report()
    assert report["total_trades"] == 10
    assert report["win_rate"] == 0.0
    assert report["buckets"]["very_high"]["brier_score"] == pytest.approx(0.81)


def test_online_learner_not_enough_samples_passes_through():
    from autfin.learning.online import OnlineLearner

    learner = OnlineLearner()
    learner.update(conviction=0.9, outcome=0.0)
    assert learner.get_adjustment(0.9) == 0.9


def test_learning_package_exports_its_api():
    import autfin.learning as learning

    assert set(learning.__all__) >= {
        "record_learning",
        "retrieve_learnings",
        "pre_trade_context",
        "post_trade_learn",
        "get_learning_summary",
        "CalibrationTracker",
        "OnlineLearner",
        "ABTestingFramework",
    }
