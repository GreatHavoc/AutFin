"""Tests for the memory / llm / observability / settings layer."""
from __future__ import annotations

import pytest


# ── config/settings ─────────────────────────────────────────────────────────
def test_settings_and_model_config_agree():
    """Two config objects used to name different models for the same tier."""
    from autfin.config.settings import AutFinSettings
    from autfin.types import MODEL_CONFIG

    s = AutFinSettings()
    assert s.deep_model == MODEL_CONFIG.deep
    assert s.fast_model == MODEL_CONFIG.fast
    assert s.nvidia_deep_model == MODEL_CONFIG.deep


def test_settings_still_reads_env(tmp_path, monkeypatch):
    from autfin.config.settings import AutFinSettings

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEP_MODEL=env/model\nMAX_DAILY_TRADES=9\n", encoding="utf-8")

    s = AutFinSettings()
    assert s.deep_model == "env/model"
    assert s.max_daily_trades == 9


# ── llm ─────────────────────────────────────────────────────────────────────
def test_llm_package_imports_without_the_metadata_backport():
    """importlib_patch imported the importlib_metadata backport unconditionally;
    on Python 3.13 it is absent, and the whole llm package failed to import."""

    from autfin.llm import importlib_patch

    assert importlib_patch._backend is not None


def test_model_tier_registry_tracks_the_live_config():
    from autfin.llm.model_tiers import AGENT_TIERS, MODEL_REGISTRY, ModelTier
    from autfin.types import MODEL_CONFIG

    assert MODEL_REGISTRY[ModelTier.DEEP] == MODEL_CONFIG.reasoning
    assert MODEL_REGISTRY[ModelTier.MEDIUM] == MODEL_CONFIG.deep
    assert MODEL_REGISTRY[ModelTier.FAST] == MODEL_CONFIG.fast
    assert AGENT_TIERS  # non-empty, else every agent falls back to a default


def test_tool_param_rules_cover_the_order_executor():
    from autfin.llm.validation import TOOL_PARAM_RULES

    rules = TOOL_PARAM_RULES["execute_order"]
    assert rules["quantity"]["max"] == 100000
    assert rules["action"]["allowed"] == ["BUY", "SELL"]


# ── observability ───────────────────────────────────────────────────────────
def test_observability_imports_and_records():
    from autfin.observability import (
        ObservabilityMetrics,
        get_metrics_snapshot,
        is_langsmith_configured,
        reset_tracer,
    )

    metrics = ObservabilityMetrics()
    metrics.record_agent_call("risk", 42.0, True)
    metrics.record_trade(0.05, 0.7)
    metrics.record_learning_quality(
        {"high_conv": {"delta": 0.1, "sample_size": 3}}, [{"pattern": "x"}]
    )

    snapshot = get_metrics_snapshot()
    assert set(snapshot) >= {
        "agent_metrics",
        "llm",
        "traces",
        "learning",
        "graphs",
        "last_updated",
    }

    # LangSmith is not a dependency; the handler must degrade, not raise.
    assert is_langsmith_configured() is False
    reset_tracer()


def test_roro_works_on_a_database_that_has_no_research_output(tmp_path, monkeypatch):
    """compute_roro() used to warn "no such table" and return zeros on any DB
    that had never recorded research — indistinguishable from useless research."""
    from autfin.observability import roro

    db = tmp_path / "autfin.db"
    monkeypatch.setattr(roro, "DB_PATH", db)

    fresh = roro.compute_roro(30)
    assert fresh["interpretation"] == "Insufficient data"

    roro.record_research_output(
        symbol="RELIANCE",
        depth="deep",
        decision_id="d1",
        scores={"fundamentals": 0.8, "technical": 0.6},
    )
    after = roro.compute_roro(30)
    assert after["total_decisions"] == 1
    assert after["by_depth"]["deep"]["count"] == 1


# ── memory ──────────────────────────────────────────────────────────────────
def test_learning_journal_round_trip_and_win_rate(tmp_path, monkeypatch):
    """wins were counted as `outcome == 'completed'`, so a losing closed trade
    was a 'win' and win_rate was inflated."""
    from autfin.memory import database, performance

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "autfin.db")
    database.init_database()

    performance.record_learning("RELIANCE", "BUY", 0.8, 0.05, "good thesis", outcome="completed")
    performance.record_learning("TCS", "BUY", 0.5, -0.04, "bad thesis", outcome="completed")

    stats = performance.get_win_rate()
    assert stats["total_trades"] == 2
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["win_rate"] == 0.5
    symbols = [row["symbol"] for row in performance.get_symbol_performance()]
    assert symbols == ["RELIANCE", "TCS"]


def test_vector_store_builds_with_chroma(tmp_path):
    from autfin.memory.vector_store import VectorStore, get_chroma_path

    assert get_chroma_path().name == "chroma"

    store = VectorStore(mode="in-memory", collection_names=["market_data"])
    assert store.list_collections() == ["market_data"]
    ids = store.add(
        "market_data",
        embeddings=[[0.0] * 384, [1.0] * 384],
        metadatas=[{"symbol": "RELIANCE"}, {"symbol": "TCS"}],
    )
    assert len(ids) == 2
    assert store.get_collection("market_data") is not None


def test_embeddings_are_stable_and_comparable():
    from autfin.memory.embeddings import cosine_similarity, generate_embedding

    a = generate_embedding("margin expansion")
    assert len(a) == 384
    assert cosine_similarity(a, generate_embedding("margin expansion")) == pytest.approx(1.0)
    assert cosine_similarity(a, generate_embedding("totally unrelated text")) < 1.0


def test_connection_pool_reuses_and_recovers(tmp_path, monkeypatch):
    from autfin.memory import database
    from autfin.memory.connection_pool import get_db_pool

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "autfin.db")
    database.init_database()

    pool = get_db_pool(tmp_path / "autfin.db")
    with pool.connection() as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
    with pool.connection() as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1
