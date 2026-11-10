"""Tests for the memory / llm / observability / settings layer.

Several of these used to test hand-rolled duplicates of framework features and
now assert the framework replacement instead: RunUsage/Instrumentation for the
metrics recorder and tracer, and pydantic-evals' native API for the eval runner.
"""
from __future__ import annotations

import pytest


# ── config/settings + types.MODEL_CONFIG ───────────────────────────────────
def test_model_config_is_the_single_tier_table():
    """settings.py's four model fields and llm/model_tiers' registry were
    duplicates that disagreed twice; both are gone."""
    from autfin.config.settings import AutFinSettings
    from autfin.types import MODEL_CONFIG

    settings = AutFinSettings()
    assert not any(
        hasattr(settings, f)
        for f in ("deep_model", "fast_model", "nvidia_model", "nvidia_deep_model")
    )


def test_model_config_reads_the_documented_env_names(monkeypatch):
    from autfin.types import ModelConfig

    monkeypatch.setenv("DEEP_MODEL", "env/deep-model")
    monkeypatch.setenv("FAST_MODEL", "env/fast-model")

    cfg = ModelConfig()
    assert cfg.deep == "env/deep-model"
    assert cfg.fast == "env/fast-model"
    assert cfg.reasoning == MODEL_CONFIG_default_reasoning()


def MODEL_CONFIG_default_reasoning() -> str:
    from autfin.types import ModelConfig

    return ModelConfig.model_fields["reasoning"].default


def test_model_tier_registry_derives_from_the_single_table():
    from autfin.llm.model_tiers import AGENT_TIERS, MODEL_REGISTRY, ModelTier
    from autfin.types import MODEL_CONFIG

    assert MODEL_REGISTRY[ModelTier.DEEP] == MODEL_CONFIG.reasoning
    assert MODEL_REGISTRY[ModelTier.MEDIUM] == MODEL_CONFIG.deep
    assert MODEL_REGISTRY[ModelTier.FAST] == MODEL_CONFIG.fast
    assert AGENT_TIERS  # non-empty, else every agent falls back to a default


def test_tool_arguments_are_validated_by_the_framework_not_a_rule_table():
    """llm/validation.py's TOOL_PARAM_RULES duplicated Pydantic schema
    validation; the module is deleted and the signature is the contract."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("autfin.llm.validation")

    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    def place_order(symbol: str, quantity: int) -> str:
        """Place an order."""
        return "ok"

    model = TestModel(call_tools=["place_order"])
    agent = Agent(model, tools=[place_order])
    agent.run_sync("Place an order.")

    tool = model.last_model_request_parameters.function_tools[0]
    assert tool.name == "place_order"
    assert sorted(tool.parameters_json_schema["properties"]) == ["quantity", "symbol"]
    assert sorted(tool.parameters_json_schema["required"]) == ["quantity", "symbol"]


# ── llm ─────────────────────────────────────────────────────────────────────
def test_llm_package_imports_without_the_metadata_backport():
    """importlib_patch imported the importlib_metadata backport unconditionally;
    on Python 3.13 it is absent, and the whole llm package failed to import."""

    from autfin.llm import importlib_patch

    assert importlib_patch._backend is not None


# ── observability ───────────────────────────────────────────────────────────
def test_observability_reduced_to_langsmith_config():
    """metrics.py / tracer.py / roro.py duplicated RunUsage, Instrumentation and
    an eval; they are deleted and only the LangSmith wrapper remains."""
    from autfin.observability import get_callback_handler, is_langsmith_configured

    assert is_langsmith_configured() is False  # no LangSmith key in this env
    assert get_callback_handler() is None  # degrades, does not raise


def test_run_usage_carries_the_cost_and_token_accounting():
    """What the deleted ObservabilityMetrics used to compute by hand."""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel())
    result = agent.run_sync("hello")

    usage = result.usage
    assert usage.requests >= 1
    assert hasattr(usage, "cost") and hasattr(usage, "tool_calls")
    assert isinstance(usage.tool_calls, int)


def test_instrumentation_capability_is_available():
    """The replacement for the hand-rolled SQLite trace store."""
    from pydantic_ai.capabilities import Instrumentation

    agent_capability = Instrumentation()
    assert agent_capability is not None


# ── evals ───────────────────────────────────────────────────────────────────
def test_eval_runner_is_the_native_api_not_a_hand_rolled_loop():
    """run_evaluations() wrapped dataset.evaluate_sync(); the wrapper is gone
    and callers use the native API directly."""
    import logfire

    from autfin.evals import create_autfin_dataset

    logfire.configure(send_to_logfire=False)  # span evaluators require it

    dataset = create_autfin_dataset(include_llm_judges=False, include_span_eval=True)

    def task(inputs):
        return {"status": "success"}

    report = dataset.evaluate_sync(task)
    assert len(report.cases) == len(dataset.cases)
    assert report.averages  # scores were computed, not skipped


def test_llm_judge_and_geval_evaluators_exist():
    """The judge types the hand-written evaluator classes should use."""
    from pydantic_evals.evaluators import GEval, LLMJudge

    assert LLMJudge is not None and GEval is not None


def test_validation_gates_module_is_deleted():
    """validation.py's eight gates re-implemented evals + Instrumentation."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("autfin.validation")


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


def test_connection_pool_commits_on_clean_exit(tmp_path, monkeypatch):
    """The pool never committed, so every pooled write was rolled back on close."""
    from autfin.memory import database
    from autfin.memory.connection_pool import get_db_pool

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "autfin.db")
    database.init_database()

    pool = get_db_pool(tmp_path / "autfin.db")
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO learning_journal (symbol, action, conviction) VALUES ('X', 'BUY', 0.5)"
        )
    with pool.connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM learning_journal").fetchone()[0]
    assert n == 1
