"""
Tests for scheduler jobs — daily morning, midday guardian, weekly review, monthly deep dive.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from src.autfin.scheduler.jobs import (
    daily_morning_routine,
    portfolio_guardian_check,
    weekly_portfolio_review,
    monthly_deep_dive,
    register_jobs,
    trigger_job,
    TIMEZONE_IST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_graph(events: list[dict[str, Any]] | None = None) -> MagicMock:
    """Return a mock graph whose astream yields the given events."""
    graph = MagicMock()
    if events is None:
        events = [{"messages": [], "errors": []}]
    graph.astream = MagicMock(return_value=async_iter(events))
    return graph


class async_iter:
    """Minimal async iterable for testing."""

    def __init__(self, items: list):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _mock_store() -> MagicMock:
    return MagicMock(name="store")


def _mock_settings() -> MagicMock:
    return MagicMock(name="settings")


# ---------------------------------------------------------------------------
# Timezone constant
# ---------------------------------------------------------------------------


class TestTimezoneConstant:
    def test_timezone_is_ist(self):
        assert TIMEZONE_IST == "Asia/Kolkata"


# ---------------------------------------------------------------------------
# daily_morning_routine
# ---------------------------------------------------------------------------


class TestDailyMorningRoutine:
    @pytest.mark.asyncio
    async def test_success_returns_completed(self):
        graph = _mock_graph([{"messages": [], "errors": []}])
        result = await daily_morning_routine(graph, _mock_store(), _mock_settings())

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("morning-")
        assert "state" in result

    @pytest.mark.asyncio
    async def test_thread_id_format(self):
        graph = _mock_graph()
        result = await daily_morning_routine(graph, _mock_store(), _mock_settings())

        tid = result["thread_id"]
        assert tid.startswith("morning-")
        # Should have date portion: morning-YYYY-MM-DD
        parts = tid.split("-")
        assert len(parts) == 4  # morning, YYYY, MM, DD
        # Check it contains a date-like string
        date_part = tid.replace("morning-", "")
        assert len(date_part) == 10  # YYYY-MM-DD
        assert date_part[4] == "-"
        assert date_part[7] == "-"

    @pytest.mark.asyncio
    async def test_initial_state_has_run_mode(self):
        graph = _mock_graph()
        await daily_morning_routine(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        assert initial_state["run_mode"] == "daily"
        assert initial_state["context"] == "morning_routine"

    @pytest.mark.asyncio
    async def test_initial_state_has_human_message(self):
        graph = _mock_graph()
        await daily_morning_routine(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        messages = initial_state["messages"]
        assert len(messages) == 1
        assert "daily morning routine" in messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_config_has_thread_id(self):
        graph = _mock_graph()
        result = await daily_morning_routine(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        config = call_args[0][1]
        assert config["configurable"]["thread_id"] == result["thread_id"]

    @pytest.mark.asyncio
    async def test_graph_astream_called_once(self):
        graph = _mock_graph()
        await daily_morning_routine(graph, _mock_store(), _mock_settings())

        graph.astream.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_events_returns_last_state(self):
        events = [
            {"messages": [], "available_cash": 1000},
            {"messages": [], "available_cash": 2000},
            {"messages": [], "available_cash": 3000},
        ]
        graph = _mock_graph(events)
        result = await daily_morning_routine(graph, _mock_store(), _mock_settings())

        assert result["state"]["available_cash"] == 3000

    @pytest.mark.asyncio
    async def test_failure_returns_failed_status(self):
        graph = MagicMock()
        graph.astream = MagicMock(side_effect=RuntimeError("graph exploded"))

        result = await daily_morning_routine(graph, _mock_store(), _mock_settings())

        assert result["status"] == "failed"
        assert "graph exploded" in result["error"]
        assert result["thread_id"].startswith("morning-")

    @pytest.mark.asyncio
    async def test_empty_events_returns_empty_state(self):
        graph = _mock_graph([])
        result = await daily_morning_routine(graph, _mock_store(), _mock_settings())

        assert result["status"] == "completed"
        assert result["state"] == {}

    @pytest.mark.asyncio
    async def test_stream_mode_values(self):
        graph = _mock_graph()
        await daily_morning_routine(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        assert call_args[1].get("stream_mode") == "values" or \
               (len(call_args[0]) > 2 and call_args[0][2] == "values")


# ---------------------------------------------------------------------------
# portfolio_guardian_check
# ---------------------------------------------------------------------------


class TestPortfolioGuardianCheck:
    @pytest.mark.asyncio
    async def test_success_returns_completed(self):
        graph = _mock_graph([{"messages": []}])
        result = await portfolio_guardian_check(graph, _mock_store(), _mock_settings())

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("guardian-")

    @pytest.mark.asyncio
    async def test_thread_id_format(self):
        graph = _mock_graph()
        result = await portfolio_guardian_check(graph, _mock_store(), _mock_settings())

        tid = result["thread_id"]
        assert tid.startswith("guardian-")
        # guardian-YYYY-MM-DD-HH
        date_part = tid.replace("guardian-", "")
        assert len(date_part) == 13  # YYYY-MM-DD-HH

    @pytest.mark.asyncio
    async def test_run_mode_is_adrenaline(self):
        graph = _mock_graph()
        await portfolio_guardian_check(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        assert initial_state["run_mode"] == "guardian"
        assert initial_state["context"] == "price_alert"

    @pytest.mark.asyncio
    async def test_initial_state_has_human_message(self):
        graph = _mock_graph()
        await portfolio_guardian_check(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        messages = initial_state["messages"]
        assert len(messages) == 1
        assert "guardian" in messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_failure_returns_failed_status(self):
        graph = MagicMock()
        graph.astream = MagicMock(side_effect=RuntimeError("guardian error"))

        result = await portfolio_guardian_check(graph, _mock_store(), _mock_settings())

        assert result["status"] == "failed"
        assert "guardian error" in result["error"]

    @pytest.mark.asyncio
    async def test_config_has_thread_id(self):
        graph = _mock_graph()
        result = await portfolio_guardian_check(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        config = call_args[0][1]
        assert config["configurable"]["thread_id"] == result["thread_id"]


# ---------------------------------------------------------------------------
# weekly_portfolio_review
# ---------------------------------------------------------------------------


class TestWeeklyPortfolioReview:
    @pytest.mark.asyncio
    async def test_success_returns_completed(self):
        graph = _mock_graph([{"messages": []}])
        result = await weekly_portfolio_review(graph, _mock_store(), _mock_settings())

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("weekly-")

    @pytest.mark.asyncio
    async def test_thread_id_format(self):
        graph = _mock_graph()
        result = await weekly_portfolio_review(graph, _mock_store(), _mock_settings())

        tid = result["thread_id"]
        assert tid.startswith("weekly-")
        date_part = tid.replace("weekly-", "")
        assert len(date_part) == 10  # YYYY-MM-DD

    @pytest.mark.asyncio
    async def test_run_mode_is_weekly_review(self):
        graph = _mock_graph()
        await weekly_portfolio_review(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        assert initial_state["run_mode"] == "weekly"
        assert initial_state["context"] == "weekly_review"

    @pytest.mark.asyncio
    async def test_initial_state_has_human_message(self):
        graph = _mock_graph()
        await weekly_portfolio_review(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        messages = initial_state["messages"]
        assert len(messages) == 1
        assert "weekly" in messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_failure_returns_failed_status(self):
        graph = MagicMock()
        graph.astream = MagicMock(side_effect=RuntimeError("weekly error"))

        result = await weekly_portfolio_review(graph, _mock_store(), _mock_settings())

        assert result["status"] == "failed"
        assert "weekly error" in result["error"]


# ---------------------------------------------------------------------------
# monthly_deep_dive
# ---------------------------------------------------------------------------


class TestMonthlyDeepDive:
    @pytest.mark.asyncio
    async def test_success_returns_completed(self):
        graph = _mock_graph([{"messages": []}])
        result = await monthly_deep_dive(graph, _mock_store(), _mock_settings())

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("monthly-")

    @pytest.mark.asyncio
    async def test_thread_id_format(self):
        graph = _mock_graph()
        result = await monthly_deep_dive(graph, _mock_store(), _mock_settings())

        tid = result["thread_id"]
        assert tid.startswith("monthly-")
        date_part = tid.replace("monthly-", "")
        assert len(date_part) == 7  # YYYY-MM

    @pytest.mark.asyncio
    async def test_run_mode_is_monthly_deep_dive(self):
        graph = _mock_graph()
        await monthly_deep_dive(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        assert initial_state["run_mode"] == "quarterly"
        assert initial_state["context"] == "quarterly_review"

    @pytest.mark.asyncio
    async def test_initial_state_has_human_message(self):
        graph = _mock_graph()
        await monthly_deep_dive(graph, _mock_store(), _mock_settings())

        call_args = graph.astream.call_args
        initial_state = call_args[0][0]
        messages = initial_state["messages"]
        assert len(messages) == 1
        assert "monthly" in messages[0].content.lower()

    @pytest.mark.asyncio
    async def test_failure_returns_failed_status(self):
        graph = MagicMock()
        graph.astream = MagicMock(side_effect=RuntimeError("monthly error"))

        result = await monthly_deep_dive(graph, _mock_store(), _mock_settings())

        assert result["status"] == "failed"
        assert "monthly error" in result["error"]


# ---------------------------------------------------------------------------
# register_jobs
# ---------------------------------------------------------------------------


class TestRegisterJobs:
    def test_registers_four_jobs(self):
        scheduler = MagicMock()
        graph = MagicMock()
        store = _mock_store()
        settings = _mock_settings()

        register_jobs(scheduler, graph, store, settings)

        assert scheduler.add_job.call_count == 4

    def test_job_ids_are_correct(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        job_ids = [call[1]["id"] for call in scheduler.add_job.call_args_list]
        assert "daily_morning" in job_ids
        assert "midday_guardian" in job_ids
        assert "weekly_review" in job_ids
        assert "monthly_deep_dive" in job_ids

    def test_job_names_are_descriptive(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        job_names = [call[1]["name"] for call in scheduler.add_job.call_args_list]
        assert "Daily Market Scan + Portfolio Guardian" in job_names
        assert "Mid-Day Portfolio Health Check" in job_names
        assert "Weekly Portfolio Review + Watchlist Refresh" in job_names
        assert "Monthly Strategy Review + Full Fundamental Refresh" in job_names

    def test_replace_existing_is_true(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        for call in scheduler.add_job.call_args_list:
            assert call[1].get("replace_existing") is True

    def test_daily_morning_uses_correct_function(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        daily_call = [
            c for c in scheduler.add_job.call_args_list
            if c[1]["id"] == "daily_morning"
        ][0]
        assert daily_call[0][0] is daily_morning_routine

    def test_midday_guardian_uses_correct_function(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        guardian_call = [
            c for c in scheduler.add_job.call_args_list
            if c[1]["id"] == "midday_guardian"
        ][0]
        assert guardian_call[0][0] is portfolio_guardian_check

    def test_weekly_review_uses_correct_function(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        weekly_call = [
            c for c in scheduler.add_job.call_args_list
            if c[1]["id"] == "weekly_review"
        ][0]
        assert weekly_call[0][0] is weekly_portfolio_review

    def test_monthly_deep_dive_uses_correct_function(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        monthly_call = [
            c for c in scheduler.add_job.call_args_list
            if c[1]["id"] == "monthly_deep_dive"
        ][0]
        assert monthly_call[0][0] is monthly_deep_dive

    def test_cron_triggers_have_ist_timezone(self):
        scheduler = MagicMock()
        register_jobs(scheduler, MagicMock(), _mock_store(), _mock_settings())

        for call in scheduler.add_job.call_args_list:
            trigger = call[0][1]  # second positional arg is the trigger
            assert trigger.timezone is not None

    def test_args_contain_graph_store_settings(self):
        scheduler = MagicMock()
        graph = MagicMock()
        store = _mock_store()
        settings = _mock_settings()

        register_jobs(scheduler, graph, store, settings)

        for call in scheduler.add_job.call_args_list:
            # args is passed as keyword arg: args=[graph, store, settings]
            job_args = call[1]["args"]
            assert job_args[0] is graph
            assert job_args[1] is store
            assert job_args[2] is settings


# ---------------------------------------------------------------------------
# trigger_job
# ---------------------------------------------------------------------------


class TestTriggerJob:
    @pytest.mark.asyncio
    async def test_trigger_daily_morning(self):
        graph = _mock_graph()
        scheduler = MagicMock()

        result = await trigger_job(
            scheduler, "daily_morning", graph, _mock_store(), _mock_settings()
        )

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("morning-")

    @pytest.mark.asyncio
    async def test_trigger_midday_guardian(self):
        graph = _mock_graph()
        scheduler = MagicMock()

        result = await trigger_job(
            scheduler, "midday_guardian", graph, _mock_store(), _mock_settings()
        )

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("guardian-")

    @pytest.mark.asyncio
    async def test_trigger_weekly_review(self):
        graph = _mock_graph()
        scheduler = MagicMock()

        result = await trigger_job(
            scheduler, "weekly_review", graph, _mock_store(), _mock_settings()
        )

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("weekly-")

    @pytest.mark.asyncio
    async def test_trigger_monthly_deep_dive(self):
        graph = _mock_graph()
        scheduler = MagicMock()

        result = await trigger_job(
            scheduler, "monthly_deep_dive", graph, _mock_store(), _mock_settings()
        )

        assert result["status"] == "completed"
        assert result["thread_id"].startswith("monthly-")

    @pytest.mark.asyncio
    async def test_trigger_unknown_job_raises(self):
        graph = _mock_graph()
        scheduler = MagicMock()

        with pytest.raises(ValueError, match="Unknown job_id"):
            await trigger_job(
                scheduler, "nonexistent_job", graph, _mock_store(), _mock_settings()
            )

    @pytest.mark.asyncio
    async def test_trigger_unknown_job_lists_available(self):
        graph = _mock_graph()
        scheduler = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            await trigger_job(
                scheduler, "bad_id", graph, _mock_store(), _mock_settings()
            )

        msg = str(exc_info.value)
        assert "daily_morning" in msg
        assert "midday_guardian" in msg
        assert "weekly_review" in msg
        assert "monthly_deep_dive" in msg


# ---------------------------------------------------------------------------
# Initial state construction (shared across all jobs)
# ---------------------------------------------------------------------------


class TestInitialStateConstruction:
    """Verify that all jobs build correct initial state dicts."""

    @pytest.mark.asyncio
    async def test_all_jobs_have_required_keys(self):
        graph = _mock_graph()
        store = _mock_store()
        settings = _mock_settings()

        job_fns = [
            daily_morning_routine,
            portfolio_guardian_check,
            weekly_portfolio_review,
            monthly_deep_dive,
        ]

        required_keys = {
            "thread_id", "run_mode", "available_cash", "holdings",
            "positions", "messages", "errors",
        }

        for fn in job_fns:
            await fn(graph, store, settings)
            call_args = graph.astream.call_args
            initial_state = call_args[0][0]
            assert required_keys.issubset(
                set(initial_state.keys())
            ), f"{fn.__name__} missing keys: {required_keys - set(initial_state.keys())}"
            # Reset mock for next call
            graph.astream.reset_mock()

    @pytest.mark.asyncio
    async def test_all_jobs_start_with_empty_holdings_and_positions(self):
        graph = _mock_graph()
        store = _mock_store()
        settings = _mock_settings()

        job_fns = [
            daily_morning_routine,
            portfolio_guardian_check,
            weekly_portfolio_review,
            monthly_deep_dive,
        ]

        for fn in job_fns:
            await fn(graph, store, settings)
            call_args = graph.astream.call_args
            initial_state = call_args[0][0]
            assert initial_state["holdings"] == []
            assert initial_state["positions"] == []
            assert initial_state["available_cash"] == 0.0
            graph.astream.reset_mock()


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_imports_from_init(self):
        from src.autfin.scheduler import (
            daily_morning_routine,
            portfolio_guardian_check,
            weekly_portfolio_review,
            monthly_deep_dive,
            register_jobs,
            trigger_job,
            TIMEZONE_IST,
        )

        assert callable(daily_morning_routine)
        assert callable(portfolio_guardian_check)
        assert callable(weekly_portfolio_review)
        assert callable(monthly_deep_dive)
        assert callable(register_jobs)
        assert callable(trigger_job)
        assert TIMEZONE_IST == "Asia/Kolkata"
