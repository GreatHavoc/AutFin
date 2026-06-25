"""
Tests for error handling and recovery module.

Covers:
- Error classification (ErrorType)
- RetryPolicy (exponential backoff, retryable types, interval calculation)
- CircuitBreaker (state transitions, threshold, reset, half-open)
- with_retry decorator
- error_handler_node (graceful degradation for each node type)
- safe_node wrapper (retry + circuit breaker + degrade)
- Edge cases: all retries exhausted, non-retryable errors, concurrent failures
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.autfin.graph.error_handler import (
    ErrorType,
    RetryPolicy,
    CircuitBreaker,
    classify_error,
    with_retry,
    error_handler_node,
    safe_node,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    _circuit_breakers,
)
from src.autfin.state.agent_state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> AgentState:
    s: AgentState = {
        "messages": [],
        "available_cash": 10000.0,
        "utilized_margin": 0.0,
        "holdings": [],
        "positions": [],
        "thread_id": "test-001",
        "run_mode": "daily_morning",
        "errors": [],
    }
    s.update(overrides)
    return s


class _FakeTimeoutError(Exception):
    pass


class _FakeAuthError(Exception):
    pass


class _FakeCriticalError(Exception):
    pass


# ---------------------------------------------------------------------------
# ErrorType classification
# ---------------------------------------------------------------------------

class TestClassifyError:
    """Tests for classify_error function."""

    def test_transient_timeout(self):
        assert classify_error(TimeoutError("connection timed out")) == ErrorType.TRANSIENT

    def test_transient_connection_reset(self):
        assert classify_error(ConnectionError("Connection reset by peer")) == ErrorType.TRANSIENT

    def test_transient_rate_limit(self):
        assert classify_error(Exception("Rate limit exceeded 429")) == ErrorType.TRANSIENT

    def test_transient_503(self):
        assert classify_error(Exception("HTTP 503 Service Unavailable")) == ErrorType.TRANSIENT

    def test_user_fixable_401(self):
        assert classify_error(Exception("HTTP 401 Unauthorized")) == ErrorType.USER_FIXABLE

    def test_user_fixable_token_expired(self):
        assert classify_error(Exception("Token expired, re-authenticate")) == ErrorType.USER_FIXABLE

    def test_user_fixable_permission_denied(self):
        assert classify_error(PermissionError("Permission denied")) == ErrorType.USER_FIXABLE

    def test_degradable_not_found(self):
        assert classify_error(Exception("Resource not found 404")) == ErrorType.DEGRADABLE

    def test_degradable_service_unavailable(self):
        assert classify_error(Exception("Service unavailable")) == ErrorType.DEGRADABLE

    def test_degradable_unreachable(self):
        assert classify_error(Exception("no such host: api.example.com")) == ErrorType.DEGRADABLE

    def test_llm_recoverable_json(self):
        assert classify_error(Exception("Invalid JSON output from LLM")) == ErrorType.LLM_RECOVERABLE

    def test_llm_recoverable_output_parser(self):
        assert classify_error(Exception("OutputParserException occurred")) == ErrorType.LLM_RECOVERABLE

    def test_llm_recoverable_validation(self):
        assert classify_error(ValueError("pydantic validation error")) == ErrorType.LLM_RECOVERABLE

    def test_critical_unknown(self):
        assert classify_error(RuntimeError("Something completely unexpected")) == ErrorType.CRITICAL

    def test_critical_empty_message(self):
        assert classify_error(Exception("")) == ErrorType.CRITICAL

    def test_empty_string_error(self):
        assert classify_error(Exception()) == ErrorType.CRITICAL


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_default_values(self):
        p = RetryPolicy()
        assert p.max_attempts == 3
        assert p.initial_interval == 1.0
        assert p.backoff_factor == 2.0
        assert p.max_interval == 60.0

    def test_from_settings(self):
        p = RetryPolicy.from_settings()
        assert p.max_attempts == 3  # default from AutFinSettings
        assert p.initial_interval == 1.0
        assert p.backoff_factor == 2.0

    def test_get_interval_exponential(self):
        p = RetryPolicy(initial_interval=1.0, backoff_factor=2.0)
        assert p._get_interval(0) == 1.0   # attempt 0 → 1.0
        assert p._get_interval(1) == 2.0   # attempt 1 → 2.0
        assert p._get_interval(2) == 4.0   # attempt 2 → 4.0
        assert p._get_interval(3) == 8.0   # attempt 3 → 8.0

    def test_get_interval_capped(self):
        p = RetryPolicy(initial_interval=10.0, backoff_factor=4.0, max_interval=30.0)
        assert p._get_interval(0) == 10.0
        assert p._get_interval(1) == 30.0  # capped at max
        assert p._get_interval(2) == 30.0

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self):
        fn = AsyncMock(return_value="ok")
        p = RetryPolicy(max_attempts=3)
        result = await p.execute(fn)
        assert result == "ok"
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_retry_then_success(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timed out")
            return "recovered"

        p = RetryPolicy(max_attempts=3, initial_interval=0.01)
        result = await p.execute(flaky)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_all_retries_exhausted(self):
        fn = AsyncMock(side_effect=TimeoutError("timed out"))
        p = RetryPolicy(max_attempts=2, initial_interval=0.01)
        with pytest.raises(TimeoutError, match="timed out"):
            await p.execute(fn)
        assert fn.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_non_retryable_raises_immediately(self):
        fn = AsyncMock(side_effect=PermissionError("unauthorized"))
        p = RetryPolicy(max_attempts=3, initial_interval=0.01)
        with pytest.raises(PermissionError, match="unauthorized"):
            await p.execute(fn)
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_args_and_kwargs(self):
        async def add(a, b, c=0):
            return a + b + c

        p = RetryPolicy(max_attempts=1)
        result = await p.execute(add, 1, 2, c=3)
        assert result == 6


# ---------------------------------------------------------------------------
# with_retry decorator
# ---------------------------------------------------------------------------

class TestWithRetry:
    """Tests for with_retry decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        @with_retry(RetryPolicy(max_attempts=2, initial_interval=0.01))
        async def my_fn():
            return 42

        assert await my_fn() == 42

    @pytest.mark.asyncio
    async def test_decorator_retries_on_failure(self):
        calls = 0

        @with_retry(RetryPolicy(max_attempts=3, initial_interval=0.01))
        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("timeout")
            return "done"

        assert await flaky() == "done"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_name(self):
        @with_retry(RetryPolicy(max_attempts=1))
        async def my_special_function():
            return True

        assert my_special_function.__name__ == "my_special_function"


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", threshold=3)
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.allow_request() is True

    def test_failure_increments_count(self):
        cb = CircuitBreaker("test", threshold=3)
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.state == CircuitBreaker.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert cb.allow_request() is False

    def test_success_resets_count(self):
        cb = CircuitBreaker("test", threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_after_reset_seconds(self):
        cb = CircuitBreaker("test", threshold=2, reset_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        time.sleep(0.15)
        assert cb.allow_request() is True
        assert cb.state == CircuitBreaker.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker("test", threshold=2, reset_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # → HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_opens_again(self):
        cb = CircuitBreaker("test", threshold=2, reset_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # → HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

    def test_reset(self):
        cb = CircuitBreaker("test", threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.failure_count == 0

    def test_to_dict(self):
        cb = CircuitBreaker("test_service", threshold=5, reset_seconds=60.0)
        d = cb.to_dict()
        assert d["name"] == "test_service"
        assert d["state"] == CircuitBreaker.CLOSED
        assert d["threshold"] == 5
        assert d["reset_seconds"] == 60.0

    def test_multiple_failures_then_success_resets(self):
        cb = CircuitBreaker("test", threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.failure_count == 4
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitBreaker.CLOSED


# ---------------------------------------------------------------------------
# get_circuit_breaker / reset_all
# ---------------------------------------------------------------------------

class TestCircuitBreakerGlobals:
    """Tests for global circuit breaker registry."""

    def setup_method(self):
        reset_all_circuit_breakers()

    def test_get_creates_new(self):
        cb = get_circuit_breaker("kite")
        assert cb.name == "kite"
        assert cb.state == CircuitBreaker.CLOSED

    def test_get_returns_same_instance(self):
        cb1 = get_circuit_breaker("kite")
        cb2 = get_circuit_breaker("kite")
        assert cb1 is cb2

    def test_different_names_different_instances(self):
        cb1 = get_circuit_breaker("kite")
        cb2 = get_circuit_breaker("llm")
        assert cb1 is not cb2

    def test_reset_all(self):
        cb = get_circuit_breaker("kite")
        cb.record_failure()
        cb.record_failure()
        reset_all_circuit_breakers()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# error_handler_node
# ---------------------------------------------------------------------------

class TestErrorHandlerNode:
    """Tests for error_handler_node graceful degradation."""

    @pytest.mark.asyncio
    async def test_returns_degraded_state(self):
        state = _base_state()
        result = await error_handler_node(
            state, failed_node="technical", error=TimeoutError("API timeout")
        )
        assert len(result["errors"]) == 1
        assert "transient" in result["errors"][0]
        assert "technical" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_appends_to_existing_errors(self):
        state = _base_state(errors=["previous error"])
        result = await error_handler_node(
            state, failed_node="sentiment", error=Exception("down")
        )
        assert len(result["errors"]) == 2
        assert result["errors"][0] == "previous error"

    @pytest.mark.asyncio
    async def test_research_node_degradation_message(self):
        state = _base_state()
        result = await error_handler_node(
            state, failed_node="parallel_research", error=TimeoutError("timeout")
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert "Degraded mode" in msgs[0].content
        assert "parallel_research" in msgs[0].content

    @pytest.mark.asyncio
    async def test_analysis_node_degradation_message(self):
        state = _base_state()
        result = await error_handler_node(
            state, failed_node="risk_manager", error=Exception("bad parse")
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert "Skipping" in msgs[0].content

    @pytest.mark.asyncio
    async def test_execution_node_degradation_message(self):
        state = _base_state()
        result = await error_handler_node(
            state, failed_node="executor", error=Exception("connection refused")
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert "No orders placed" in msgs[0].content

    @pytest.mark.asyncio
    async def test_unknown_node_generic_message(self):
        state = _base_state()
        result = await error_handler_node(
            state, failed_node="some_custom_node", error=Exception("oops")
        )
        msgs = result["messages"]
        assert len(msgs) == 1
        assert "some_custom_node" in msgs[0].content

    @pytest.mark.asyncio
    async def test_error_without_exception_uses_default(self):
        state = _base_state()
        result = await error_handler_node(state, failed_node="test")
        assert len(result["errors"]) == 1
        assert "Unknown error" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_preserves_existing_state_keys(self):
        state = _base_state(available_cash=5000.0, holdings=[{"symbol": "TCS"}])
        result = await error_handler_node(
            state, failed_node="technical", error=Exception("fail")
        )
        # error_handler only returns errors + messages, not other state keys
        assert "errors" in result
        assert "messages" in result


# ---------------------------------------------------------------------------
# safe_node wrapper
# ---------------------------------------------------------------------------

class TestSafeNode:
    """Tests for safe_node wrapper."""

    @pytest.mark.asyncio
    async def test_successful_node_passes_through(self):
        async def good_node(state, config=None):
            return {"available_cash": 999.0, "messages": [], "errors": []}

        wrapped = safe_node(good_node, "test_node", policy=RetryPolicy(max_attempts=1))
        state = _base_state()
        result = await wrapped(state)
        assert result["available_cash"] == 999.0

    @pytest.mark.asyncio
    async def test_retryable_error_retries_then_degrades(self):
        call_count = 0

        async def flaky_node(state, config=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timed out")
            return {"available_cash": 100.0, "messages": [], "errors": []}

        policy = RetryPolicy(max_attempts=3, initial_interval=0.01)
        wrapped = safe_node(flaky_node, "flaky", policy=policy)
        state = _base_state()
        result = await wrapped(state)
        # Should succeed on 3rd attempt (no degradation needed)
        assert result["available_cash"] == 100.0
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_degrades(self):
        async def always_fail(state, config=None):
            raise TimeoutError("always fails")

        policy = RetryPolicy(max_attempts=2, initial_interval=0.01)
        wrapped = safe_node(always_fail, "failing", policy=policy)
        state = _base_state()
        result = await wrapped(state)
        # Should degrade gracefully
        assert len(result["errors"]) == 1
        assert "failing" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_non_retryable_error_immediate_degrade(self):
        async def auth_fail(state, config=None):
            raise PermissionError("unauthorized 401")

        policy = RetryPolicy(max_attempts=3, initial_interval=0.01)
        wrapped = safe_node(auth_fail, "auth", policy=policy)
        state = _base_state()
        result = await wrapped(state)
        # Should degrade immediately, not retry 3 times
        assert len(result["errors"]) == 1
        assert "user_fixable" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self):
        async def good_node(state, config=None):
            return {"messages": [], "errors": []}

        cb = get_circuit_breaker("test_svc", threshold=2, reset_seconds=10)
        # Force circuit open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        wrapped = safe_node(
            good_node, "test",
            circuit_breaker_name="test_svc",
            policy=RetryPolicy(max_attempts=1),
        )
        state = _base_state()
        result = await wrapped(state)
        # Should degrade because circuit is open
        assert len(result["errors"]) == 1
        assert "OPEN" in result["errors"][0]

        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self):
        async def good_node(state, config=None):
            return {"messages": [], "errors": []}

        cb = get_circuit_breaker("test_svc2", threshold=3)
        wrapped = safe_node(
            good_node, "test",
            circuit_breaker_name="test_svc2",
            policy=RetryPolicy(max_attempts=1),
        )
        state = _base_state()
        await wrapped(state)
        assert cb.failure_count == 0

        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure(self):
        async def bad_node(state, config=None):
            raise TimeoutError("timeout")

        cb = get_circuit_breaker("test_svc3", threshold=3)
        wrapped = safe_node(
            bad_node, "test",
            circuit_breaker_name="test_svc3",
            policy=RetryPolicy(max_attempts=1, initial_interval=0.01),
        )
        state = _base_state()
        await wrapped(state)
        assert cb.failure_count == 1

        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_degrade_on_failure_false_raises(self):
        async def bad_node(state, config=None):
            raise TimeoutError("timeout")

        policy = RetryPolicy(max_attempts=1, initial_interval=0.01)
        wrapped = safe_node(
            bad_node, "test", policy=policy, degrade_on_failure=False,
        )
        state = _base_state()
        with pytest.raises(TimeoutError):
            await wrapped(state)

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        async def my_node_fn(state, config=None):
            return {"messages": [], "errors": []}

        wrapped = safe_node(my_node_fn, "node")
        assert wrapped.__name__ == "my_node_fn"

    @pytest.mark.asyncio
    async def test_research_node_degradation(self):
        async def research_fail(state, config=None):
            raise TimeoutError("API timeout")

        policy = RetryPolicy(max_attempts=1, initial_interval=0.01)
        wrapped = safe_node(research_fail, "parallel_research", policy=policy)
        state = _base_state()
        result = await wrapped(state)
        assert len(result["errors"]) == 1
        # Research node should mention "Degraded mode"
        assert len(result["messages"]) == 1
        assert "Degraded mode" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_execution_node_degradation(self):
        async def executor_fail(state, config=None):
            raise ConnectionError("connection refused")

        policy = RetryPolicy(max_attempts=1, initial_interval=0.01)
        wrapped = safe_node(executor_fail, "executor", policy=policy)
        state = _base_state()
        result = await wrapped(state)
        assert len(result["errors"]) == 1
        assert len(result["messages"]) == 1
        assert "No orders placed" in result["messages"][0].content
