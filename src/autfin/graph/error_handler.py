"""
Error handling and recovery for AutFin graphs.

Provides:
- Error classification (ErrorType enum)
- RetryPolicy with exponential backoff
- CircuitBreaker for external service protection
- with_retry async decorator
- error_handler_node for graceful degradation
- Graph-level error boundary integration

Error tiers (from PRD §8.5):
  1. Transient (API timeout) → RetryPolicy with exponential backoff
  2. LLM-recoverable (bad parse) → Loop back with error in state
  3. User-fixable (auth expired) → interrupt() → notify user
  4. Recoverable after retries → error_handler() → fallback path
  5. Unexpected/critical → Bubble up, log, notify, graceful degradation
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar, ParamSpec

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.config.settings import AutFinSettings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorType(str, Enum):
    """Classification of errors for recovery strategy selection."""
    TRANSIENT = "transient"               # API timeout, network blip — retry
    LLM_RECOVERABLE = "llm_recoverable"   # Bad parse, wrong format — retry prompt
    USER_FIXABLE = "user_fixable"         # Auth expired, credentials invalid — HITL
    DEGRADABLE = "degradable"             # Service down — use cached/fallback
    CRITICAL = "critical"                 # Unrecoverable — bubble up, halt


# Transient error patterns (substring match against exception message/type)
_TRANSIENT_PATTERNS: list[str] = [
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporary failure",
    "rate limit",
    "429",
    "503",
    "502",
    "retry",
    "throttl",
]

_USER_FIXABLE_PATTERNS: list[str] = [
    "unauthorized",
    "401",
    "403",
    "token expired",
    "invalid token",
    "authentication",
    "permission denied",
    "access token",
]

_DEGRADABLE_PATTERNS: list[str] = [
    "not found",
    "404",
    "service unavailable",
    "down",
    "unreachable",
    "no such host",
    "name or service not known",
]


def classify_error(exc: Exception) -> ErrorType:
    """
    Classify an exception into an error tier.

    Examines both the exception type and its string representation
    to determine the best recovery strategy.
    """
    msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # Check user-fixable first (auth issues)
    for pattern in _USER_FIXABLE_PATTERNS:
        if pattern in msg or pattern in exc_type:
            return ErrorType.USER_FIXABLE

    # Check transient
    for pattern in _TRANSIENT_PATTERNS:
        if pattern in msg or pattern in exc_type:
            return ErrorType.TRANSIENT

    # Check degradable
    for pattern in _DEGRADABLE_PATTERNS:
        if pattern in msg or pattern in exc_type:
            return ErrorType.DEGRADABLE

    # LangGraph/LangChain specific
    if "outputparser" in exc_type or "outputparser" in msg or "json" in msg:
        return ErrorType.LLM_RECOVERABLE

    if "validation" in exc_type or "pydantic" in msg:
        return ErrorType.LLM_RECOVERABLE

    # Default to critical for unknown errors
    return ErrorType.CRITICAL


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """
    Retry configuration with exponential backoff.

    Usage:
        policy = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)
        result = await policy.execute(some_async_fn, arg1, arg2)
    """
    max_attempts: int = 3
    initial_interval: float = 1.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    retryable_types: list[ErrorType] = field(
        default_factory=lambda: [ErrorType.TRANSIENT]
    )

    @classmethod
    def from_settings(cls, settings: AutFinSettings | None = None) -> RetryPolicy:
        """Create a RetryPolicy from AutFinSettings."""
        if settings is None:
            settings = AutFinSettings()
        return cls(
            max_attempts=settings.retry_max_attempts,
            initial_interval=settings.retry_initial_interval,
            backoff_factor=settings.retry_backoff_factor,
        )

    def _get_interval(self, attempt: int) -> float:
        """Calculate wait interval for a given attempt (0-indexed)."""
        interval = self.initial_interval * (self.backoff_factor ** attempt)
        return min(interval, self.max_interval)

    async def execute(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """
        Execute an async function with retries.

        Retries only when the error matches a retryable ErrorType.
        Raises the last exception after all retries exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                error_type = classify_error(exc)

                if error_type not in self.retryable_types:
                    logger.warning(
                        "Non-retryable error (%s): %s", error_type.value, exc
                    )
                    raise

                if attempt < self.max_attempts - 1:
                    interval = self._get_interval(attempt)
                    logger.info(
                        "Retryable error (attempt %d/%d, %s): %s — "
                        "retrying in %.1fs",
                        attempt + 1,
                        self.max_attempts,
                        error_type.value,
                        exc,
                        interval,
                    )
                    await asyncio.sleep(interval)

        # All retries exhausted
        raise last_exc  # type: ignore[misc]


def with_retry(
    policy: RetryPolicy | None = None,
) -> Callable[
    [Callable[..., Awaitable[T]]],
    Callable[..., Awaitable[T]],
]:
    """
    Decorator that wraps an async function with retry logic.

    Usage:
        @with_retry(RetryPolicy(max_attempts=3))
        async def fetch_data(url: str) -> dict:
            ...
    """
    if policy is None:
        policy = RetryPolicy()

    def decorator(
        fn: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await policy.execute(fn, *args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Circuit breaker to protect against cascading failures.

    States:
      - CLOSED:   Normal operation, requests pass through
      - OPEN:     Too many failures, requests blocked
      - HALF_OPEN: Testing if service recovered

    Usage:
        breaker = CircuitBreaker("kite_sdk", threshold=3, reset_seconds=300)
        if breaker.allow_request():
            try:
                result = await call_service()
                breaker.record_success()
            except Exception as exc:
                breaker.record_failure()
                raise
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        threshold: int = 3,
        reset_seconds: float = 300.0,
    ):
        self.name = name
        self.threshold = threshold
        self.reset_seconds = reset_seconds
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        if self.state == self.CLOSED:
            return True

        if self.state == self.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.reset_seconds:
                self.state = self.HALF_OPEN
                logger.info(
                    "Circuit breaker '%s': OPEN → HALF_OPEN (after %.0fs)",
                    self.name,
                    elapsed,
                )
                return True
            return False

        # HALF_OPEN: allow one request to test
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == self.HALF_OPEN:
            logger.info("Circuit breaker '%s': HALF_OPEN → CLOSED", self.name)
            self.state = self.CLOSED
            self.failure_count = 0
        elif self.state == self.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            if self.state != self.OPEN:
                logger.warning(
                    "Circuit breaker '%s': %s → OPEN (failures=%d, threshold=%d)",
                    self.name,
                    self.state,
                    self.failure_count,
                    self.threshold,
                )
            self.state = self.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit breaker state for logging/storage."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "threshold": self.threshold,
            "reset_seconds": self.reset_seconds,
        }


# ---------------------------------------------------------------------------
# Global circuit breakers (per-service)
# ---------------------------------------------------------------------------

_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    threshold: int | None = None,
    reset_seconds: float | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker for a named service."""
    if name not in _circuit_breakers:
        settings = AutFinSettings()
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            threshold=threshold or settings.circuit_breaker_threshold,
            reset_seconds=reset_seconds or settings.circuit_breaker_reset_seconds,
        )
    return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing/manual recovery)."""
    for cb in _circuit_breakers.values():
        cb.reset()


# ---------------------------------------------------------------------------
# Error handler node (graceful degradation)
# ---------------------------------------------------------------------------

async def error_handler_node(
    state: AgentState,
    config: RunnableConfig | None = None,
    *,
    failed_node: str = "",
    error: Exception | None = None,
) -> dict[str, Any]:
    """
    Generic error handler for graceful degradation.

    When a node fails after retries are exhausted, this handler:
    1. Classifies the error
    2. Logs it with context
    3. Appends to state['errors']
    4. Returns a degraded state (empty/partial results) so the graph continues

    This prevents a single service failure from halting the entire pipeline.
    """
    errors: list[str] = list(state.get("errors", []) or [])
    messages: list[Any] = []

    exc = error or Exception(f"Unknown error in node '{failed_node}'")
    error_type = classify_error(exc)
    error_msg = f"[{error_type.value}] {failed_node}: {exc}"
    errors.append(error_msg)

    logger.error("Error handler caught: %s", error_msg)

    # Build degraded response based on what failed
    # Include minimal keys so downstream nodes don't crash with KeyError
    result: dict[str, Any] = {
        "errors": errors,
        "messages": messages,
        "technical_reports": state.get("technical_reports", []),
        "fundamental_reports": state.get("fundamental_reports", []),
        "sentiment_reports": state.get("sentiment_reports", []),
        "macro_signals": state.get("macro_signals"),
        "research_debates": state.get("research_debates", []),
        "risk_assessments": state.get("risk_assessments", []),
        "trade_decisions": state.get("trade_decisions", []),
    }

    # If the failed node was a research agent, provide empty results
    # so downstream nodes can continue with partial data
    research_nodes = {
        "parallel_research", "market_data", "technical",
        "fundamentals", "sentiment", "global_macro",
    }
    analysis_nodes = {"researcher", "risk_manager", "portfolio_manager"}
    execution_nodes = {"executor", "guardian_executor"}

    if failed_node in research_nodes:
        # Research failed — downstream analysis can still run with stale/cached data
        messages.append(AIMessage(
            content=(
                f"⚠ Degraded mode: {failed_node} failed ({error_type.value}). "
                f"Continuing with available data."
            )
        ))
    elif failed_node in analysis_nodes:
        # Analysis failed — skip to report
        messages.append(AIMessage(
            content=(
                f"⚠ Degraded mode: {failed_node} failed ({error_type.value}). "
                f"Skipping to daily report."
            )
        ))
    elif failed_node in execution_nodes:
        # Execution failed — log and continue, no trades placed
        messages.append(AIMessage(
            content=(
                f"⚠ Execution failed: {failed_node} ({error_type.value}). "
                f"No orders placed this run. Will retry next cycle."
            )
        ))
    else:
        messages.append(AIMessage(
            content=(
                f"⚠ Node '{failed_node}' failed ({error_type.value}): {exc}. "
                f"Continuing in degraded mode."
            )
        ))

    return result


# ---------------------------------------------------------------------------
# Safe node wrapper
# ---------------------------------------------------------------------------

def safe_node(
    node_fn: Callable[..., Awaitable[dict[str, Any]]],
    node_name: str,
    policy: RetryPolicy | None = None,
    circuit_breaker_name: str | None = None,
    degrade_on_failure: bool = True,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    """
    Wrap an async graph node with retry, circuit breaker, and error handling.

    This is the main integration point for adding error recovery to nodes.
    If the wrapped node fails after retries, the error handler provides
    a degraded state so the graph can continue.

    Usage:
        builder.add_node(
            "technical",
            safe_node(technical_node, "technical", policy=RetryPolicy.from_settings())
        )
    """
    if policy is None:
        policy = RetryPolicy.from_settings()

    breaker = None
    if circuit_breaker_name:
        breaker = get_circuit_breaker(circuit_breaker_name)

    @functools.wraps(node_fn)
    async def wrapper(
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        # Check circuit breaker
        if breaker and not breaker.allow_request():
            logger.warning(
                "Circuit breaker '%s' is OPEN — degrading node '%s'",
                circuit_breaker_name,
                node_name,
            )
            if degrade_on_failure:
                return await error_handler_node(
                    state, config,
                    failed_node=node_name,
                    error=Exception(
                        f"Circuit breaker '{circuit_breaker_name}' is OPEN"
                    ),
                )
            raise RuntimeError(
                f"Circuit breaker '{circuit_breaker_name}' is OPEN"
            )

        last_exc: Exception | None = None
        for attempt in range(policy.max_attempts):
            try:
                result = await node_fn(state, config)
                if breaker:
                    breaker.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                error_type = classify_error(exc)

                # Circuit breaker tracking
                if breaker:
                    breaker.record_failure()

                # Non-retryable → immediate degradation or re-raise
                if error_type not in policy.retryable_types:
                    logger.warning(
                        "Non-retryable error in '%s' (%s): %s",
                        node_name, error_type.value, exc,
                    )
                    break

                # Retryable but attempts remain
                if attempt < policy.max_attempts - 1:
                    interval = policy._get_interval(attempt)
                    logger.info(
                        "Retrying node '%s' (attempt %d/%d) in %.1fs: %s",
                        node_name, attempt + 1, policy.max_attempts,
                        interval, exc,
                    )
                    await asyncio.sleep(interval)

        # All retries exhausted or non-retryable error
        if degrade_on_failure:
            return await error_handler_node(
                state, config,
                failed_node=node_name,
                error=last_exc,
            )
        raise last_exc  # type: ignore[misc]

    return wrapper
