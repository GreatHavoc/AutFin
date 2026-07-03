"""
Centralized LLM client for AutFin.

Connects directly to OpenAI-compatible endpoints via the openai SDK.
Supports Modal vLLM (primary), NVIDIA NIM (fallback), and any
OpenAI-compatible provider. No litellm dependency.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level token accumulator (read by budget guard node)
_token_accumulator = {"tokens": 0, "cost": 0.0}


# ================== RATE LIMITER ==================


class _RateLimiter:
    """Smart rate limiter for NIM free tier.

    Strategy:
    - Semaphore limits concurrent requests (max 1 at a time for NIM free)
    - Adaptive delay: grows on 429, shrinks on success
    - Exponential backoff with jitter on consecutive 429s
    - Auto-recovery: gradually reduces delay when stable
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._semaphore = None  # Created per event loop
        self._last_call = 0.0
        self._min_delay = 2.0       # Start with 2s between calls (safe for NIM free)
        self._max_delay = 120.0     # Max 2 min backoff
        self._backoff_until = 0.0
        self._consecutive_429 = 0
        self._total_429 = 0
        self._total_calls = 0

    def _get_semaphore(self):
        """Get or create semaphore for current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)  # 1 concurrent request
        return self._semaphore

    async def acquire(self):
        """Wait until it's safe to make a request."""
        async with self._lock:
            now = time.monotonic()

            # If in backoff, wait for it to expire
            if now < self._backoff_until:
                wait = self._backoff_until - now + random.uniform(0.5, 2.0)
                logger.info("Rate limiter: backoff active, waiting %.0fs", wait)
                await asyncio.sleep(wait)
                now = time.monotonic()

            # Enforce minimum delay between calls
            elapsed = now - self._last_call
            if elapsed < self._min_delay:
                wait = self._min_delay - elapsed + random.uniform(0.1, 0.5)
                logger.debug("Rate limiter: enforcing %.1fs gap", wait)
                await asyncio.sleep(wait)

            self._last_call = time.monotonic()
            self._total_calls += 1

    def on_success(self):
        """Call after successful request — gradually recover."""
        self._consecutive_429 = 0
        # Shrink delay by 10% toward minimum
        if self._min_delay > 2.0:
            self._min_delay = max(2.0, self._min_delay * 0.9)

    def on_429(self, retry_after: float = 0.0):
        """Call on 429 error — increase delays aggressively."""
        self._consecutive_429 += 1
        self._total_429 += 1

        # Use Retry-After header if provided, otherwise exponential backoff
        if retry_after > 0:
            backoff = retry_after + random.uniform(1.0, 3.0)
        else:
            # Exponential: 10s, 20s, 40s, 80s, 120s max
            backoff = min(10 * (2 ** (self._consecutive_429 - 1)), self._max_delay)

        self._backoff_until = time.monotonic() + backoff

        # Also increase minimum delay between calls
        self._min_delay = min(10.0, self._min_delay + 1.0)

        logger.warning(
            "Rate limited (429 #%d, total=%d) — backing off %.0fs, delay now %.1fs",
            self._consecutive_429, self._total_429, backoff, self._min_delay,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self._total_calls,
            "total_429": self._total_429,
            "consecutive_429": self._consecutive_429,
            "min_delay": round(self._min_delay, 1),
            "backoff_active": time.monotonic() < self._backoff_until,
        }


# Global rate limiter instance
_rate_limiter = _RateLimiter()


def get_token_usage():
    """Get accumulated token usage for budget tracking."""
    return dict(_token_accumulator)


def reset_token_usage():
    """Reset token accumulator (call at start of each run)."""
    _token_accumulator["tokens"] = 0
    _token_accumulator["cost"] = 0.0


# ================== OPENAI SDK ==================

# The openai SDK connects directly to OpenAI-compatible endpoints.
# Modal vLLM, NVIDIA NIM, OpenRouter, etc. all expose OpenAI-compatible APIs.

_OPENAI_SDK_AVAILABLE = False

try:
    from openai import AsyncOpenAI as _AsyncOpenAI

    _OPENAI_SDK_AVAILABLE = True
except ImportError:
    pass


# Lazy langsmith traceable decorator for LLM I/O tracing
_traceable = None

try:
    if os.environ.get("LANGSMITH_API_KEY"):
        from langsmith import traceable as _traceable_fn

        _traceable = _traceable_fn
except Exception:
    pass


# ================== LLM CLIENT ==================


class LLMClient:
    """
    Agentic LLM client — connects directly to OpenAI-compatible endpoints.

    Uses the openai SDK to connect to any OpenAI-compatible endpoint
    (Modal vLLM, NVIDIA NIM, OpenRouter, etc.). No litellm dependency.

    The caller chooses the right model for the task. On failure, returns None —
    agents handle degradation autonomously (retry same model, switch strategy, skip).
    """

    def __init__(
        self,
        deep_model: str = "minimaxai/minimax-m3",
        fast_model: str = "meta/llama-3.1-8b-instruct",
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 180.0,
        base_url: Optional[str] = None,
        api_key: str = "dummy",
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM client.

        Args:
            deep_model: Primary model name (plain name, no provider prefix)
            fast_model: Fast model name for lightweight tasks
            temperature: Sampling temperature (0.0 = deterministic)
            max_retries: Max retries on transient failures
            timeout: Request timeout in seconds
            base_url: OpenAI-compatible endpoint URL (e.g. Modal vLLM URL)
            api_key: API key (use "dummy" for self-hosted endpoints)
            extra_body: Extra body params passed to the API (e.g. vLLM-specific)
        """
        self.deep_model = deep_model
        self.fast_model = fast_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = base_url
        self.api_key = api_key
        self.extra_body = extra_body or {}
        self._call_count = 0
        self._error_count = 0
        self._client: Optional[_AsyncOpenAI] = None

        if not _OPENAI_SDK_AVAILABLE:
            logger.warning(
                "openai SDK not installed — LLM features disabled. "
                "Install with: pip install openai"
            )

    def _get_client(self) -> _AsyncOpenAI:
        """Get or create the AsyncOpenAI client."""
        if self._client is None:
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            kwargs["timeout"] = self.timeout
            self._client = _AsyncOpenAI(**kwargs)
        return self._client

    @property
    def is_available(self) -> bool:
        """Whether the openai SDK is installed and client is usable."""
        return _OPENAI_SDK_AVAILABLE

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        state: dict | None = None,
    ) -> Optional[str]:
        """
        Generate a response from the LLM — single model, agentic design.

        No workflow fallback chain. The caller chooses the right model for the task.
        Retries transient failures with the same model only.

        Args:
            prompt: User prompt
            system_prompt: Optional system/instruction prompt
            model: Override model (default: deep_model)
            temperature: Override temperature
            max_tokens: Max response tokens
            state: Optional AgentState dict for token/cost tracking

        Returns:
            Generated text, or None on failure
        """
        if not _OPENAI_SDK_AVAILABLE:
            return None

        target_model = model or self.deep_model
        temp = temperature if temperature is not None else self.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_client()

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Wait for rate limiter before each call
                await _rate_limiter.acquire()

                t_start = time.monotonic()
                kwargs: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "timeout": self.timeout,
                }
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body

                response = await client.chat.completions.create(**kwargs)

                elapsed_ms = (time.monotonic() - t_start) * 1000
                self._call_count += 1
                _rate_limiter.on_success()

                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                self._record_llm_metrics(target_model, prompt_tokens, completion_tokens, elapsed_ms, state)

                content = response.choices[0].message.content
                if content:
                    content = content.strip()
                    # Strip Qwen thinking mode artifacts if present
                    if "<think>" in content:
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                return content if content else None
            except Exception as e:
                self._error_count += 1
                last_error = e
                error_str = str(e).lower()
                # Detect rate limiting (429) — extract Retry-After if present
                is_rate_limited = "429" in error_str or "rate" in error_str or "too many" in error_str
                if is_rate_limited:
                    # Try to extract Retry-After from error
                    retry_after = 0.0
                    if "retry-after" in error_str:
                        try:
                            retry_after = float(error_str.split("retry-after")[1].split()[0].strip('":,'))
                        except (ValueError, IndexError):
                            pass
                    _rate_limiter.on_429(retry_after)

                    # On 429, wait for backoff to fully expire before retry
                    backoff_wait = max(0, _rate_limiter._backoff_until - time.monotonic()) + random.uniform(1.0, 3.0)
                    if attempt < self.max_retries:
                        logger.info(
                            "Rate limited on attempt %d/%d — waiting %.0fs before retry",
                            attempt + 1, self.max_retries + 1, backoff_wait,
                        )
                        await asyncio.sleep(backoff_wait)
                    else:
                        logger.warning("Rate limited on final attempt %d/%d — giving up", attempt + 1, self.max_retries + 1)
                elif attempt < self.max_retries:
                    wait_time = random.uniform(0.5, 1.0)
                    logger.debug(
                        "LLM call attempt %d/%d failed on %s: %s — retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, target_model, e, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning("LLM call failed on %s after %d attempts: %s", target_model, self.max_retries + 1, e)

        return None

    async def generate_with_tools(
        self,
        system_prompt: str,
        tools: List[Dict[str, Any]],
        tool_callables: Dict[str, Callable],
        user_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: int = 10,
        max_tokens: int = 4096,
        state: dict | None = None,
    ) -> Optional[str]:
        """
        Agentic tool-calling loop — no workflow fallback.

        LLM calls tools until it produces a text response.
        Retries transient failures with the same model.

        Args:
            system_prompt: Agent's role and task description
            tools: OpenAI-format tool definitions [{"type": "function", "function": {...}}, ...]
            tool_callables: Mapping of tool name → async callable
            user_prompt: Optional initial user message
            model: Model override (default: deep_model)
            temperature: Temperature override
            max_iterations: Max tool-calling iterations
            max_tokens: Max tokens per LLM call
            state: Optional AgentState dict for token/cost tracking

        Returns:
            Final text response, or None on failure
        """
        if not _OPENAI_SDK_AVAILABLE:
            return None

        target_model = model or self.deep_model
        temp = temperature if temperature is not None else self.temperature

        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        else:
            # Some LLMs (NVIDIA NIM) require a user message before tools
            messages.append({"role": "user", "content": "Begin."})

        client = self._get_client()

        tool_iterations = 0
        for iteration in range(max_iterations):
            try:
                t_start = time.monotonic()
                kwargs: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "timeout": self.timeout,
                }
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body

                response = await client.chat.completions.create(**kwargs)

                elapsed_ms = (time.monotonic() - t_start) * 1000
                self._call_count += 1

                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                self._record_llm_metrics(target_model, prompt_tokens, completion_tokens, elapsed_ms, state)

                message = response.choices[0].message

                if message.tool_calls:
                    # Convert openai tool_calls to dict format for message history
                    tool_calls_dict = []
                    for tc in message.tool_calls:
                        tool_calls_dict.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })
                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_dict})
                    tool_iterations += 1

                    tool_results = await asyncio.gather(
                        *[_execute_tool(tc, tool_callables) for tc in message.tool_calls],
                        return_exceptions=True,
                    )

                    for tc, result in zip(message.tool_calls, tool_results):
                        if isinstance(result, Exception):
                            result_str = f"Error executing tool: {str(result)}"
                        else:
                            result_str = str(result) if result is not None else "null"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": _compress_tool_result(result_str),
                        })
                    continue

                content = message.content.strip() if message.content else None
                # Strip Qwen thinking mode artifacts if present
                if content and "<think>" in content:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content

            except Exception as e:
                self._error_count += 1
                logger.warning("LLM tool-calling error (iteration %d) on %s: %s", iteration, target_model, e)
                error_str = str(e).lower()
                is_rate_limited = "429" in error_str or "rate" in error_str or "too many" in error_str
                is_timeout = "timeout" in error_str or "timed out" in error_str
                is_connection_error = "connection" in error_str or "internal" in error_str or "500" in error_str or "502" in error_str or "503" in error_str or is_timeout
                if is_rate_limited or is_connection_error:
                    if is_rate_limited:
                        _rate_limiter.on_429()
                    backoff_wait = random.uniform(10.0, 20.0) if is_timeout else (random.uniform(5.0, 15.0) if is_connection_error else max(0, _rate_limiter._backoff_until - time.monotonic()) + random.uniform(1.0, 3.0))
                    if iteration < self.max_retries:
                        logger.warning("Rate/connection/timeout error on %s — waiting %.0fs (iteration %d)", target_model, backoff_wait, iteration)
                        await asyncio.sleep(backoff_wait)
                    else:
                        logger.warning("Rate/connection/timeout error on final iteration %d — giving up", iteration)
                if iteration < self.max_retries:
                    continue
                return None

        logger.error("Tool-calling loop exceeded max iterations (%d)", tool_iterations)
        return None

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        state: dict | None = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a JSON response from the LLM.

        Appends JSON instruction to system prompt and attempts to parse the response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Override model
            temperature: Override temperature
            state: Optional AgentState dict for token/cost tracking

        Returns:
            Parsed JSON dict, or None on failure
        """
        json_system = (system_prompt or "") + (
            "\n\nYou MUST respond with valid JSON only. "
            "No markdown, no code fences, no explanation — just the raw JSON object."
        )

        raw = await self.generate(
            prompt=prompt,
            system_prompt=json_system,
            model=model,
            temperature=temperature,
            state=state,
        )

        if not raw:
            return None

        from src.autfin.utils import extract_json
        result = extract_json(raw)
        if result:
            return result
        logger.warning("Failed to parse LLM JSON response")
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "available": _OPENAI_SDK_AVAILABLE,
            "total_calls": self._call_count,
            "total_errors": self._error_count,
            "deep_model": self.deep_model,
            "fast_model": self.fast_model,
            "base_url": self.base_url,
        }

    def _record_llm_metrics(
        self, model: str, prompt_tokens: int, completion_tokens: int, elapsed_ms: float, state: dict | None = None
    ):
        try:
            from src.autfin.observability.metrics import record_llm

            record_llm(prompt_tokens + completion_tokens)
        except Exception:
            pass
        try:
            from src.autfin.observability.tracer import get_tracer

            tracer = get_tracer()
            tid = tracer.start_trace(f"llm-{model}", {"model": model})
            tracer.log_llm_call(tid, model, prompt_tokens, completion_tokens, elapsed_ms)
            tracer.end_trace(tid)
        except Exception:
            pass
        if state is not None:
            state["total_tokens"] = state.get("total_tokens", 0) + prompt_tokens + completion_tokens
            cost_map = {"groq": 0.05, "openrouter": 0.0, "gemini": 0.0, "cerebras": 0.0, "ollama": 0.0}
            rate = cost_map.get(model.split("/")[0], 0.01)
            state["total_cost"] = state.get("total_cost", 0.0) + (prompt_tokens + completion_tokens) * rate / 1_000_000


# ================== HELPERS ==================

# Per-tool input validation schemas
TOOL_PARAM_RULES = {
    "execute_order": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
        "action": {"type": str, "allowed": ["BUY", "SELL"]},
        "quantity": {"type": int, "min": 1, "max": 100000},
        "price": {"type": float, "min": 0.01},
    },
    "log_decision": {
        "decision_type": {"type": str, "max_len": 50},
        "details": {"type": dict},
    },
    "research_stock": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
        "depth": {"type": str, "allowed": ["quick", "standard", "deep"]},
    },
    "research_batch": {
        "symbols": {"type": list},
        "depth": {"type": str, "allowed": ["quick", "standard", "deep"]},
    },
    "screen_candidates": {
        "universe": {"type": str},
        "max_candidates": {"type": int, "min": 1, "max": 100},
    },
    "validate_thesis": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
        "holding_data": {"type": dict},
    },
    "run_risk_checks": {
        "trade": {"type": dict},
    },
    "allocate_capital": {
        "approved_trades": {"type": list},
        "available_cash": {"type": float, "min": 0},
    },
    "build_stock_pitch": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
        "research_data": {"type": dict},
    },
    "retrieve_learnings": {
        "query": {"type": str},
        "filters": {"type": dict},
    },
    "record_learning": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
        "learning_type": {"type": str},
        "content": {"type": str},
        "confidence": {"type": float, "min": 0.0, "max": 1.0},
    },
    "calibrate_conviction": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
    },
    "detect_patterns": {
        "min_samples": {"type": int, "min": 1},
    },
    "request_human_approval": {
        "question": {"type": str},
        "trade_details": {"type": dict},
    },
    "check_guardian": {},
    "analyze_macro_context": {},
    "get_market_status": {},
    "get_market_quotes": {
        "symbols": {"type": list},
    },
    "get_portfolio_state": {},
    "validate_reasonability": {
        "details": {"type": dict},
    },
    "finalize_session": {},
    "get_macro_outlook": {},
    "get_sector_exposure": {},
}


def _validate_tool_args(func_name: str, args: dict) -> tuple[dict, str | None]:
    """Validate and sanitize tool arguments from LLM. Returns (cleaned_args, error_msg)."""
    rules = TOOL_PARAM_RULES.get(func_name)
    if rules is None:
        return args, None

    cleaned = {}
    for key, val in args.items():
        rule = rules.get(key)
        if rule is None:
            cleaned[key] = val
            continue

        # Type coercion
        expected = rule.get("type")
        if expected is not None:
            try:
                val = expected(val)
            except (ValueError, TypeError):
                return cleaned, f"Invalid type for {key}: expected {expected.__name__}"

        # String constraints
        if isinstance(val, str):
            if rule.get("upper"):
                val = val.upper().strip()
            max_len = rule.get("max_len")
            if max_len and len(val) > max_len:
                val = val[:max_len]

        # Numeric constraints
        if isinstance(val, (int, float)):
            min_val = rule.get("min")
            max_val = rule.get("max")
            if min_val is not None and val < min_val:
                val = min_val
            if max_val is not None and val > max_val:
                val = max_val

        # Allowed values
        allowed = rule.get("allowed")
        if allowed and val not in allowed:
            return cleaned, f"Invalid value for {key}: {val}. Must be one of {allowed}"

        cleaned[key] = val

    return cleaned, None


def _compress_tool_result(result_str: str, max_chars: int = 8000) -> str:
    """Intelligently compress tool result to fit within max_chars.

    Strategy:
    1. If already fits, return as-is
    2. Try to find natural break point (last closing brace) in last 30%
    3. Fall back to head + tail + truncation marker
    """
    if len(result_str) <= max_chars:
        return result_str

    # Try to find last complete JSON object or section in the truncated slice
    truncated = result_str[:max_chars]
    last_brace = truncated.rfind("}")
    if last_brace > max_chars * 0.7:
        # Found a closing brace in the natural break zone — preserve it
        return truncated[:last_brace + 1]

    # Head + tail compression: take first 60% + last 25% + truncation marker
    head_size = int(max_chars * 0.6)
    tail_size = int(max_chars * 0.25)
    head = result_str[:head_size]
    tail = result_str[-tail_size:]
    truncated_chars = len(result_str) - head_size - tail_size
    return f"{head}\n\n... [truncated {truncated_chars} chars] ...\n\n{tail}"


async def _execute_tool(
    tool_call: Any,
    tool_callables: Dict[str, Callable],
) -> Any:
    """Execute a single tool call from the LLM's response with input validation."""
    func_name = tool_call.function.name

    try:
        raw_args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        return {"status": "error", "message": "Invalid JSON in tool arguments"}

    # Validate and sanitize arguments
    args, error = _validate_tool_args(func_name, raw_args)
    if error:
        return {"status": "error", "message": error}

    fn = tool_callables.get(func_name)
    if fn is None:
        return {"status": "error", "message": f"Unknown tool: {func_name}"}

    result = fn(**args)
    if asyncio.iscoroutine(result):
        result = await result
    return result


# ================== SINGLETON ==================

_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Get or create the global LLM client instance.

    Reads model configuration from AutFinSettings.
    Connects directly to OpenAI-compatible endpoints (Modal vLLM, NVIDIA NIM, etc.).

    Returns:
        LLMClient instance
    """
    global _default_client

    if _default_client is None:
        try:
            from src.autfin.config.settings import AutFinSettings

            settings = AutFinSettings()

            # If Modal vLLM is configured, use it as primary
            if settings.modal_vllm_url:
                _default_client = LLMClient(
                    deep_model=settings.modal_vllm_model,
                    fast_model=settings.fast_model,
                    temperature=settings.model_temperature,
                    base_url=settings.modal_vllm_url,
                    api_key="dummy",  # Self-hosted vLLM doesn't need a real key
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                logger.info("LLM client configured: Modal vLLM (%s) at %s", settings.modal_vllm_model, settings.modal_vllm_url)
            else:
                # Use NVIDIA NIM with MiniMax M3 (OpenAI-compatible endpoint)
                nvidia_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
                if nvidia_key:
                    _default_client = LLMClient(
                        deep_model="minimaxai/minimax-m3",
                        fast_model="meta/llama-3.1-8b-instruct",
                        temperature=settings.model_temperature,
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key=nvidia_key,
                    )
                    logger.info("LLM client configured: NVIDIA NIM (minimaxai/minimax-m3)")
                else:
                    # No provider configured — disabled
                    _default_client = LLMClient()
                    logger.warning("No LLM provider configured. Set MODAL_VLLM_URL or NVIDIA_NIM_API_KEY in .env")
        except Exception as e:
            logger.debug("Settings load failed (%s), using default LLM config", e)
            _default_client = LLMClient()

    return _default_client


def reset_llm_client() -> None:
    """Reset the global LLM client (useful for testing)."""
    global _default_client
    _default_client = None
