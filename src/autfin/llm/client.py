"""
Centralized LLM client for AutFin.

Uses litellm for all provider routing. The caller chooses the right model for the task.
Returns None on failure — agents handle degradation autonomously.
"""

import asyncio
import json
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level token accumulator (read by budget guard node)
_token_accumulator = {"tokens": 0, "cost": 0.0}

# Module-level rate limiter — prevents 429s across all LLM calls
_rate_limiter = {
    "last_call_time": 0.0,
    "min_interval": 1.0,        # Min seconds between calls (1 RPM = 1 req/sec)
    "backoff_until": 0.0,       # Unix timestamp when backoff expires
    "consecutive_429": 0,       # Count of consecutive 429 errors
}


async def _wait_for_rate_limit():
    """Enforce rate limiting before each LLM call."""
    now = time.monotonic()

    # If in backoff period, wait until it expires
    backoff_remaining = _rate_limiter["backoff_until"] - now
    if backoff_remaining > 0:
        wait = backoff_remaining + random.uniform(0.1, 0.5)  # Add jitter
        logger.debug("Rate limiter: waiting %.1fs for backoff to expire", wait)
        await asyncio.sleep(wait)

    # Enforce minimum interval between calls
    elapsed = now - _rate_limiter["last_call_time"]
    if elapsed < _rate_limiter["min_interval"]:
        wait = _rate_limiter["min_interval"] - elapsed + random.uniform(0.05, 0.2)
        logger.debug("Rate limiter: enforcing %.2fs gap between calls", wait)
        await asyncio.sleep(wait)

    _rate_limiter["last_call_time"] = time.monotonic()


def _on_rate_limit_error():
    """Handle a 429 error — increase backoff exponentially."""
    _rate_limiter["consecutive_429"] += 1
    # Exponential backoff: 5s, 10s, 20s, 40s, 60s max
    backoff = min(5 * (2 ** (_rate_limiter["consecutive_429"] - 1)), 60.0)
    _rate_limiter["backoff_until"] = time.monotonic() + backoff
    _rate_limiter["min_interval"] = min(2.0, _rate_limiter["min_interval"] + 0.2)
    logger.warning(
        "Rate limited (429 #%d) — backing off for %.0fs, min_interval=%.1fs",
        _rate_limiter["consecutive_429"], backoff, _rate_limiter["min_interval"],
    )


def _on_success():
    """Reset rate limiter state on successful call."""
    _rate_limiter["consecutive_429"] = 0
    # Gradually reduce interval back toward 1.0s
    if _rate_limiter["min_interval"] > 1.0:
        _rate_limiter["min_interval"] = max(1.0, _rate_limiter["min_interval"] - 0.1)


def get_token_usage():
    """Get accumulated token usage for budget tracking."""
    return dict(_token_accumulator)


def reset_token_usage():
    """Reset token accumulator (call at start of each run)."""
    _token_accumulator["tokens"] = 0
    _token_accumulator["cost"] = 0.0


# Lazy import flags
_LITELLM_AVAILABLE = False
_litellm = None

try:
    import litellm as _litellm_mod

    _litellm = _litellm_mod
    _LITELLM_AVAILABLE = True
    # Suppress litellm's own verbose logging
    _litellm_mod.suppress_debug_info = True
    import os

    os.environ.setdefault("LITELLM_LOG", "ERROR")
except ImportError:
    pass


class LLMClient:
    """
    Agentic LLM client — no workflow fallback chains.

    Uses litellm to route requests across providers (OpenRouter, Groq, Gemini, Cerebras, Ollama).
    The caller chooses the right model for the task. On failure, returns None —
    agents handle degradation autonomously (retry same model, switch strategy, skip).
    """

    def __init__(
        self,
        deep_model: str = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        fast_model: str = "nvidia_nim/meta/llama-3.1-8b-instruct",
        temperature: float = 0.0,
        max_retries: int = 1,
        timeout: float = 30.0,
    ):
        """
        Initialize LLM client. v3.0: litellm handles all provider routing.

        Args:
            deep_model: Primary model (litellm format: provider/model-name)
            fast_model: Fast model for lightweight tasks
            temperature: Sampling temperature (0.0 = deterministic)
            max_retries: Max retries on transient failures (1 = fail fast)
            timeout: Request timeout in seconds
        """
        self.deep_model = deep_model
        self.fast_model = fast_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self._available = _LITELLM_AVAILABLE
        self._call_count = 0
        self._error_count = 0

        if not self._available:
            logger.warning(
                "litellm not installed — LLM features disabled. "
                "Install with: pip install litellm"
            )

    @property
    def is_available(self) -> bool:
        """Whether litellm is installed and usable."""
        return self._available

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
        if not self._available:
            return None

        target_model = model or self.deep_model
        temp = temperature if temperature is not None else self.temperature

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Wait for rate limiter before each call
                await _wait_for_rate_limit()

                t_start = time.monotonic()
                response = await _litellm.acompletion(
                    model=target_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
                elapsed_ms = (time.monotonic() - t_start) * 1000
                self._call_count += 1
                _on_success()

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                self._record_llm_metrics(target_model, prompt_tokens, completion_tokens, elapsed_ms, state)

                content = response.choices[0].message.content
                return content.strip() if content else None
            except Exception as e:
                self._error_count += 1
                last_error = e
                error_str = str(e).lower()
                # Detect rate limiting (429) — exponential backoff
                is_rate_limited = "429" in error_str or "rate" in error_str or "too many" in error_str
                if is_rate_limited:
                    _on_rate_limit_error()
                if attempt < self.max_retries:
                    wait_time = random.uniform(3.0, 6.0) if is_rate_limited else 0.5
                    logger.debug(
                        "LLM call attempt %d/%d failed on %s%s: %s — retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, target_model,
                        " (RATE LIMITED)" if is_rate_limited else "", e, wait_time,
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
        if not self._available:
            return None

        target_model = model or self.deep_model
        temp = temperature if temperature is not None else self.temperature

        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        else:
            # Some LLMs (NVIDIA NIM) require a user message before tools
            messages.append({"role": "user", "content": "Begin."})

        for iteration in range(max_iterations):
            try:
                t_start = time.monotonic()
                response = await _litellm.acompletion(
                    model=target_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temp,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                )
                elapsed_ms = (time.monotonic() - t_start) * 1000
                self._call_count += 1

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                self._record_llm_metrics(target_model, prompt_tokens, completion_tokens, elapsed_ms, state)

                message = response.choices[0].message

                if message.tool_calls:
                    messages.append(message)

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

                return message.content.strip() if message.content else None

            except Exception as e:
                self._error_count += 1
                logger.warning("LLM tool-calling error (iteration %d) on %s: %s", iteration, target_model, e)
                error_str = str(e).lower()
                is_rate_limited = "429" in error_str or "rate" in error_str or "too many" in error_str
                if is_rate_limited:
                    logger.warning("Rate limited on %s — will retry", target_model)
                if iteration < self.max_retries:
                    continue
                return None

        logger.error("Tool-calling loop exceeded max iterations (%d)", max_iterations)
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

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()

        try:
            import json

            return json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "available": self._available,
            "total_calls": self._call_count,
            "total_errors": self._error_count,
            "deep_model": self.deep_model,
            "fast_model": self.fast_model,
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
        "symbol": {"type": str, "max_len": 20, "upper": True},
        "action": {"type": str, "allowed": ["BUY", "SELL", "HOLD"]},
        "conviction": {"type": float, "min": 0.0, "max": 1.0},
    },
    "research_stock": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
    },
    "screen_candidates": {},
    "validate_thesis": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
    },
    "run_risk_checks": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
    },
    "allocate_capital": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
    },
    "build_stock_pitch": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
    },
    "retrieve_learnings": {},
    "record_learning": {},
    "calibrate_conviction": {},
    "detect_patterns": {},
    "request_human_approval": {},
    "check_guardian": {},
    "analyze_macro_context": {},
    "get_market_status": {},
    "get_portfolio_state": {},
    "get_market_quotes": {
        "symbol": {"type": str, "max_len": 20, "upper": True},
    },
    "finalize_session": {},
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

    Returns:
        LLMClient instance
    """
    global _default_client

    if _default_client is None:
        try:
            from src.autfin.config.settings import AutFinSettings

            settings = AutFinSettings()
            _default_client = LLMClient(
                deep_model=settings.deep_model,
                fast_model=settings.fast_model,
                temperature=settings.model_temperature,
            )
        except Exception as e:
            logger.debug("Settings load failed (%s), using default LLM config", e)
            _default_client = LLMClient()

    return _default_client


def reset_llm_client() -> None:
    """Reset the global LLM client (useful for testing)."""
    global _default_client
    _default_client = None
