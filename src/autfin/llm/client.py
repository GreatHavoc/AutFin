"""
Centralized LLM client for AutFin.

Pydantic AI LLM access via langchain-openai ``ChatOpenAI``. Connects to
any OpenAI-compatible endpoint (Modal vLLM, NVIDIA NIM, OpenRouter, etc.) and
drives tool-calling loops through Pydantic AI agents â€” no
hand-rolled ReAct loop and no raw ``openai`` SDK. No litellm dependency.
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
        """Call after successful request â€” gradually recover."""
        self._consecutive_429 = 0
        # Shrink delay by 10% toward minimum
        if self._min_delay > 2.0:
            self._min_delay = max(2.0, self._min_delay * 0.9)

    def on_429(self, retry_after: float = 0.0):
        """Call on 429 error â€” increase delays aggressively."""
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
            "Rate limited (429 #%d, total=%d) â€” backing off %.0fs, delay now %.1fs",
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


# ================== LANGCHAIN OPENAI ==================

# ChatOpenAI connects to any OpenAI-compatible endpoint via base_url.
# Modal vLLM, NVIDIA NIM, OpenRouter, etc. all expose OpenAI-compatible APIs.

_LANGCHAIN_OPENAI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI

    _LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.tools import StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    _LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    _LANGCHAIN_CORE_AVAILABLE = False


# LangSmith tracing for automatic LLM call tracing
_traceable = None

try:
    from langsmith import traceable as _traceable_fn

    _traceable = _traceable_fn
except ImportError:
    pass


# ================== LLM CLIENT ==================


class LLMClient:
    """
    Agentic LLM client — Pydantic AI-native, built on langchain-openai ChatOpenAI.

    Connects to any OpenAI-compatible endpoint (Modal vLLM, NVIDIA NIM,
    OpenRouter, etc.) via ``ChatOpenAI(base_url=..., api_key=...)``. Tool-calling
    loops are driven by Pydantic AI agents — no hand-rolled ReAct
    loop, no raw ``openai`` SDK, no ``ModelMetaclass`` serialization hazard.

    The caller chooses the right model for the task. On failure, returns None —
    agents handle degradation autonomously (retry same model, switch strategy, skip).
    """

    def __init__(
        self,
        deep_model: str = "thinkingmachines/inkling",
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
        self._model: Optional[Any] = None

        if not _LANGCHAIN_OPENAI_AVAILABLE:
            logger.warning(
                "langchain-openai not installed — LLM features disabled. "
                "Install with: pip install langchain-openai"
            )

    @property
    def is_available(self) -> bool:
        """Whether langchain-openai is installed and a backend is usable."""
        return _LANGCHAIN_OPENAI_AVAILABLE

    def _build_model(self, model: Optional[str] = None, temperature: Optional[float] = None) -> Any:
        """Build (and cache) a ``ChatOpenAI`` bound to the configured endpoint."""
        if self._model is None:
            kwargs: Dict[str, Any] = {
                "model": model or self.deep_model,
                "temperature": temperature if temperature is not None else self.temperature,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "api_key": self.api_key,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.extra_body:
                # vLLM / provider-specific knobs (e.g. enable_thinking)
                kwargs["model_kwargs"] = self.extra_body
            self._model = _ChatOpenAI(**kwargs)
        return self._model

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
        Generate a response from the LLM via ChatOpenAI.ainvoke.

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
        if not _LANGCHAIN_OPENAI_AVAILABLE:
            return None

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        client = self._build_model(model, temperature)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                await _rate_limiter.acquire()
                t_start = time.monotonic()
                response = await client.ainvoke(messages)
                elapsed_ms = (time.monotonic() - t_start) * 1000
                self._call_count += 1
                _rate_limiter.on_success()

                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0

                updates = self._record_llm_metrics(model or self.deep_model, prompt_tokens, completion_tokens, elapsed_ms, state)
                if updates and state is not None:
                    state.update(updates)

                if isinstance(response.content, str):
                    content = response.content
                else:
                    # Non-string content (None, list of content blocks) → no text
                    content = ""
                if content:
                    content = content.strip()
                    if "<think>" in content:
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content if content else None
            except Exception as e:
                self._error_count += 1
                last_error = e
                error_str = str(e).lower()
                is_rate_limited = "429" in error_str or "rate" in error_str or "too many" in error_str
                if is_rate_limited:
                    _rate_limiter.on_429()
                    backoff_wait = max(0, _rate_limiter._backoff_until - time.monotonic()) + random.uniform(1.0, 3.0)
                    if attempt < self.max_retries:
                        logger.info("Rate limited on attempt %d/%d — waiting %.0fs before retry", attempt + 1, self.max_retries + 1, backoff_wait)
                        await asyncio.sleep(backoff_wait)
                    else:
                        logger.warning("Rate limited on final attempt %d/%d — giving up", attempt + 1, self.max_retries + 1)
                elif attempt < self.max_retries:
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    logger.debug("LLM call attempt %d/%d failed: %s — retrying", attempt + 1, self.max_retries + 1, e)
                else:
                    logger.warning("LLM call failed after %d attempts: %s", self.max_retries + 1, e)

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
            "available": _LANGCHAIN_OPENAI_AVAILABLE,
            "total_calls": self._call_count,
            "total_errors": self._error_count,
            "deep_model": self.deep_model,
            "fast_model": self.fast_model,
            "base_url": self.base_url,
        }

    def _record_llm_metrics(
        self, model: str, prompt_tokens: int, completion_tokens: int, elapsed_ms: float, state: dict | None = None
    ) -> dict | None:
        """Record LLM metrics and return state updates if state provided."""
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
        cost_map = {
            "minimaxai": 0.01,
            "thinkingmachines": 0.01,
            "openai": 0.01,
            "anthropic": 0.01,
            "google": 0.01,
        }
        if state is not None:
            return {
                "total_tokens": state.get("total_tokens", 0) + prompt_tokens + completion_tokens,
                "total_cost": state.get("total_cost", 0.0)
                + (prompt_tokens + completion_tokens) * cost_map.get(model.split("/")[0], 0.01) / 1_000_000,
            }
        return None


def _dict_to_pydantic(schema: Dict[str, Any], name: str) -> Any:
    """Build a pydantic model from a JSON-schema dict for StructuredTool args_schema."""
    from pydantic import BaseModel, Field, create_model

    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", []) if isinstance(schema, dict) else [])
    fields: Dict[str, Any] = {}
    _py_type = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            fields[prop_name] = (Any, None)
            continue
        py_type = _py_type.get(prop_schema.get("type"), Any)
        default = prop_schema.get("default", ...)
        if prop_name not in required:
            default = prop_schema.get("default", None)
            py_type = Optional[py_type]
        description = prop_schema.get("description", "")
        fields[prop_name] = (py_type, Field(default=default, description=description))
    model_name = "".join(p.capitalize() for p in name.split("_")) + "Args"
    try:
        return create_model(model_name, **fields)  # type: ignore[call-overload]
    except Exception:
        return create_model(model_name, **{k: (Any, None) for k in properties})


from src.autfin.llm.validation import (
    TOOL_PARAM_RULES,
    _validate_tool_args,
    _compress_tool_result,
)

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
                # Use NVIDIA NIM with ThinkingMachines Inkling (OpenAI-compatible endpoint)
                nvidia_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
                if nvidia_key:
                    _default_client = LLMClient(
                        deep_model="thinkingmachines/inkling",
                        fast_model="meta/llama-3.1-8b-instruct",
                        temperature=settings.model_temperature,
                        base_url="https://integrate.api.nvidia.com/v1",
                        api_key=nvidia_key,
                    )
                    logger.info("LLM client configured: NVIDIA NIM (thinkingmachines/inkling)")
                else:
                    # No provider configured â€” disabled
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
