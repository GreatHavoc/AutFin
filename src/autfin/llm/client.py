"""
Centralized LLM client for AutFin.

Uses litellm for all provider routing. The caller chooses the right model for the task.
Returns None on failure — agents handle degradation autonomously.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level token accumulator (read by budget guard node)
_token_accumulator = {"tokens": 0, "cost": 0.0}


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
                # Detect rate limiting (429) — wait longer before retry
                is_rate_limited = "429" in error_str or "rate" in error_str or "too many" in error_str
                if attempt < self.max_retries:
                    wait_time = 2.0 if is_rate_limited else 0.5
                    logger.debug(
                        "LLM call attempt %d/%d failed on %s%s: %s — retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, target_model,
                        " (RATE LIMITED)" if is_rate_limited else "", e, wait_time,
                    )
                    import asyncio
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
                            "content": result_str[:8000],
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


async def _execute_tool(
    tool_call: Any,
    tool_callables: Dict[str, Callable],
) -> Any:
    """Execute a single tool call from the LLM's response."""
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)
    fn = tool_callables.get(func_name)
    if fn is None:
        return f"Error: unknown tool '{func_name}'"
    result = fn(**func_args)
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
