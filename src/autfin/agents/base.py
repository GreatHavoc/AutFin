"""AutFin v7.0 — Agent Factory & Base Configuration

Creates type-safe agents with shared configuration, models, and capabilities.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Literal, TypeVar

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching, ModelFallback
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import UsageLimits

from autfin.agents.guardrails import get_guardrails_for_agent
from autfin.types import MODEL_CONFIG, AgentDeps

OutputT = TypeVar("OutputT")

# NVIDIA NIM is OpenAI-wire-compatible at this endpoint. pydantic-ai ships no native
# NVIDIA provider class, so we use the built-in OpenAIProvider pointed at NIM.
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _nvidia_api_key() -> str:
    load_dotenv()  # ensure .env (NVIDIA_NIM_API_KEY) is loaded regardless of import order
    return os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY") or "dummy-key"


def create_nvidia_model(model_name: str, **kwargs) -> OpenAIChatModel:
    """Create OpenAI-compatible model for NVIDIA NIM endpoint."""
    provider = OpenAIProvider(base_url=_NVIDIA_BASE_URL, api_key=_nvidia_api_key())
    return OpenAIChatModel(model_name=model_name, provider=provider, **kwargs)


def get_model(model_type: Literal["deep", "fast", "reasoning", "code"] = "deep") -> OpenAIChatModel:
    """Get model by type from configuration."""
    model_name = getattr(MODEL_CONFIG, model_type)
    return create_nvidia_model(model_name)


def create_agent[OutputT](
    *,
    name: str,
    output_type: type[OutputT],
    instructions: str,
    model_type: Literal["deep", "fast", "reasoning", "code"] = "deep",
    tools: list[Callable] | None = None,
    toolsets: list[AbstractToolset[AgentDeps]] | None = None,
    capabilities: list[Any] | None = None,
    output_validators: list[Callable] | None = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    retries: int = 2,
) -> Agent[AgentDeps, OutputT]:
    """
    Factory for creating typed agents with shared configuration.

    Args:
        name: Agent name (used for identification)
        output_type: Pydantic model for structured output
        instructions: System instructions for the agent
        model_type: Model tier - "deep" (default), "fast", "reasoning", "code"
        tools: List of tool functions to register
        toolsets: List of Pydantic AI toolsets
        capabilities: List of Pydantic AI capabilities
        output_validators: List of output validator functions
        model_settings: Model settings (temperature, max_tokens, etc.)
        usage_limits: Usage limits for the agent
        retries: Tool retry count

    Returns:
        Configured Agent instance
    """
    model = get_model(model_type)

    default_settings = ModelSettings(
        temperature=0.0,
        max_tokens=8192,
    )

    # Guardrails are harness capabilities, so they belong in the same list as
    # Thinking/Memory/SubAgents. Without this, guardrails.py defined a full
    # input/output/tool suite that no agent ever ran.
    from pydantic_ai.capabilities import HandleDeferredToolCalls, Instrumentation

    from autfin.agents.order_approval import resolve_order_approvals
    from autfin.agents.safety import get_injection_defense
    from autfin.agents.spend import get_spend_limits
    from autfin.agents.tool_output import get_tool_output_limits

    # Guardrails and the approval handler are capabilities, so they belong in the
    # same list as Thinking/Memory/SubAgents. Order tools are wrapped with
    # ApprovalRequiredToolset in tools/__init__, so they pause here: paper mode is
    # approved automatically, live mode bubbles up as DeferredToolRequests.
    # Instrumentation rides along per agent so spans are emitted by every
    # entrypoint (tests, `python -m autfin`), not just run.py's global
    # logfire.instrument_pydantic_ai() call, and it no-ops until a tracer
    # provider is configured.
    # The defender rides with them: every tool that returns third-party text
    # (browser scrape, sentiment, news) would otherwise hand the model words an
    # outside party wrote.
    # So does the fallback: NIM's throttle is per-model and measured in tokens per
    # minute, so a throttled deep tier used to end the whole run. The two tiers
    # are different models, so the other one can serve the step. Skipped when the
    # tiers are configured to the same id, where it would only retry the throttled
    # endpoint.
    fallback_name = "fast" if model_type in ("deep", "reasoning") else "deep"
    agent_caps = [
        HandleDeferredToolCalls(handler=resolve_order_approvals),
        # One shared instance, not one per agent: the default store is built
        # per capability, so a fresh one here would give every agent its own
        # daily ceiling.
        get_spend_limits(),
        get_tool_output_limits(),
        get_injection_defense(),
        # Prompt caching through the unified ModelSettings `cache` flag. Verified
        # against the NIM endpoint: the setting is accepted, not rejected. Whether
        # it produces cache reads is unproven — the throttle killed the second
        # turn of the probe — so this is the documented lever switched on, not a
        # measured saving. No-op on models that cache implicitly.
        Caching(),
        Instrumentation(),
    ]
    if getattr(MODEL_CONFIG, model_type) != getattr(MODEL_CONFIG, fallback_name):
        agent_caps.append(ModelFallback(get_model(fallback_name)))

    agent = Agent[AgentDeps, OutputT](
        capabilities=[
            *agent_caps,
            *(capabilities or []),
            *get_guardrails_for_agent(name),
        ],
        name=name,
        model=model,
        deps_type=AgentDeps,
        output_type=output_type,
        instructions=instructions,
        model_settings=model_settings or default_settings,
        retries=retries,
        toolsets=toolsets or [],
    )

    if tools:
        for tool in tools:
            agent.tool(tool)

    if output_validators:
        for validator in output_validators:
            agent.output_validator(validator)

    if usage_limits:
        agent = agent.override(usage_limits=usage_limits)

    # Set agent name for identification
    agent.name = name

    return agent


def create_subagent[OutputT](
    *,
    name: str,
    description: str,
    output_type: type[OutputT],
    instructions: str,
    model_type: Literal["deep", "fast", "reasoning", "code"] = "deep",
    tools: list[Callable] | None = None,
    capabilities: list[Any] | None = None,
    usage_limits: UsageLimits | None = None,
    max_calls: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[Agent[AgentDeps, OutputT], dict]:
    """
    Create a sub-agent with metadata for SubAgents capability.

    Returns:
        Tuple of (agent, metadata_dict) where metadata contains
        name, description, and run controls for SubAgent constructor.
    """
    agent = create_agent(
        name=name,
        output_type=output_type,
        instructions=instructions,
        model_type=model_type,
        tools=tools,
        capabilities=capabilities,
        usage_limits=usage_limits,
    )

    metadata = {
        "name": name,
        "description": description,
    }

    if usage_limits:
        metadata["usage_limits"] = usage_limits
    if max_calls:
        metadata["max_calls"] = max_calls
    if timeout_seconds:
        metadata["timeout_seconds"] = timeout_seconds

    return agent, metadata