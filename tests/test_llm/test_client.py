"""Tests for src/autfin/llm/client.py — LLMClient with openai SDK."""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ================== FIXTURES ==================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset global LLM client singleton before each test."""
    import src.autfin.llm.client as mod

    mod._default_client = None
    yield
    mod._default_client = None


@pytest.fixture
def mock_openai_client():
    """Create a mock openai AsyncOpenAI client."""
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    return mock_client


@pytest.fixture
def client_with_mock(mock_openai_client):
    """Create an LLMClient with mocked openai client."""
    import src.autfin.llm.client as mod

    c = mod.LLMClient(
        deep_model="test-deep-model",
        fast_model="test-fast-model",
        base_url="https://test.example.com/v1",
        api_key="test-key",
    )
    c._client = mock_openai_client
    yield c, mock_openai_client


@pytest.fixture
def mock_completion_response():
    """Create a mock openai completion response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Test LLM response"
    response.choices[0].message.tool_calls = None
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


# ================== INIT TESTS ==================


class TestLLMClientInit:
    def test_default_models(self):
        from src.autfin.llm.client import LLMClient

        c = LLMClient()
        assert c.deep_model == "qwen3.6-35b-a3b"
        assert c.fast_model == "meta/llama-3.1-8b-instruct"

    def test_custom_models(self):
        from src.autfin.llm.client import LLMClient

        c = LLMClient(
            deep_model="custom/deep",
            fast_model="custom/fast",
            temperature=0.5,
        )
        assert c.deep_model == "custom/deep"
        assert c.fast_model == "custom/fast"
        assert c.temperature == 0.5

    def test_initial_stats(self):
        from src.autfin.llm.client import LLMClient

        c = LLMClient()
        stats = c.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_errors"] == 0

    def test_base_url_stored(self):
        from src.autfin.llm.client import LLMClient

        c = LLMClient(base_url="https://custom.api.com/v1")
        assert c.base_url == "https://custom.api.com/v1"


# ================== AVAILABILITY TESTS ==================


class TestAvailability:
    def test_available_with_openai_sdk(self):
        from src.autfin.llm.client import LLMClient

        c = LLMClient()
        assert c.is_available is True


# ================== GENERATE TESTS ==================


class TestGenerate:
    @pytest.mark.asyncio
    async def test_successful_generation(self, client_with_mock, mock_completion_response):
        c, mock_client = client_with_mock
        mock_client.chat.completions.create.return_value = mock_completion_response

        result = await c.generate("What is Reliance?")

        assert result == "Test LLM response"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-deep-model"
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert c._call_count == 1

    @pytest.mark.asyncio
    async def test_with_system_prompt(self, client_with_mock, mock_completion_response):
        c, mock_client = client_with_mock
        mock_client.chat.completions.create.return_value = mock_completion_response

        await c.generate("test", system_prompt="You are an analyst")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, client_with_mock, mock_completion_response):
        c, mock_client = client_with_mock

        # Primary fails, fallback succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("Rate limited"),
            mock_completion_response,
        ]

        result = await c.generate("test")

        assert result == "Test LLM response"
        assert mock_client.chat.completions.create.call_count == 2
        assert c._error_count == 1
        assert c._call_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_all_models_fail(self, client_with_mock):
        c, mock_client = client_with_mock
        mock_client.chat.completions.create.side_effect = Exception("All failed")

        result = await c.generate("test")

        assert result is None
        assert c._error_count == c.max_retries + 1

    @pytest.mark.asyncio
    async def test_strips_empty_response(self, client_with_mock):
        c, mock_client = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = ""
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 0
        mock_client.chat.completions.create.return_value = response

        result = await c.generate("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_strips_none_content(self, client_with_mock):
        c, mock_client = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 0
        mock_client.chat.completions.create.return_value = response

        result = await c.generate("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_strips_thinking_tags(self, client_with_mock):
        c, mock_client = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "<think>internal reasoning</think>Final answer"
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = response

        result = await c.generate("test")
        assert result == "Final answer"

    @pytest.mark.asyncio
    async def test_custom_model_override(self, client_with_mock, mock_completion_response):
        c, mock_client = client_with_mock
        mock_client.chat.completions.create.return_value = mock_completion_response

        await c.generate("test", model="custom/model")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "custom/model"

    @pytest.mark.asyncio
    async def test_custom_temperature(self, client_with_mock, mock_completion_response):
        c, mock_client = client_with_mock
        mock_client.chat.completions.create.return_value = mock_completion_response

        await c.generate("test", temperature=0.7)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_default_temperature(self, client_with_mock, mock_completion_response):
        c, mock_client = client_with_mock
        mock_client.chat.completions.create.return_value = mock_completion_response

        await c.generate("test")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0


# ================== GENERATE_JSON TESTS ==================


class TestGenerateJSON:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self, client_with_mock):
        c, mock_client = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"recommendation": "buy", "confidence": 0.8}'
        response.choices[0].message.tool_calls = None
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = response

        result = await c.generate_json("Analyze this stock")

        assert result == {"recommendation": "buy", "confidence": 0.8}

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self, client_with_mock):
        c, mock_client = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        response.choices[0].message.tool_calls = None
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = response

        result = await c.generate_json("test")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, client_with_mock):
        c, mock_client = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "This is not JSON at all"
        response.choices[0].message.tool_calls = None
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = response

        result = await c.generate_json("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_llm_fails(self, client_with_mock):
        c, mock_client = client_with_mock
        mock_client.chat.completions.create.side_effect = Exception("LLM down")

        result = await c.generate_json("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_appends_json_instruction_to_system(self, client_with_mock):
        c, mock_client = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"ok": true}'
        response.choices[0].message.tool_calls = None
        response.usage = MagicMock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = response

        await c.generate_json("test", system_prompt="You are an analyst")

        call_kwargs = mock_client.chat.completions.create.call_args
        system_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "JSON" in system_msg
        assert "You are an analyst" in system_msg


# ================== SINGLETON TESTS ==================


class TestSingleton:
    def test_get_creates_instance(self):
        from src.autfin.llm.client import get_llm_client, LLMClient

        client = get_llm_client()
        assert client is not None
        assert isinstance(client, LLMClient)

    def test_get_returns_same_instance(self):
        from src.autfin.llm.client import get_llm_client

        c1 = get_llm_client()
        c2 = get_llm_client()
        assert c1 is c2

    def test_reset_clears_singleton(self):
        from src.autfin.llm.client import get_llm_client, reset_llm_client

        c1 = get_llm_client()
        reset_llm_client()
        c2 = get_llm_client()
        assert c1 is not c2


# ================== STATS TESTS ==================


class TestStats:
    def test_get_stats_returns_dict(self, client_with_mock):
        c, _ = client_with_mock
        stats = c.get_stats()
        assert isinstance(stats, dict)
        assert "available" in stats
        assert "total_calls" in stats
        assert "total_errors" in stats
        assert "deep_model" in stats
        assert "fast_model" in stats
        assert "base_url" in stats


# ================== HELPERS TESTS ==================


class TestHelpers:
    def test_compress_tool_result_short(self):
        from src.autfin.llm.client import _compress_tool_result

        short = '{"key": "value"}'
        assert _compress_tool_result(short) == short

    def test_compress_tool_result_long(self):
        from src.autfin.llm.client import _compress_tool_result

        long_str = '{"data": "' + "x" * 10000 + '"}'
        result = _compress_tool_result(long_str, max_chars=100)
        assert len(result) < 200
        assert "truncated" in result

    def test_validate_tool_args_no_rules(self):
        from src.autfin.llm.client import _validate_tool_args

        args = {"arbitrary": "value"}
        cleaned, error = _validate_tool_args("unknown_tool", args)
        assert cleaned == args
        assert error is None

    def test_validate_tool_args_with_rules(self):
        from src.autfin.llm.client import _validate_tool_args

        args = {"symbol": "reliance", "action": "BUY", "quantity": 100, "price": 2500.0}
        cleaned, error = _validate_tool_args("execute_order", args)
        assert error is None
        assert cleaned["symbol"] == "RELIANCE"  # uppercased

    def test_validate_tool_args_invalid_action(self):
        from src.autfin.llm.client import _validate_tool_args

        args = {"symbol": "RELIANCE", "action": "INVALID", "quantity": 100, "price": 2500.0}
        _, error = _validate_tool_args("execute_order", args)
        assert error is not None
        assert "INVALID" in error
