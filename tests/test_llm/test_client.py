"""Tests for src/autfin/llm/client.py — LLMClient with fallback routing."""
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
def mock_litellm():
    """Create a mock litellm module with acompletion."""
    mock = MagicMock()
    mock.acompletion = AsyncMock()
    mock.suppress_debug_info = False
    return mock


@pytest.fixture
def client_with_mock(mock_litellm):
    """Create an LLMClient with mocked litellm."""
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        # Force re-import to pick up the mock
        import importlib
        import src.autfin.llm.client as mod

        mod._litellm = mock_litellm
        mod._LITELLM_AVAILABLE = True
        importlib.reload(mod)

        c = mod.LLMClient(
            deep_model="openrouter/test:free",
            fast_model="groq/test",
        )
        yield c, mock_litellm


@pytest.fixture
def mock_completion_response():
    """Create a mock completion response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Test LLM response"
    return response


# ================== INIT TESTS ==================


class TestLLMClientInit:
    def test_default_models(self):
        from src.autfin.llm.client import LLMClient

        c = LLMClient()
        assert c.deep_model == "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
        assert c.fast_model == "nvidia_nim/meta/llama-3.1-8b-instruct"

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


# ================== AVAILABILITY TESTS ==================


class TestAvailability:
    def test_not_available_without_litellm(self):
        with patch.dict(sys.modules, {"litellm": None}):
            import importlib
            import src.autfin.llm.client as mod

            mod._litellm = None
            mod._LITELLM_AVAILABLE = False
            importlib.reload(mod)

            c = mod.LLMClient()
            assert c.is_available is False

    def test_available_with_litellm(self, client_with_mock):
        c, _ = client_with_mock
        assert c.is_available is True


# ================== GENERATE TESTS ==================


class TestGenerate:
    @pytest.mark.asyncio
    async def test_returns_none_when_unavailable(self):
        with patch.dict(sys.modules, {"litellm": None}):
            import importlib
            import src.autfin.llm.client as mod

            mod._litellm = None
            mod._LITELLM_AVAILABLE = False
            importlib.reload(mod)

            c = mod.LLMClient()
            result = await c.generate("test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_successful_generation(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.return_value = mock_completion_response

        result = await c.generate("What is Reliance?")

        assert result == "Test LLM response"
        mock_llm.acompletion.assert_called_once()
        call_kwargs = mock_llm.acompletion.call_args
        assert call_kwargs.kwargs["model"] == "openrouter/test:free"
        # No system_prompt passed, so only user message
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert c._call_count == 1

    @pytest.mark.asyncio
    async def test_with_system_prompt(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.return_value = mock_completion_response

        await c.generate("prompt", system_prompt="You are an analyst")

        call_kwargs = mock_llm.acompletion.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "You are an analyst"

    @pytest.mark.asyncio
    async def test_without_system_prompt(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.return_value = mock_completion_response

        await c.generate("prompt")

        call_kwargs = mock_llm.acompletion.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock

        # Primary fails, fallback succeeds
        mock_llm.acompletion.side_effect = [
            Exception("Rate limited"),
            mock_completion_response,
        ]

        result = await c.generate("test")

        assert result == "Test LLM response"
        assert mock_llm.acompletion.call_count == 2
        assert c._error_count == 1
        assert c._call_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_all_models_fail(self, client_with_mock):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.side_effect = Exception("All failed")

        result = await c.generate("test")

        assert result is None
        assert c._error_count == c.max_retries + 1  # 1 attempt + max_retries retries

    @pytest.mark.asyncio
    async def test_strips_empty_response(self, client_with_mock):
        c, mock_llm = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = ""
        mock_llm.acompletion.return_value = response

        result = await c.generate("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_strips_none_content(self, client_with_mock):
        c, mock_llm = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = None
        mock_llm.acompletion.return_value = response

        result = await c.generate("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_model_override(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.return_value = mock_completion_response

        await c.generate("test", model="custom/model")

        call_kwargs = mock_llm.acompletion.call_args
        assert call_kwargs.kwargs["model"] == "custom/model"

    @pytest.mark.asyncio
    async def test_custom_temperature(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.return_value = mock_completion_response

        await c.generate("test", temperature=0.7)

        call_kwargs = mock_llm.acompletion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_default_temperature(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.return_value = mock_completion_response

        await c.generate("test")

        call_kwargs = mock_llm.acompletion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0


# ================== GENERATE_JSON TESTS ==================


class TestGenerateJSON:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self, client_with_mock):
        c, mock_llm = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"recommendation": "buy", "confidence": 0.8}'
        mock_llm.acompletion.return_value = response

        result = await c.generate_json("Analyze this stock")

        assert result == {"recommendation": "buy", "confidence": 0.8}

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self, client_with_mock):
        c, mock_llm = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        mock_llm.acompletion.return_value = response

        result = await c.generate_json("test")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, client_with_mock):
        c, mock_llm = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "This is not JSON at all"
        mock_llm.acompletion.return_value = response

        result = await c.generate_json("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_llm_fails(self, client_with_mock):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.side_effect = Exception("LLM down")

        result = await c.generate_json("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_appends_json_instruction_to_system(self, client_with_mock):
        c, mock_llm = client_with_mock

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '{"ok": true}'
        mock_llm.acompletion.return_value = response

        await c.generate_json("test", system_prompt="You are an analyst")

        call_kwargs = mock_llm.acompletion.call_args
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
    def test_stats_track_calls(self, client_with_mock, mock_completion_response):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.return_value = mock_completion_response

        import asyncio

        asyncio.run(c.generate("test1"))
        asyncio.run(c.generate("test2"))

        stats = c.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_errors"] == 0

    def test_stats_track_errors(self, client_with_mock):
        c, mock_llm = client_with_mock
        mock_llm.acompletion.side_effect = Exception("fail")

        import asyncio

        asyncio.run(c.generate("test"))

        stats = c.get_stats()
        assert stats["total_errors"] >= 1


# ================== IMPORT FLAG TESTS ==================


class TestImportFlags:
    def test_client_module_has_flags(self):
        import src.autfin.llm.client as mod

        assert hasattr(mod, "_LITELLM_AVAILABLE")
        assert hasattr(mod, "_litellm")
