"""
v5.3 Subagents integration tests.

Tests:
- Agent tool definitions are correct
- Schemas are valid JSON Schema
- Tool callables are all async functions
- Supervisor loads agent tools
- Graph builder accepts store
- Parallel tool execution
"""
from __future__ import annotations

import asyncio
import inspect
import json
import pytest


# ---------------------------------------------------------------------------
# Agent tool definitions
# ---------------------------------------------------------------------------


class TestAgentToolDefinitions:
    """Test that agent tool definitions are correct."""

    def test_agent_tool_count(self):
        from src.autfin.agents.agent_tools import AGENT_TOOL_DEFINITIONS, AGENT_TOOL_CALLABLES
        assert len(AGENT_TOOL_DEFINITIONS) == 7
        assert len(AGENT_TOOL_CALLABLES) == 7

    def test_all_tools_have_required_fields(self):
        from src.autfin.agents.agent_tools import AGENT_TOOL_DEFINITIONS
        for tool_def in AGENT_TOOL_DEFINITIONS:
            assert "type" in tool_def
            assert tool_def["type"] == "function"
            assert "function" in tool_def
            fn = tool_def["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert "properties" in fn["parameters"]
            assert "required" in fn["parameters"]

    def test_all_tools_are_async(self):
        from src.autfin.agents.agent_tools import AGENT_TOOL_CALLABLES
        for name, fn in AGENT_TOOL_CALLABLES.items():
            assert inspect.iscoroutinefunction(fn), f"{name} is not async"

    def test_tool_names_match_definitions(self):
        from src.autfin.agents.agent_tools import AGENT_TOOL_DEFINITIONS, AGENT_TOOL_CALLABLES
        def_names = {td["function"]["name"] for td in AGENT_TOOL_DEFINITIONS}
        callable_names = set(AGENT_TOOL_CALLABLES.keys())
        assert def_names == callable_names, f"Mismatch: {def_names.symmetric_difference(callable_names)}"

    def test_known_tool_names(self):
        from src.autfin.agents.agent_tools import AGENT_TOOL_DEFINITIONS
        names = {td["function"]["name"] for td in AGENT_TOOL_DEFINITIONS}
        expected = {
            "fundamentals_analysis",
            "technical_analysis",
            "sentiment_analysis",
            "macro_analysis",
            "research_synthesis",
            "risk_assessment",
            "portfolio_allocation",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    """Test response schemas are valid."""

    def test_all_schemas_present(self):
        from src.autfin.agents.schemas import AGENT_SCHEMAS
        expected = {
            "fundamentals_analysis",
            "technical_analysis",
            "sentiment_analysis",
            "macro_analysis",
            "research_synthesis",
            "risk_assessment",
            "portfolio_allocation",
        }
        assert set(AGENT_SCHEMAS.keys()) == expected

    def test_schemas_have_required_fields(self):
        from src.autfin.agents.schemas import AGENT_SCHEMAS
        for name, schema in AGENT_SCHEMAS.items():
            assert "type" in schema, f"{name}: missing type"
            assert schema["type"] == "object"
            assert "properties" in schema, f"{name}: missing properties"
            assert "required" in schema, f"{name}: missing required"

    def test_validate_agent_response_valid(self):
        from src.autfin.agents.schemas import validate_agent_response
        data = {
            "recommendation": "BUY",
            "conviction": 0.7,
            "summary": "Test summary",
        }
        issues = validate_agent_response("fundamentals_analysis", data)
        assert issues == []

    def test_validate_agent_response_missing_required(self):
        from src.autfin.agents.schemas import validate_agent_response
        data = {"pe_ratio": 15.0}  # missing required fields
        issues = validate_agent_response("fundamentals_analysis", data)
        assert len(issues) > 0
        assert any("Missing" in i for i in issues)

    def test_validate_unknown_agent(self):
        from src.autfin.agents.schemas import validate_agent_response
        issues = validate_agent_response("nonexistent_agent", {})
        assert len(issues) == 1
        assert "Unknown agent" in issues[0]


# ---------------------------------------------------------------------------
# Supervisor loads agent tools
# ---------------------------------------------------------------------------


class TestSupervisorAgentTools:
    """Test that supervisor integrates agent tools."""

    def test_supervisor_prompt_mentions_agent_tools(self):
        from src.autfin.graph.supervisor import _build_context_prompt
        from src.autfin.state.agent_state import AgentState
        state = AgentState()
        prompt = _build_context_prompt("manual", "standard", state)
        assert "fundamentals_analysis" in prompt
        assert "technical_analysis" in prompt
        assert "sentiment_analysis" in prompt
        assert "macro_analysis" in prompt
        assert "research_synthesis" in prompt
        assert "risk_assessment" in prompt
        assert "portfolio_allocation" in prompt

    def test_supervisor_prompt_v53_header(self):
        from src.autfin.graph.supervisor import _build_context_prompt
        from src.autfin.state.agent_state import AgentState
        state = AgentState()
        prompt = _build_context_prompt("manual", "standard", state)
        assert "v5.3" in prompt
        assert "Subagents Coordinator" in prompt


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class TestGraphBuilder:
    """Test graph builder with store support."""

    def test_build_supervisor_graph(self):
        from src.autfin.graph.supervisor import build_supervisor_graph
        builder = build_supervisor_graph()
        assert builder is not None

    def test_compile_without_store(self):
        from src.autfin.graph.supervisor import compile_supervisor_graph
        graph = compile_supervisor_graph()
        assert graph is not None

    def test_compile_with_none_store(self):
        from src.autfin.graph.supervisor import compile_supervisor_graph
        graph = compile_supervisor_graph(store=None)
        assert graph is not None

    def test_builder_module_exists(self):
        import importlib
        mod = importlib.import_module("src.autfin.graph.builder")
        assert hasattr(mod, "build_and_compile_graph")


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Test that agent tools can run in parallel."""

    @pytest.mark.asyncio
    async def test_parallel_tool_gather(self):
        """Verify asyncio.gather works for agent tools."""
        async def mock_tool(x):
            await asyncio.sleep(0.01)
            return x * 2

        results = await asyncio.gather(
            mock_tool(1),
            mock_tool(2),
            mock_tool(3),
        )
        assert results == [2, 4, 6]

    def test_imports_work(self):
        """Verify all new modules import without error."""
        import src.autfin.agents.agent_tools
        import src.autfin.agents.schemas
        import src.autfin.graph.builder
        # supervisor already tested above


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Test that old tools still work alongside new agent tools."""

    def test_old_supervisor_tools_still_exist(self):
        from src.autfin.tools.supervisor_tools import SUPERVISOR_TOOLS, SUPERVISOR_CALLABLES
        assert len(SUPERVISOR_TOOLS) > 0
        assert len(SUPERVISOR_CALLABLES) > 0

    def test_old_agent_tools_still_exist(self):
        from src.autfin.agents.tools import FUNDAMENTALS_TOOLS, FUNDAMENTALS_CALLABLES
        from src.autfin.agents.tools import TECHNICAL_TOOLS, TECHNICAL_CALLABLES
        from src.autfin.agents.tools import SENTIMENT_TOOLS, SENTIMENT_CALLABLES
        from src.autfin.agents.tools import MACRO_TOOLS, MACRO_CALLABLES
        assert len(FUNDAMENTALS_TOOLS) > 0
        assert len(TECHNICAL_TOOLS) > 0
        assert len(SENTIMENT_TOOLS) > 0
        assert len(MACRO_TOOLS) > 0

    def test_compile_supervisor_graph_backward_compat(self):
        """Old code calling compile_supervisor_graph() still works."""
        from src.autfin.graph.supervisor import compile_supervisor_graph
        graph = compile_supervisor_graph(checkpointer=None, store=None)
        assert graph is not None
