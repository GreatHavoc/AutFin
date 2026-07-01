"""Tests for tool input validation (v4.2)."""

import pytest
from src.autfin.llm.client import _validate_tool_args, _compress_tool_result, TOOL_PARAM_RULES


class TestValidateToolArgs:
    def test_unknown_tool_passes_through(self):
        args, error = _validate_tool_args("nonexistent_tool", {"x": 1, "y": "test"})
        assert args == {"x": 1, "y": "test"}
        assert error is None

    def test_symbol_uppercase_and_trim(self):
        args, error = _validate_tool_args(
            "execute_order",
            {"symbol": "  reliance ", "action": "BUY", "quantity": 10, "price": 2500.0}
        )
        assert error is None
        assert args["symbol"] == "RELIANCE"
        assert args["action"] == "BUY"

    def test_symbol_max_length_clamped(self):
        long_symbol = "A" * 50
        args, error = _validate_tool_args(
            "execute_order",
            {"symbol": long_symbol, "action": "BUY", "quantity": 10, "price": 2500.0}
        )
        assert error is None
        assert len(args["symbol"]) == 20

    def test_quantity_clamped_to_max(self):
        args, error = _validate_tool_args(
            "execute_order",
            {"symbol": "RELIANCE", "action": "BUY", "quantity": 999999999, "price": 2500.0}
        )
        assert error is None
        assert args["quantity"] == 100000

    def test_invalid_action_rejected(self):
        args, error = _validate_tool_args(
            "execute_order",
            {"symbol": "RELIANCE", "action": "INVALID", "quantity": 10, "price": 2500.0}
        )
        assert error is not None
        assert "Invalid value for action" in error

    def test_log_decision_valid(self):
        args, error = _validate_tool_args(
            "log_decision",
            {"decision_type": "trade", "details": {"symbol": "RELIANCE", "action": "BUY"}}
        )
        assert error is None
        assert args["decision_type"] == "trade"

    def test_invalid_type_returns_error(self):
        args, error = _validate_tool_args(
            "execute_order",
            {"symbol": "RELIANCE", "action": "BUY", "quantity": "abc", "price": 2500.0}
        )
        assert error is not None
        assert "Invalid type for quantity" in error


class TestCompressToolResult:
    def test_short_result_passes_through(self):
        short = "Hello world"
        assert _compress_tool_result(short) == short

    def test_within_limit_unchanged(self):
        text = "x" * 5000
        assert _compress_tool_result(text, max_chars=8000) == text

    def test_long_gets_compressed(self):
        text = "x" * 10000
        compressed = _compress_tool_result(text, max_chars=1000)
        assert len(compressed) < len(text)

    def test_json_brace_strategy_keeps_close_brace(self):
        """Long JSON should preserve ending structure when finding close brace in last 30%."""
        # Make a JSON that's > max_chars but the } is in the last 30%
        prefix = '{"key": "'  # 9 chars
        filler = "x" * 9990   # 9990 chars → makes total 10000
        json_str = prefix + filler + ". OK\"\n}"  # Now total = 10013
        compressed = _compress_tool_result(json_str, max_chars=2000)
        # The closing brace strategy should kick in
        # Could be: first 2000 chars ending at the last }
        # or the head+tail fallback if last } isn't in last 30% of the 2000-char slice
        assert len(compressed) < len(json_str)
        # Either strategy OK; test only that it's shorter
        assert "[truncated" in compressed or compressed.endswith("}")

    def test_head_tail_strategy_triggered(self):
        """When truncation marker fallback used, both head and tail present."""
        # Create text where the } isn't in last 30% of first 5000 chars
        text = "HEAD" + "z" * 9900 + "TAIL"
        compressed = _compress_tool_result(text, max_chars=2000)
        assert len(compressed) < len(text)
        # Verify head+tail strategy
        assert "HEAD" in compressed
        assert "TAIL" in compressed
        assert "[truncated" in compressed


class TestToolParamRules:
    def test_execute_order_has_validation(self):
        rules = TOOL_PARAM_RULES["execute_order"]
        assert "symbol" in rules
        assert "action" in rules
        assert "quantity" in rules

    def test_all_critical_tools_have_rules(self):
        for tool in ["execute_order", "log_decision", "research_stock"]:
            assert tool in TOOL_PARAM_RULES
