"""
Tests for Supervisor Graph — context prompt building, learning section, constants.
All tests run locally with mocked dependencies.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.autfin.graph.supervisor import (
    _build_context_prompt,
    _learning_section,
    MAX_GRAPH_ITERATIONS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_max_graph_iterations_is_positive(self):
        assert MAX_GRAPH_ITERATIONS == 20
        assert MAX_GRAPH_ITERATIONS > 0


# ---------------------------------------------------------------------------
# _learning_section tests
# ---------------------------------------------------------------------------

class TestLearningSection:
    def test_empty_learning_context(self):
        state = {}
        result = _learning_section(state)
        assert result == ""

    def test_empty_learning_context_dict(self):
        state = {"learning_context": {}}
        result = _learning_section(state)
        assert result == ""

    def test_calibration_section(self):
        state = {
            "learning_context": {
                "calibration": {
                    "high": {"actual_win_rate": 0.7, "expected_win_rate": 0.6, "delta": 0.1, "status": "over"},
                    "medium": {"actual_win_rate": 0.5, "expected_win_rate": 0.5, "delta": 0.0, "status": "on_track"},
                }
            }
        }
        result = _learning_section(state)
        assert "SELF-LEARNING CONTEXT" in result
        assert "Conviction calibration:" in result
        assert "high:" in result
        assert "actual=0.7" in result
        assert "expected=0.6" in result

    def test_patterns_section(self):
        state = {
            "learning_context": {
                "patterns": [
                    {"pattern": "sector_concentration", "recommendation": "Reduce banking exposure"},
                    {"pattern": "overtrading", "recommendation": "Reduce trade frequency"},
                ]
            }
        }
        result = _learning_section(state)
        assert "Detected patterns:" in result
        assert "sector_concentration" in result
        assert "Reduce banking exposure" in result

    def test_recent_trades_section(self):
        state = {
            "learning_context": {
                "recent_trades": [{"symbol": "RELIANCE"}, {"symbol": "TCS"}]
            }
        }
        result = _learning_section(state)
        assert "Recent trades in journal: 2" in result

    def test_full_learning_context(self):
        state = {
            "learning_context": {
                "calibration": {"high": {"actual_win_rate": 0.7, "expected_win_rate": 0.6, "delta": 0.1, "status": "over"}},
                "patterns": [{"pattern": "test_pattern", "recommendation": "test_rec"}],
                "recent_trades": [{"symbol": "RELIANCE"}],
            }
        }
        result = _learning_section(state)
        assert "SELF-LEARNING CONTEXT" in result
        assert "Conviction calibration:" in result
        assert "Detected patterns:" in result
        assert "Recent trades in journal: 1" in result
        assert "calibrate_conviction" in result

    def test_patterns_limited_to_five(self):
        patterns = [{"pattern": f"p{i}", "recommendation": f"r{i}"} for i in range(10)]
        state = {"learning_context": {"patterns": patterns}}
        result = _learning_section(state)
        # Should only include first 5
        assert "p0" in result
        assert "p4" in result
        assert "p5" not in result


# ---------------------------------------------------------------------------
# _build_context_prompt tests
# ---------------------------------------------------------------------------

class TestBuildContextPrompt:
    def test_morning_routine_context(self):
        state = {
            "holdings": [{"symbol": "RELIANCE"}, {"symbol": "TCS"}],
            "available_cash": 50000.0,
        }
        prompt = _build_context_prompt("morning_routine", "standard", state)
        assert "MORNING ROUTINE" in prompt
        assert "Holdings: 2" in prompt
        assert "Cash: ₹50,000.00" in prompt
        assert "depth=standard" in prompt

    def test_price_alert_context(self):
        state = {
            "alert_symbol": "RELIANCE",
            "price_move_pct": 0.05,
        }
        prompt = _build_context_prompt("price_alert", "quick", state)
        assert "PRICE ALERT" in prompt
        assert "RELIANCE" in prompt
        assert "5.0%" in prompt

    def test_quarterly_review_context(self):
        state = {
            "holdings": [{"symbol": "RELIANCE"}],
        }
        prompt = _build_context_prompt("quarterly_review", "deep", state)
        assert "QUARTERLY REVIEW" in prompt
        assert "Deep dive every holding" in prompt
        assert "1 positions" in prompt

    def test_manual_context(self):
        state = {}
        prompt = _build_context_prompt("manual", "standard", state)
        assert "MANUAL REQUEST" in prompt
        assert "depth=standard" in prompt

    def test_unknown_context_falls_back_to_manual(self):
        state = {}
        prompt = _build_context_prompt("unknown_context", "quick", state)
        assert "UNKNOWN CONTEXT" in prompt
        assert "MANUAL REQUEST" not in prompt  # should be unknown, not manual

    def test_base_prompt_contains_tools(self):
        state = {}
        prompt = _build_context_prompt("manual", "standard", state)
        assert "27 tools" in prompt
        # Old tools still present
        assert "get_portfolio_state" in prompt
        assert "screen_candidates" in prompt
        assert "research_stock" in prompt
        assert "build_stock_pitch" in prompt
        assert "request_human_approval" in prompt
        # v5.3 agent tools present
        assert "fundamentals_analysis" in prompt
        assert "technical_analysis" in prompt
        assert "sentiment_analysis" in prompt
        assert "macro_analysis" in prompt
        assert "research_synthesis" in prompt
        assert "risk_assessment" in prompt
        assert "portfolio_allocation" in prompt

    def test_learning_section_included(self):
        state = {
            "learning_context": {
                "calibration": {"high": {"actual_win_rate": 0.7, "expected_win_rate": 0.6, "delta": 0.1, "status": "over"}}
            }
        }
        prompt = _build_context_prompt("manual", "standard", state)
        assert "SELF-LEARNING CONTEXT" in prompt

    def test_depth_tiers_documented(self):
        state = {}
        prompt = _build_context_prompt("manual", "standard", state)
        assert "quick=skip DCF/peers/moat" in prompt
        assert "standard=all but DCF" in prompt
        assert "deep=everything" in prompt

    def test_risk_rules_documented(self):
        state = {}
        prompt = _build_context_prompt("manual", "standard", state)
        assert "risk/reward ≥ 2:1" in prompt
        assert "Conviction ≥ 0.5" in prompt
        assert "Sector concentration max 30%" in prompt
        assert "Cash reserve 10%" in prompt


# ---------------------------------------------------------------------------
# _build_context_prompt — edge cases
# ---------------------------------------------------------------------------

class TestBuildContextPromptEdgeCases:
    def test_empty_holdings(self):
        state = {"holdings": [], "available_cash": 0}
        prompt = _build_context_prompt("morning_routine", "standard", state)
        assert "Holdings: 0" in prompt

    def test_large_cash_amount(self):
        state = {"holdings": [], "available_cash": 1000000.0}
        prompt = _build_context_prompt("morning_routine", "standard", state)
        assert "Cash: ₹1,000,000.00" in prompt

    def test_price_alert_zero_move(self):
        state = {"alert_symbol": "TCS", "price_move_pct": 0.0}
        prompt = _build_context_prompt("price_alert", "quick", state)
        assert "0.0%" in prompt

    def test_price_alert_negative_move(self):
        state = {"alert_symbol": "INFY", "price_move_pct": -0.03}
        prompt = _build_context_prompt("price_alert", "quick", state)
        assert "-3.0%" in prompt
