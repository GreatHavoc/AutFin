"""
Tool argument validation utilities.

Provides validation and compression helpers for tool arguments and results.
Used by subagents that need custom validation beyond pydantic args_schema.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from src.autfin.utils import extract_json


# Per-tool input validation schemas
TOOL_PARAM_RULES: Dict[str, Dict[str, Dict[str, Any]]] = {
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


def _validate_tool_args(func_name: str, args: dict) -> Tuple[dict, Optional[str]]:
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


__all__ = [
    "TOOL_PARAM_RULES",
    "_validate_tool_args",
    "_compress_tool_result",
]