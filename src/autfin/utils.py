"""Shared utilities for AutFin."""

import json
import re
from typing import Optional, Dict, Any


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract a JSON object from LLM output.

    Handles:
    - Plain JSON
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Text before/after the JSON object
    - Qwen thinking mode artifacts
    - JSON embedded in analysis text

    Returns:
        Parsed dict or None if no valid JSON found
    """
    if not text:
        return None

    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code fences
    fence_pattern = r"```(?:json)?\s*\n(.*?)\n\s*```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Find first { ... } block with proper brace counting
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    best_candidate = None

    for i, ch in enumerate(text[brace_start:], start=brace_start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    # Keep the first valid JSON object found
                    if best_candidate is None:
                        best_candidate = parsed
                    # Prefer dicts with meaningful keys (not just {})
                    if isinstance(parsed, dict) and len(parsed) >= 3:
                        return parsed
                except json.JSONDecodeError:
                    # If this brace pair failed, try from the next { after this }
                    continue

    # Return first valid candidate even if small
    if best_candidate is not None:
        return best_candidate

    # Last resort: try to find any { ... } anywhere in text
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue

    return None
