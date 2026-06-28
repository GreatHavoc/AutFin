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

    # Find first { ... } block
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    # Count braces to find matching closing brace
    depth = 0
    in_string = False
    escape_next = False
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
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    return None
