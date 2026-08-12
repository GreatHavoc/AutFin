"""AutFin v7.0 — Guardrails Configuration

Input/Output/Tool guardrails for financial safety and compliance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai_harness.guardrails import (
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    ToolGuardrail,
)
from pydantic_ai_harness.guardrails.detectors import (
    secret_data,
    personal_data,
    blocked_keywords,
    for_text,
    for_tool_result_text,
)


# ─── Workspace Path for Tool Guardrails ───
WORKSPACE = Path("./workspace").resolve()


# ─── Custom Guard Functions ───

def stay_in_workspace(call_info: Any) -> GuardrailResult:
    """Ensure file operations stay within workspace."""
    if call_info.name in ('write_file', 'edit_file', 'delete_file'):
        target = Path(str(call_info.args.get('path', ''))).resolve()
        try:
            target.relative_to(WORKSPACE)
        except ValueError:
            return GuardrailResult.block(f'{target} is outside workspace.')
    return GuardrailResult.allow()


def validate_trade_args(call_info: Any) -> GuardrailResult:
    """Validate trade arguments before execution."""
    if call_info.name in ('place_order', 'execute_trade'):
        args = call_info.args
        
        # Check quantity is positive
        qty = args.get('quantity', 0)
        if qty <= 0:
            return GuardrailResult.block('Quantity must be positive')
        
        # Check price is reasonable for limit orders
        order_type = args.get('order_type', '').upper()
        price = args.get('price')
        if order_type in ('LIMIT', 'SL', 'SL-M') and (price is None or price <= 0):
            return GuardrailResult.block(f'{order_type} order requires valid price')
        
        # Check symbol format
        symbol = args.get('tradingsymbol', '')
        if not symbol or not symbol.isalnum():
            return GuardrailResult.block('Invalid trading symbol')
    
    return GuardrailResult.allow()


def sanitize_output_result(result_info: Any) -> GuardrailResult:
    """Sanitize tool results before returning to model."""
    result = result_info.result
    
    # Handle string results
    if isinstance(result, str):
        cleaned = secret_data()(result)
        if cleaned != result:
            return GuardrailResult.replace(cleaned)
    
    # Handle dict results (e.g., trade confirmations)
    if isinstance(result, dict):
        import json
        json_str = json.dumps(result)
        cleaned = secret_data()(json_str)
        if cleaned != json_str:
            import json as json2
            return GuardrailResult.replace(json2.loads(cleaned))
    
    return GuardrailResult.allow()


def enforce_citations(output: Any) -> GuardrailResult:
    """Require citations in research outputs."""
    if not isinstance(output, str):
        return GuardrailResult.allow()  # Let typed output through
    
    # Check for citation patterns
    has_citation = any(marker in output.lower() for marker in [
        'source:', 'ref:', 'citation:', '[1]', '(source', 'according to'
    ])
    
    if not has_citation and len(output) > 200:
        return GuardrailResult.retry(
            'Include at least one source citation for research outputs. '
            'Add "Source: [URL/Report]" or similar citations.'
        )
    
    return GuardrailResult.allow()


def validate_risk_reward(output: Any) -> GuardrailResult:
    """Validate risk/reward ratio in trade recommendations."""
    if not isinstance(output, dict):
        return GuardrailResult.allow()
    
    # Check for trade recommendation structure
    if 'risk_reward' in output:
        rr = output.get('risk_reward')
        if isinstance(rr, (int, float)) and rr < 2.0:
            return GuardrailResult.retry(
                f'Risk/reward ratio {rr:.2f} is below minimum 2.0. '
                'Adjust stop loss or target price.'
            )
    
    if 'conviction' in output:
        conv = output.get('conviction')
        if isinstance(conv, (int, float)) and conv < 0.5:
            return GuardrailResult.retry(
                f'Conviction {conv:.2f} is below minimum 0.5. '
                'Increase confidence or reject trade.'
            )
    
    return GuardrailResult.allow()


# ─── Guardrail Instances ───

# Input Guardrails - validate prompts before model sees them
input_guardrails = InputGuardrail(
    guard=[
        secret_data(),  # Remove API keys, tokens from prompts
        personal_data(only=['us_ssn', 'credit_card', 'iban']),  # Remove PII
        blocked_keywords(['internal-only', 'confidential', 'password', 'secret']),
    ],
    parallel=False,  # Sequential for redaction
)

# Output Guardrails - validate model outputs before returning
output_guardrails = OutputGuardrail(
    guard=[
        for_text(secret_data()),  # Remove secrets from text outputs
        for_text(personal_data(only=['us_ssn', 'credit_card', 'iban']), on_other='allow'),
        enforce_citations,  # Require citations for long outputs
        validate_risk_reward,  # Validate trade recommendations
    ],
)

# Tool Guardrails - validate tool calls and results
tool_guardrails = ToolGuardrail(
    guard=validate_trade_args,  # Validate trade arguments
    result_guard=sanitize_output_result,  # Sanitize tool results
    tools=['place_order', 'execute_trade', 'write_file', 'edit_file'],
    hidden=['delete_everything', 'drop_database'],  # Never expose these
)


# ─── Agent-Specific Guardrail Configs ───

def get_guardrails_for_agent(agent_name: str) -> list:
    """Get appropriate guardrails for an agent."""
    guardrail_map = {
        "market_data": [input_guardrails],  # Only input validation
        "analyze": [input_guardrails, output_guardrails],  # Input + output
        "research": [input_guardrails, output_guardrails],  # Need citations
        "risk": [input_guardrails, output_guardrails, tool_guardrails],  # Full suite
        "debate": [input_guardrails, output_guardrails],
        "conference": [input_guardrails, output_guardrails],
        "memory": [input_guardrails, tool_guardrails],  # File operations
        "quant": [input_guardrails, output_guardrails, tool_guardrails],  # Code execution
        "supervisor": [input_guardrails, output_guardrails, tool_guardrails],  # Full suite
    }
    return guardrail_map.get(agent_name, [input_guardrails, output_guardrails])