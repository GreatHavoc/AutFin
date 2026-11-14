"""Untrusted-input defenses, attached to the graph rather than promised by it.

Two gaps this closes:

1. **Indirect prompt injection.** Every tool that returns third-party text —
   the Obscura browser scrape in `tools/browser_agent.py`, the sentiment tools
   that read Reddit and Moneycontrol, yfinance news — hands the model text an
   outside party wrote. `PromptInjectionDefender` classifies each locally
   executed tool result and withholds the ones that carry instructions aimed at
   the agent, replacing them with a note so the content never reaches the model.
   It was installed as a dependency in the upgrade and never attached, which is
   why the plan called this the highest unmitigated risk in the repo.

2. **Agent-written artifacts.** Memory is injected into every run's context, so
   anything an agent writes there is a second chance to plant an instruction.
   `ArtifactScanner` runs over what is stored, and flags the patterns that
   matter — role override, exfiltration, invisible unicode, injection, HTML
   comments.

Both are process-wide singletons: the defender is stateless per result, and the
scanner is a set of patterns, so one instance shared by every agent is the same
work done once per call rather than per agent.
"""
from __future__ import annotations

from pydantic_ai_harness.artifact_scanning import ArtifactScanner
from pydantic_ai_harness.prompt_injection_defender import PromptInjectionDefender

# Explicit id: a defender on a durable-capable agent needs one, and naming it now
# means a durable capability can be added later without a migration.
DEFENDER_ID = "injection_defense"

# Pattern detection only. `semantic_detection=True` adds a local ML classifier
# that also catches injection under unrecognised fields, at the cost of loading a
# model into every process; revisit when the scraped corpus outgrows patterns.
_injection_defender: PromptInjectionDefender | None = None
_artifact_scanner: ArtifactScanner | None = None


def get_injection_defense() -> PromptInjectionDefender:
    """Classify tool results and withhold the ones carrying instructions."""
    global _injection_defender
    if _injection_defender is None:
        _injection_defender = PromptInjectionDefender(
            block_high_risk=True, id=DEFENDER_ID
        )
    return _injection_defender


def get_artifact_scanner() -> ArtifactScanner:
    """Scan what the agents write into memory before it is stored."""
    global _artifact_scanner
    if _artifact_scanner is None:
        _artifact_scanner = ArtifactScanner()
    return _artifact_scanner