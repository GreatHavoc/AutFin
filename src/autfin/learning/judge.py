"""LLM-as-Judge hallucination guard for AutFin.

Verifies that agent-generated answers are grounded in retrieved context chunks
before returning them to the user. Uses a fast LLM to score faithfulness.

Typical flow:
  1. Agent generates answer from research results
  2. judge_answer() checks: is every claim in the answer supported by context?
  3. Returns {faithful: bool, score: float, unsupported_claims: list}

Target: faithfulness > 0.9 (vs ~0.95-0.97 baseline without judge).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_claims(text: str) -> List[str]:
    """Split text into individual factual claims (sentences)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10 and not s.startswith(("#", "*", "-")):
            claims.append(s)
    return claims


def _claim_in_context(claim: str, context_chunks: List[str]) -> bool:
    """Check if a claim is roughly supported by any context chunk.

    Uses keyword overlap as a lightweight proxy. For production,
    replace with embedding similarity > 0.85.
    """
    claim_words = set(claim.lower().split())
    # Remove common stop words
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "not", "this", "that",
            "with", "from", "by", "as", "be", "has", "have", "had", "it",
            "its", "can", "will", "would", "should", "may", "might"}
    claim_words -= stop

    for chunk in context_chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(claim_words & chunk_words)
        if overlap >= max(2, len(claim_words) * 0.5):
            return True
    return False


async def judge_answer(
    answer: str,
    context_chunks: List[str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify answer faithfulness against retrieved context.

    Args:
        answer: The generated answer to check
        context_chunks: The source chunks used to generate the answer
        model: Optional LLM model to use (falls back to keyword check)

    Returns:
        {
            faithful: bool,
            score: float (0-1),
            total_claims: int,
            supported_claims: int,
            unsupported_claims: list[str],
        }
    """
    if not answer or not context_chunks:
        return {
            "faithful": True,
            "score": 1.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
        }

    claims = _extract_claims(answer)
    if not claims:
        return {
            "faithful": True,
            "score": 1.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
        }

    # Try LLM judge first
    try:
        from autfin.llm.client import get_llm_client
        llm = get_llm_client()
        if llm.is_available and model is None:
            return await _llm_judge(answer, context_chunks, llm)
    except Exception:
        pass

    # Fallback: keyword-based faithfulness check
    supported = 0
    unsupported = []
    for claim in claims:
        if _claim_in_context(claim, context_chunks):
            supported += 1
        else:
            unsupported.append(claim)

    score = supported / len(claims) if claims else 1.0
    return {
        "faithful": score >= 0.8,
        "score": round(score, 3),
        "total_claims": len(claims),
        "supported_claims": supported,
        "unsupported_claims": unsupported[:5],  # Cap at 5
    }


async def _llm_judge(
    answer: str,
    context_chunks: List[str],
    llm,
) -> Dict[str, Any]:
    """Use LLM to judge faithfulness."""
    context_text = "\n---\n".join(context_chunks[:10])  # Limit context

    result = await llm.generate_json(
        prompt=(
            f"You are a fact-checking judge. Verify if the ANSWER is fully "
            f"supported by the CONTEXT. Identify any claims in the answer that "
            f"are NOT supported by the context.\n\n"
            f"## CONTEXT\n{context_text[:3000]}\n\n"
            f"## ANSWER\n{answer[:2000]}\n\n"
            f"Return JSON: {{\"faithful\": bool, \"score\": 0.0-1.0, "
            f"\"unsupported_claims\": [\"claim1\", ...]}}"
        ),
        system_prompt=(
            "You are a precise fact-checking judge. Only return valid JSON. "
            "A claim is 'unsupported' if it contradicts the context or adds "
            "information not present in the context."
        ),
    )

    if not result or not isinstance(result, dict):
        # LLM failed, fall back to keyword check
        supported = sum(1 for c in _extract_claims(answer) if _claim_in_context(c, context_chunks))
        total = max(len(_extract_claims(answer)), 1)
        return {
            "faithful": supported / total >= 0.8,
            "score": round(supported / total, 3),
            "total_claims": total,
            "supported_claims": supported,
            "unsupported_claims": [],
        }

    claims = _extract_claims(answer)
    unsupported = result.get("unsupported_claims", [])
    return {
        "faithful": result.get("faithful", True),
        "score": round(result.get("score", 1.0), 3),
        "total_claims": len(claims),
        "supported_claims": len(claims) - len(unsupported),
        "unsupported_claims": unsupported[:5],
    }
