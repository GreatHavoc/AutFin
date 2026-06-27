"""RAG quality evaluation — lightweight alternative to full RAGAS.

Measures key RAG pipeline quality metrics without heavy dependencies:
- Faithfulness: Is the answer grounded in retrieved context?
- Context precision: Are retrieved documents relevant?
- Answer relevancy: Does the answer address the question?

Stores results in SQLite for trend tracking and quality gates.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"

# Quality thresholds (from RAGAS benchmarks)
THRESHOLD_FAITHFULNESS = 0.90
THRESHOLD_CONTEXT_PRECISION = 0.85
THRESHOLD_ANSWER_RELEVANCY = 0.80


def _keyword_overlap(text: str, context: str) -> float:
    """Simple keyword overlap score between answer and context."""
    if not text or not context:
        return 0.0
    text_words = set(text.lower().split())
    ctx_words = set(context.lower().split())
    if not text_words:
        return 0.0
    overlap = text_words & ctx_words
    return len(overlap) / len(text_words)


def _sentence_overlap(text: str, context: str) -> float:
    """Sentence-level overlap — checks if answer sentences appear in context."""
    if not text or not context:
        return 0.0
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return 0.0
    ctx_lower = context.lower()
    supported = sum(1 for s in sentences if s.lower() in ctx_lower)
    return supported / len(sentences)


def evaluate_faithfulness(answer: str, context: str) -> Dict[str, Any]:
    """Evaluate if answer is grounded in retrieved context.

    Uses keyword overlap + sentence containment as proxy for LLM-based faithfulness.
    For production, replace with LLM-as-Judge (see learning/judge.py).

    Args:
        answer: Generated answer text
        context: Retrieved context chunks

    Returns:
        Dict with score (0-1) and explanation
    """
    if not answer or not context:
        return {"score": 0.0, "explanation": "Empty answer or context"}

    keyword_score = _keyword_overlap(answer, context)
    sentence_score = _sentence_overlap(answer, context)

    # Weighted average: sentences more important than keywords
    score = 0.4 * keyword_score + 0.6 * sentence_score

    return {
        "score": round(min(1.0, score), 3),
        "keyword_overlap": round(keyword_score, 3),
        "sentence_overlap": round(sentence_score, 3),
        "explanation": (
            f"Faithfulness: {score:.1%} — "
            f"{sentence_score:.0%} of sentences found in context, "
            f"{keyword_score:.0%} keyword overlap"
        ),
    }


def evaluate_context_precision(query: str, context: str, retrieved_docs: List[str]) -> Dict[str, Any]:
    """Evaluate if retrieved documents are relevant to the query.

    Args:
        query: Original user question
        context: Combined context string
        retrieved_docs: List of individual retrieved document chunks

    Returns:
        Dict with score (0-1) and per-doc relevance
    """
    if not retrieved_docs:
        return {"score": 0.0, "explanation": "No documents retrieved"}

    query_words = set(query.lower().split())
    doc_scores = []

    for doc in retrieved_docs:
        doc_words = set(doc.lower().split())
        if query_words:
            relevance = len(query_words & doc_words) / len(query_words)
        else:
            relevance = 0.0
        doc_scores.append(round(relevance, 3))

    # Precision@k: fraction of docs that are relevant (score > 0.3)
    relevant_count = sum(1 for s in doc_scores if s > 0.3)
    precision = relevant_count / len(doc_scores) if doc_scores else 0.0

    return {
        "score": round(precision, 3),
        "doc_relevance": doc_scores,
        "relevant_count": relevant_count,
        "total_docs": len(retrieved_docs),
        "explanation": (
            f"Context precision: {precision:.1%} — "
            f"{relevant_count}/{len(retrieved_docs)} docs relevant to query"
        ),
    }


def evaluate_answer_relevancy(query: str, answer: str) -> Dict[str, Any]:
    """Evaluate if the answer addresses the question.

    Args:
        query: Original user question
        answer: Generated answer text

    Returns:
        Dict with score (0-1)
    """
    if not query or not answer:
        return {"score": 0.0, "explanation": "Empty query or answer"}

    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())

    # How many query concepts appear in answer
    if query_words:
        coverage = len(query_words & answer_words) / len(query_words)
    else:
        coverage = 0.0

    # Answer length as a sanity check (too short = probably not addressing query)
    word_count = len(answer.split())
    length_factor = min(1.0, word_count / 20)  # Full score at 20+ words

    score = 0.7 * coverage + 0.3 * length_factor

    return {
        "score": round(min(1.0, score), 3),
        "query_coverage": round(coverage, 3),
        "answer_length": word_count,
        "explanation": (
            f"Relevancy: {score:.1%} — "
            f"{coverage:.0%} query concepts in answer, "
            f"{word_count} words"
        ),
    }


def evaluate_rag_quality(
    query: str,
    answer: str,
    context: str,
    retrieved_docs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all RAG quality evaluations.

    Args:
        query: User question
        answer: Generated answer
        context: Retrieved context
        retrieved_docs: Individual doc chunks (optional)

    Returns:
        Dict with all scores + overall quality gate
    """
    faithfulness = evaluate_faithfulness(answer, context)
    relevancy = evaluate_answer_relevancy(query, answer)

    if retrieved_docs:
        precision = evaluate_context_precision(query, context, retrieved_docs)
    else:
        precision = {"score": 0.5, "explanation": "No docs provided for precision check"}

    # Overall quality gate
    all_pass = (
        faithfulness["score"] >= THRESHOLD_FAITHFULNESS
        and precision["score"] >= THRESHOLD_CONTEXT_PRECISION
        and relevancy["score"] >= THRESHOLD_ANSWER_RELEVANCY
    )

    return {
        "faithfulness": faithfulness,
        "context_precision": precision,
        "answer_relevancy": relevancy,
        "overall_pass": all_pass,
        "timestamp": datetime.now().isoformat(),
    }


def store_evaluation(
    conn: sqlite3.Connection,
    query: str,
    evaluation: Dict[str, Any],
) -> None:
    """Store evaluation results in SQLite for trend tracking."""
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                faithfulness REAL,
                context_precision REAL,
                answer_relevancy REAL,
                overall_pass INTEGER,
                timestamp TEXT,
                details_json TEXT
            )
        """)
        scores = evaluation
        conn.execute(
            """INSERT INTO rag_evaluations
               (query, faithfulness, context_precision, answer_relevancy, overall_pass, timestamp, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                query,
                scores.get("faithfulness", {}).get("score", 0),
                scores.get("context_precision", {}).get("score", 0),
                scores.get("answer_relevancy", {}).get("score", 0),
                1 if scores.get("overall_pass") else 0,
                scores.get("timestamp", datetime.now().isoformat()),
                json.dumps(scores),
            ),
        )
    except Exception as e:
        logger.warning("Failed to store RAG evaluation: %s", e)


def get_quality_trend(
    conn: sqlite3.Connection,
    limit: int = 30,
) -> Dict[str, Any]:
    """Get recent RAG quality trend for monitoring."""
    try:
        rows = conn.execute(
            """SELECT faithfulness, context_precision, answer_relevancy, overall_pass, timestamp
               FROM rag_evaluations ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        if not rows:
            return {"evaluations": 0, "avg_faithfulness": 0, "avg_precision": 0, "avg_relevancy": 0}

        avg_f = sum(r[0] for r in rows) / len(rows)
        avg_p = sum(r[1] for r in rows) / len(rows)
        avg_r = sum(r[2] for r in rows) / len(rows)
        pass_rate = sum(1 for r in rows if r[3]) / len(rows)

        return {
            "evaluations": len(rows),
            "avg_faithfulness": round(avg_f, 3),
            "avg_context_precision": round(avg_p, 3),
            "avg_answer_relevancy": round(avg_r, 3),
            "pass_rate": round(pass_rate, 3),
            "thresholds": {
                "faithfulness": THRESHOLD_FAITHFULNESS,
                "context_precision": THRESHOLD_CONTEXT_PRECISION,
                "answer_relevancy": THRESHOLD_ANSWER_RELEVANCY,
            },
        }
    except Exception as e:
        logger.warning("Failed to get quality trend: %s", e)
        return {"evaluations": 0, "error": str(e)}
