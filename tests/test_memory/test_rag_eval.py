"""Tests for RAG quality evaluation (memory/rag_eval.py)."""

import sqlite3

import pytest

from src.autfin.memory.rag_eval import (
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
    evaluate_rag_quality,
    get_quality_trend,
    store_evaluation,
)


class TestEvaluateFaithfulness:
    def test_empty_answer(self):
        result = evaluate_faithfulness("", "some context")
        assert result["score"] == 0.0

    def test_empty_context(self):
        result = evaluate_faithfulness("answer", "")
        assert result["score"] == 0.0

    def test_high_faithfulness(self):
        context = "The stock market closed higher today with Nifty up 1.5%"
        answer = "The stock market closed higher today"
        result = evaluate_faithfulness(answer, context)
        assert result["score"] > 0.5

    def test_low_faithfulness(self):
        context = "The stock market closed higher today"
        answer = "Quantum computing will revolutionize space travel"
        result = evaluate_faithfulness(answer, context)
        assert result["score"] < 0.3

    def test_returns_explanation(self):
        result = evaluate_faithfulness("test answer", "test context")
        assert "explanation" in result
        assert "Faithfulness" in result["explanation"]


class TestEvaluateContextPrecision:
    def test_no_documents(self):
        result = evaluate_context_precision("query", "context", [])
        assert result["score"] == 0.0

    def test_relevant_docs(self):
        query = "Nifty 50 performance today"
        docs = [
            "Nifty 50 index rose 1.5% today",
            "Bank Nifty gained 2% in today's session",
            "Gold prices fell due to strong dollar",
        ]
        result = evaluate_context_precision(query, "", docs)
        assert result["score"] > 0.0
        assert result["relevant_count"] >= 1

    def test_irrelevant_docs(self):
        query = "Nifty 50 performance"
        docs = [
            "Recipe for chocolate cake",
            "How to train a puppy",
            "Benefits of meditation",
        ]
        result = evaluate_context_precision(query, "", docs)
        assert result["score"] < 0.5

    def test_per_doc_relevance(self):
        query = "RELIANCE stock price"
        docs = ["RELIANCE closed at 2500", "TCS results today"]
        result = evaluate_context_precision(query, "", docs)
        assert len(result["doc_relevance"]) == 2


class TestEvaluateAnswerRelevancy:
    def test_empty_query(self):
        result = evaluate_answer_relevancy("", "answer")
        assert result["score"] == 0.0

    def test_empty_answer(self):
        result = evaluate_answer_relevancy("query", "")
        assert result["score"] == 0.0

    def test_relevant_answer(self):
        query = "What is the Nifty 50 PE ratio?"
        answer = "The Nifty 50 PE ratio is currently around 22, which is above the historical average"
        result = evaluate_answer_relevancy(query, answer)
        assert result["score"] > 0.5

    def test_short_answer_penalized(self):
        query = "What is the Nifty 50 PE ratio?"
        answer = "Yes"
        result = evaluate_answer_relevancy(query, answer)
        assert result["answer_length"] == 1
        assert result["score"] < 0.5

    def test_returns_explanation(self):
        result = evaluate_answer_relevancy("test query", "test answer here")
        assert "Relevancy" in result["explanation"]


class TestEvaluateRagQuality:
    def test_full_evaluation(self):
        result = evaluate_rag_quality(
            query="What is the market outlook?",
            answer="The market outlook is bullish with Nifty expected to reach 25000",
            context="Market analysts are bullish on Indian equities. Nifty target 25000.",
            retrieved_docs=["Market outlook bullish", "Nifty target 25000"],
        )
        assert "faithfulness" in result
        assert "context_precision" in result
        assert "answer_relevancy" in result
        assert "overall_pass" in result
        assert "timestamp" in result

    def test_quality_gate(self):
        # Good answer, good context
        result = evaluate_rag_quality(
            query="Nifty 50 performance today",
            answer="Nifty 50 rose 1.5% today closing at 24500",
            context="Nifty 50 index gained 1.5% in today's trading session",
        )
        # Should have reasonable scores
        assert result["faithfulness"]["score"] > 0


class TestStoreAndTrend:
    def test_store_evaluation(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        evaluation = {
            "faithfulness": {"score": 0.95},
            "context_precision": {"score": 0.88},
            "answer_relevancy": {"score": 0.82},
            "overall_pass": True,
            "timestamp": "2026-06-28T10:00:00",
        }
        store_evaluation(conn, "What is Nifty?", evaluation)
        row = conn.execute("SELECT * FROM rag_evaluations").fetchone()
        conn.close()
        assert row is not None
        assert row[1] == "What is Nifty?"  # query
        assert row[2] == 0.95  # faithfulness

    def test_get_quality_trend(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        # Store some evaluations
        for i in range(5):
            evaluation = {
                "faithfulness": {"score": 0.9 + i * 0.01},
                "context_precision": {"score": 0.85},
                "answer_relevancy": {"score": 0.80},
                "overall_pass": True,
                "timestamp": f"2026-06-2{i}T10:00:00",
            }
            store_evaluation(conn, f"query_{i}", evaluation)
        trend = get_quality_trend(conn, limit=10)
        conn.close()
        assert trend["evaluations"] == 5
        assert trend["avg_faithfulness"] > 0.9
        assert "thresholds" in trend

    def test_empty_trend(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        trend = get_quality_trend(conn)
        conn.close()
        assert trend["evaluations"] == 0
