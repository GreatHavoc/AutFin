"""
Tests for HybridSentimentAnalyzer (FinBERT + VADER + keyword fallback).
All tests run locally — no network calls, no model downloads.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.autfin.tools.sentiment_analyzer import (
    HybridSentimentAnalyzer,
    get_sentiment_analyzer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Fresh HybridSentimentAnalyzer instance with keyword fallback."""
    a = HybridSentimentAnalyzer()
    # Force keyword fallback by clearing any loaded models
    a._finbert = None
    a._vader = None
    return a


# ---------------------------------------------------------------------------
# Keyword fallback tests
# ---------------------------------------------------------------------------

class TestKeywordFallback:
    """Tests for the keyword-based fallback analyzer."""

    def test_positive_text(self, analyzer):
        result = analyzer._keyword_analyze("Stock gains rise rally strong bullish growth profit")
        assert result["score"] > 0
        assert result["label"] == "positive"
        assert result["method"] == "keyword"

    def test_negative_text(self, analyzer):
        result = analyzer._keyword_analyze("Stock drops fall decline weak bearish loss risk crisis")
        assert result["score"] < 0
        assert result["label"] == "negative"

    def test_neutral_text(self, analyzer):
        result = analyzer._keyword_analyze("The company held its annual meeting yesterday")
        assert result["score"] == 0.0
        assert result["label"] == "neutral"

    def test_empty_text(self, analyzer):
        result = analyzer._keyword_analyze("")
        assert result["score"] == 0.0
        assert result["label"] == "neutral"

    def test_score_range(self, analyzer):
        result = analyzer._keyword_analyze("gain rise rally strong bullish")
        assert -1.0 <= result["score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_mixed_text(self, analyzer):
        result = analyzer._keyword_analyze("Stock gains but risks remain and concerns about decline")
        assert -1.0 <= result["score"] <= 1.0


# ---------------------------------------------------------------------------
# VADER tests (mocked)
# ---------------------------------------------------------------------------

class TestVADER:
    """Tests for VADER analysis with mocked analyzer."""

    def test_vader_positive(self, analyzer):
        mock_vader = MagicMock()
        mock_vader.polarity_scores.return_value = {"pos": 0.8, "neg": 0.0, "neu": 0.2, "compound": 0.75}
        analyzer._vader = mock_vader

        result = analyzer._vader_analyze("Stock gains rally strong bullish profit")
        assert result["score"] > 0
        assert result["method"] == "vader"
        mock_vader.polarity_scores.assert_called_once()

    def test_vader_negative(self, analyzer):
        mock_vader = MagicMock()
        mock_vader.polarity_scores.return_value = {"pos": 0.0, "neg": 0.8, "neu": 0.2, "compound": -0.75}
        analyzer._vader = mock_vader

        result = analyzer._vader_analyze("Stock crashes losses decline bearish")
        assert result["score"] < 0

    def test_vader_neutral(self, analyzer):
        mock_vader = MagicMock()
        mock_vader.polarity_scores.return_value = {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
        analyzer._vader = mock_vader

        result = analyzer._vader_analyze("The company held its meeting")
        assert result["score"] == 0.0

    def test_vader_fallback_to_keyword(self, analyzer):
        """When VADER is not loaded, should fall back to keyword."""
        analyzer._vader = None
        result = analyzer._vader_analyze("Stock gains rally bullish profit")
        assert result["method"] == "keyword"


# ---------------------------------------------------------------------------
# FinBERT tests (mocked)
# ---------------------------------------------------------------------------

class TestFinBERT:
    """Tests for FinBERT analysis with mocked pipeline."""

    def test_finbert_positive(self, analyzer):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "positive", "score": 0.92},
            {"label": "neutral", "score": 0.05},
            {"label": "negative", "score": 0.03},
        ]
        analyzer._finbert = mock_pipeline

        result = analyzer._finbert_analyze("Reliance reports record quarterly profit")
        assert result["score"] > 0
        assert result["label"] == "positive"
        assert result["method"] == "finbert"

    def test_finbert_negative(self, analyzer):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "negative", "score": 0.85},
            {"label": "neutral", "score": 0.10},
            {"label": "positive", "score": 0.05},
        ]
        analyzer._finbert = mock_pipeline

        result = analyzer._finbert_analyze("TCS shares crash after poor earnings")
        assert result["score"] < 0
        assert result["label"] == "negative"

    def test_finbert_fallback_to_vader(self, analyzer):
        """When FinBERT is not loaded, should fall back to VADER."""
        analyzer._finbert = None
        mock_vader = MagicMock()
        mock_vader.polarity_scores.return_value = {"compound": 0.5}
        analyzer._vader = mock_vader

        result = analyzer._finbert_analyze("Stock is doing well")
        assert result["method"] == "vader"

    def test_finbert_truncates_long_text(self, analyzer):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.5}]
        analyzer._finbert = mock_pipeline

        long_text = "word " * 1000
        analyzer._finbert_analyze(long_text)
        # Verify pipeline was called with truncated text
        called_text = mock_pipeline.call_args[0][0]
        assert len(called_text) <= 2000


# ---------------------------------------------------------------------------
# Public API tests
# ---------------------------------------------------------------------------

class TestPublicAPI:
    """Tests for the analyze() and analyze_batch() public methods."""

    def test_analyze_news_uses_finbert(self, analyzer):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.9}]
        analyzer._finbert = mock_pipeline

        result = analyzer.analyze("Record profits", source_type="news")
        assert result["method"] == "finbert"

    def test_analyze_social_uses_vader(self, analyzer):
        mock_vader = MagicMock()
        mock_vader.polarity_scores.return_value = {"compound": 0.6}
        analyzer._vader = mock_vader

        result = analyzer.analyze("This stock is awesome!", source_type="social")
        assert result["method"] == "vader"

    def test_analyze_fallback_uses_keyword(self, analyzer):
        result = analyzer.analyze("Stock gains rally", source_type="fallback")
        assert result["method"] == "keyword"

    def test_analyze_empty_text(self, analyzer):
        result = analyzer.analyze("")
        assert result["score"] == 0.0
        assert result["method"] == "none"

    def test_analyze_batch_empty(self, analyzer):
        result = analyzer.analyze_batch([], source_type="news")
        assert result["count"] == 0
        assert result["score"] == 0.0

    def test_analyze_batch_multiple(self, analyzer):
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = [
            [{"label": "positive", "score": 0.8}],
            [{"label": "negative", "score": 0.7}],
        ]
        analyzer._finbert = mock_pipeline

        result = analyzer.analyze_batch(
            ["Stock gains rally", "Stock crashes losses"], source_type="news"
        )
        assert result["count"] == 2
        assert -1.0 <= result["score"] <= 1.0

    def test_analyze_batch_skips_empty(self, analyzer):
        result = analyzer.analyze_batch(["", "  ", "Stock gains"], source_type="fallback")
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------

class TestSingleton:
    """Tests for get_sentiment_analyzer singleton."""

    def test_returns_same_instance(self):
        a1 = get_sentiment_analyzer()
        a2 = get_sentiment_analyzer()
        assert a1 is a2

    def test_returns_hybrid_analyzer(self):
        result = get_sentiment_analyzer()
        assert isinstance(result, HybridSentimentAnalyzer)
