"""Hybrid sentiment analyzer: FinBERT (news) + VADER (social) + keyword fallback.

All models run locally — zero API cost. FinBERT is lazy-loaded on first use (~400MB download).
VADER is instant (pure Python, no model download).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level caches for loaded models
_finbert_pipeline = None
_vader_analyzer = None
_finbert_load_attempted = False
_vader_load_attempted = False


class HybridSentimentAnalyzer:
    """ML-based sentiment analysis with graceful fallback.

    Routes text to the appropriate analyzer:
      - News articles → FinBERT (ProsusAI/finbert, 0.97 accuracy on financial text)
      - Social media  → VADER (fast, good for informal text, emojis, slang)
      - Fallback      → Keyword matching (existing implementation)

    All models run locally. FinBERT is lazy-loaded on first call.
    """

    def __init__(self):
        self._finbert = None
        self._vader = None

    # ------------------------------------------------------------------
    # Model loaders (lazy, cached at module level)
    # ------------------------------------------------------------------

    def _load_finbert(self):
        """Load FinBERT model via HuggingFace transformers."""
        global _finbert_pipeline, _finbert_load_attempted

        if _finbert_load_attempted:
            self._finbert = _finbert_pipeline
            return

        _finbert_load_attempted = True
        try:
            from transformers import pipeline as hf_pipeline
            _finbert_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=False,
                top_k=None,
            )
            self._finbert = _finbert_pipeline
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning("FinBERT unavailable, will use VADER/keyword fallback: %s", e)
            _finbert_pipeline = None
            self._finbert = None

    def _load_vader(self):
        """Load VADER sentiment analyzer (instant, no model download)."""
        global _vader_analyzer, _vader_load_attempted

        if _vader_load_attempted:
            self._vader = _vader_analyzer
            return

        _vader_load_attempted = True
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
            self._vader = _vader_analyzer
            logger.info("VADER sentiment analyzer loaded")
        except ImportError:
            logger.warning("VADER not installed, will use keyword fallback")
            _vader_analyzer = None
            self._vader = None

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def _finbert_analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text using FinBERT (financial ML model).

        FinBERT outputs: {label: "positive"|"negative"|"neutral", score: float}
        We map to a [-1, 1] score: positive→+score, negative→-score, neutral→0.
        """
        if not self._finbert:
            self._load_finbert()
        if not self._finbert:
            return self._vader_analyze(text)  # fallback

        try:
            # FinBERT max input is 512 tokens
            truncated = text[:2000] if len(text) > 2000 else text
            results = self._finbert(truncated)

            # Results format: [[{label, score}, ...]] or [{label, score}, ...]
            if results and isinstance(results[0], list):
                results = results[0]

            # Find the top prediction
            best = max(results, key=lambda x: x["score"])
            label = best["label"].lower()
            confidence = best["score"]

            label_to_score = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            raw_score = label_to_score.get(label, 0.0)

            return {
                "score": round(raw_score * confidence, 4),
                "confidence": round(confidence, 4),
                "method": "finbert",
                "label": label,
            }
        except Exception as e:
            logger.debug("FinBERT inference failed, falling back to VADER: %s", e)
            return self._vader_analyze(text)

    def _vader_analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text using VADER (lexicon + rule-based, fast)."""
        if not self._vader:
            self._load_vader()
        if not self._vader:
            return self._keyword_analyze(text)

        try:
            scores = self._vader.polarity_scores(text)
            compound = scores["compound"]
            return {
                "score": round(compound, 4),
                "confidence": round(abs(compound), 4),
                "method": "vader",
                "label": "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral",
            }
        except Exception as e:
            logger.debug("VADER analysis failed, falling back to keyword: %s", e)
            return self._keyword_analyze(text)

    def _keyword_analyze(self, text: str) -> Dict[str, Any]:
        """Fallback: keyword-based sentiment (existing implementation)."""
        if not text:
            return {"score": 0.0, "confidence": 0.0, "method": "keyword", "label": "neutral"}

        text_lower = text.lower()

        positive_words = [
            "gain", "rise", "rising", "up", " rally", "strong", "buoyant",
            "positive", "bullish", "growth", "profit", "beats", "upgrade",
            "outperform", "best", "excellent", "better", "advantage",
            "favourable", "opportunity",
        ]
        negative_words = [
            "drop", "fall", "falling", "down", "plunge", "weak", "bearish",
            "negative", "sell", "downgrade", "underperform", "worst", "worse",
            "risk", "concern", "loss", "miss", "decline", "downturn",
            "recession", "crisis", "trouble",
        ]

        pos_count = sum(text_lower.count(w) for w in positive_words)
        neg_count = sum(text_lower.count(w) for w in negative_words)
        total = pos_count + neg_count

        if total == 0:
            return {"score": 0.0, "confidence": 0.0, "method": "keyword", "label": "neutral"}

        raw_score = (pos_count - neg_count) / total
        severity = min(1.0, total / 5)
        scaled_score = raw_score * (0.3 + 0.7 * severity)
        score = max(-1.0, min(1.0, scaled_score))

        label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        return {
            "score": round(score, 4),
            "confidence": round(severity, 4),
            "method": "keyword",
            "label": label,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str, source_type: str = "news") -> Dict[str, Any]:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze.
            source_type: "news" → FinBERT, "social" → VADER, "fallback" → keyword.

        Returns:
            {score: float [-1,1], confidence: float [0,1], method: str, label: str}
        """
        if not text or not text.strip():
            return {"score": 0.0, "confidence": 0.0, "method": "none", "label": "neutral"}

        if source_type == "news":
            return self._finbert_analyze(text)
        elif source_type == "social":
            return self._vader_analyze(text)
        else:
            return self._keyword_analyze(text)

    def analyze_batch(self, texts: list, source_type: str = "news") -> Dict[str, Any]:
        """Analyze multiple texts and return aggregate sentiment.

        Args:
            texts: List of text strings.
            source_type: "news" | "social" | "fallback".

        Returns:
            {score: float, confidence: float, count: int, method: str}
        """
        if not texts:
            return {"score": 0.0, "confidence": 0.0, "count": 0, "method": "none"}

        results = [self.analyze(t, source_type) for t in texts if t and t.strip()]
        if not results:
            return {"score": 0.0, "confidence": 0.0, "count": 0, "method": "none"}

        avg_score = sum(r["score"] for r in results) / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        method = results[0]["method"]

        return {
            "score": round(avg_score, 4),
            "confidence": round(avg_confidence, 4),
            "count": len(results),
            "method": method,
        }


# Module-level singleton (lazy initialized)
_analyzer: Optional[HybridSentimentAnalyzer] = None


def get_sentiment_analyzer() -> HybridSentimentAnalyzer:
    """Get or create the global HybridSentimentAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = HybridSentimentAnalyzer()
    return _analyzer
