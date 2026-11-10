"""AutFin v7.0 — Sentiment Toolset

Wraps sentiment analysis as a reusable FunctionToolset for Pydantic AI agents.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, RunContext

from autfin.tools.sentiment_analyzer import get_sentiment_analyzer
from autfin.tools.sentiment_tools import SentimentTools
from autfin.types import AgentDeps


def create_sentiment_toolset() -> FunctionToolset[AgentDeps]:
    """Create a FunctionToolset with all sentiment analysis tools."""
    toolset = FunctionToolset[AgentDeps]()
    
    # Sentiment tools instance
    def _get_sentiment_tools() -> SentimentTools:
        return SentimentTools()
    
    analyzer = get_sentiment_analyzer()
    
    @toolset.tool
    async def get_sentiment_data(ctx: RunContext[AgentDeps], symbol: str) -> dict[str, Any]:
        """Get comprehensive sentiment analysis for a symbol.

        Multi-source pipeline:
        - Marketaux (API, 0.4 weight) — structured news feed
        - Moneycontrol (browser, 0.2 weight) — Indian market news
        - Economic Times (browser, 0.2 weight) — Indian business news
        - Reddit (browser, 0.2 weight) — retail sentiment

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        tools = _get_sentiment_tools()
        try:
            result = await tools.get_multi_source_sentiment(symbol)
            sources = result.get("sources", [])
            # Extract catalysts/risks from news headlines
            catalysts = []
            risks = []
            for src in sources:
                if src.get("score", 0) > 0.2:
                    catalysts.append(f"{src['source']}: positive sentiment ({src['score']:.2f})")
                elif src.get("score", 0) < -0.2:
                    risks.append(f"{src['source']}: negative sentiment ({src['score']:.2f})")
            return {
                "symbol": symbol,
                "sentiment_score": result.get("weighted_score", 0.0),
                "sentiment_label": result.get("category", "neutral"),
                "news_sentiment": result.get("weighted_score", 0.0),
                "announcement_sentiment": 0.0,
                "catalysts": catalysts,
                "risks": risks,
                "summary": f"Multi-source sentiment: {result.get('category', 'neutral')} "
                f"(score={result.get('weighted_score', 0):.3f}, "
                f"confidence={result.get('confidence', 0):.3f})",
            }
        except Exception as e:
            return {
                "symbol": symbol,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "news_sentiment": 0.0,
                "announcement_sentiment": 0.0,
                "catalysts": [],
                "risks": [f"Sentiment pipeline error: {e}"],
                "summary": f"Sentiment analysis failed: {e}",
            }

    @toolset.tool
    async def analyze_text_sentiment(ctx: RunContext[AgentDeps], text: str) -> dict[str, Any]:
        """Analyze sentiment of arbitrary text using hybrid analyzer.
        
        Args:
            text: Text to analyze
        """
        result = analyzer.analyze(text)
        return {
            "sentiment_score": result.score,
            "sentiment_label": result.label,
            "confidence": result.confidence,
            "finbert_score": result.finbert_score,
            "vader_score": result.vader_score,
            "keywords": result.keywords,
        }

    @toolset.tool
    async def get_news_sentiment(
        ctx: RunContext[AgentDeps],
        symbol: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get news sentiment for a symbol from Marketaux.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            days: Lookback period in days
        """
        tools = _get_sentiment_tools()
        try:
            result = await tools.get_news_sentiment([symbol], days=days, limit=20)
            if result.get("status") == "success":
                agg = result.get("aggregate_sentiment", {})
                return {
                    "symbol": symbol,
                    "news_count": result.get("total_articles", 0),
                    "avg_sentiment": agg.get("weighted_score", 0.0),
                    "articles": [
                        {"title": a.get("title", ""), "sentiment": a.get("sentiment_score", 0)}
                        for a in result.get("articles", [])[:5]
                    ],
                }
            return {"symbol": symbol, "news_count": 0, "avg_sentiment": 0.0, "articles": []}
        except Exception:
            return {"symbol": symbol, "news_count": 0, "avg_sentiment": 0.0, "articles": []}

    @toolset.tool
    async def get_reddit_sentiment(
        ctx: RunContext[AgentDeps],
        symbol: str,
        subreddits: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get Reddit sentiment for a symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            subreddits: Optional list of subreddits to search
        """
        tools = _get_sentiment_tools()
        try:
            result = await tools.get_reddit_sentiment(symbol, subreddits=subreddits)
            return {
                "symbol": symbol,
                "posts_analyzed": result.get("posts_count", 0),
                "avg_sentiment": result.get("score", 0.0),
                "top_posts": result.get("top_posts", [])[:3],
            }
        except Exception:
            return {"symbol": symbol, "posts_analyzed": 0, "avg_sentiment": 0.0, "top_posts": []}

    return toolset


# Create the default toolset instance
sentiment_toolset = create_sentiment_toolset()


def get_sentiment_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared sentiment toolset instance."""
    return sentiment_toolset