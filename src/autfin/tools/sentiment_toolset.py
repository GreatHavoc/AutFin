"""AutFin v7.0 — Sentiment Toolset

Wraps sentiment analysis as a reusable FunctionToolset for Pydantic AI agents.
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, RunContext

from autfin.tools.sentiment_tools import SentimentTools
from autfin.tools.sentiment_analyzer import get_sentiment_analyzer
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
        - Moneycontrol (scraper, 0.2 weight) — Indian market news
        - Economic Times (scraper, 0.2 weight) — Indian business news
        - Reddit (PRAW, 0.2 weight) — retail sentiment
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        tools = _get_sentiment_tools()
        # This would call the full sentiment pipeline
        # For now, return structure
        return {
            "symbol": symbol,
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "news_sentiment": 0.0,
            "announcement_sentiment": 0.0,
            "catalysts": [],
            "risks": [],
            "summary": "Sentiment analysis placeholder",
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
        # Placeholder - would call Marketaux API
        return {
            "symbol": symbol,
            "news_count": 0,
            "avg_sentiment": 0.0,
            "articles": [],
        }

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
        # Placeholder - would call Reddit API
        return {
            "symbol": symbol,
            "posts_analyzed": 0,
            "avg_sentiment": 0.0,
            "top_posts": [],
        }

    return toolset


# Create the default toolset instance
sentiment_toolset = create_sentiment_toolset()


def get_sentiment_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared sentiment toolset instance."""
    return sentiment_toolset