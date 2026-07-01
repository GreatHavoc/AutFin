"""Sentiment analysis agent for AutFin — agentic with tool-calling loop."""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.tools.sentiment_tools import SentimentTools
from src.autfin.llm.client import get_llm_client
from src.autfin.agents.tools import SENTIMENT_TOOLS, SENTIMENT_CALLABLES

logger = logging.getLogger(__name__)


class SentimentAgent:
    """
    Analyzes market sentiment from news and social media.
    
    Integrates with Marketaux API to fetch and analyze news articles.
    Implements time-weighted sentiment scoring for more accurate signals.
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize SentimentAgent.
        
        Args:
            api_token: Marketaux API token. If None, loads from environment.
        """
        self.api_token = api_token or os.getenv("MARKETAUX_API_KEY")
        self._tools: Optional[SentimentTools] = None
    
    async def _get_tools(self) -> SentimentTools:
        """Get or create SentimentTools instance."""
        if self._tools is None:
            self._tools = SentimentTools(api_token=self.api_token)
        return self._tools
    
    async def close(self):
        """Clean up resources."""
        if self._tools:
            await self._tools.close()
            self._tools = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def _get_sentiment_category(self, score: float) -> str:
        """
        Get sentiment category from score.
        
        Categories:
        - Strong Buy: > 0.6
        - Buy: 0.3 to 0.6
        - Neutral: -0.3 to 0.3
        - Sell: -0.6 to -0.3
        - Strong Sell: < -0.6
        """
        if score > 0.6:
            return "strong_buy"
        elif score > 0.3:
            return "buy"
        elif score > -0.3:
            return "neutral"
        elif score > -0.6:
            return "sell"
        else:
            return "strong_sell"
    
    async def get_news_sentiment(
        self,
        symbol: str,
        days: int = 30,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Fetch news and calculate sentiment for a stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            days: Number of days of history to fetch
            limit: Maximum number of articles
            
        Returns:
            Dictionary with news sentiment analysis
        """
        try:
            tools = await self._get_tools()
            
            # Fetch news with sentiment
            result = await tools.get_news_sentiment(
                symbols=[symbol],
                days=days,
                limit=limit
            )
            
            if result.get("status") != "success":
                return {
                    "status": "error",
                    "symbol": symbol,
                    "error": result.get("message", "Unknown error"),
                    "sentiment": None
                }
            
            # Get aggregate sentiment
            aggregate = result.get("aggregate_sentiment", {})
            weighted_score = aggregate.get("weighted_score", 0.0)
            
            # Determine sentiment category
            sentiment_category = self._get_sentiment_category(weighted_score)
            
            return {
                "status": "success",
                "symbol": symbol,
                "days_analyzed": days,
                "articles_count": result.get("total_articles", 0),
                "sentiment": {
                    "score": weighted_score,
                    "category": sentiment_category,
                    "confidence": aggregate.get("confidence", 0.0),
                    "recent_articles": aggregate.get("recent_articles", 0),
                    "articles": result.get("articles", [])[:10]  # Top 10 articles
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "symbol": symbol,
                "error": str(e),
                "sentiment": None
            }

    def aggregate_sentiment(self, sentiments: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate sentiment across multiple symbols or sources.
        
        Args:
            sentiments: List of sentiment dictionaries
            
        Returns:
            Aggregated sentiment with weighted average and confidence
        """
        if not sentiments:
            return {
                "weighted_score": 0.0,
                "unweighted_score": 0.0,
                "symbol_count": 0,
                "min_score": 0.0,
                "max_score": 0.0,
                "sentiment_categories": {},
                "confidence": 0.0,
                "category": "neutral"
            }
        
        # Extract scores
        scores = []
        categories = {}
        
        for sentiment in sentiments:
            score = sentiment.get("sentiment", {}).get("score", 0.0)
            scores.append(score)
            
            # Count categories
            category = sentiment.get("sentiment", {}).get("category", "neutral")
            categories[category] = categories.get(category, 0) + 1
        
        # Calculate statistics
        if scores:
            weighted_score = sum(scores) / len(scores)
            unweighted_score = weighted_score
            min_score = min(scores)
            max_score = max(scores)
        else:
            weighted_score = 0.0
            unweighted_score = 0.0
            min_score = 0.0
            max_score = 0.0
        
        # Determine overall category
        overall_category = self._get_sentiment_category(weighted_score)
        
        # Calculate confidence based on variety and count
        symbol_count = len(sentiments)
        category_variety = len(categories)
        confidence = min(1.0, (symbol_count / 10) * (0.5 + 0.5 * category_variety / 5))
        
        return {
            "weighted_score": round(weighted_score, 4),
            "unweighted_score": round(unweighted_score, 4),
            "symbol_count": symbol_count,
            "min_score": round(min_score, 4),
            "max_score": round(max_score, 4),
            "sentiment_categories": categories,
            "confidence": round(confidence, 4),
            "category": overall_category
        }
    
    async def analyze_news(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyze news sentiment for stocks in state.
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Get symbols from holdings or default to major indices
        symbols = []
        
        # Try to get symbols from holdings
        holdings = state.get("holdings", [])
        for holding in holdings:
            if "tradingsymbol" in holding:
                symbols.append(holding["tradingsymbol"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        # If no symbols, default to major indices
        if not unique_symbols:
            unique_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
        
        # Fetch sentiment for each symbol (Marketaux primary)
        results = []
        for symbol in unique_symbols[:10]:  # Limit to 10 symbols
            result = await self.get_news_sentiment(symbol, days=30, limit=20)
            results.append(result)
        
        return {
            "sentiment_reports": {
                "timestamp": datetime.now().isoformat(),
                "symbols_analyzed": len(results),
                "results": results
            }
        }
    
    async def analyze_social(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyze social media sentiment (placeholder for future implementation).
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with social sentiment analysis
        """
        # Social media sentiment analysis would go here
        # For now, return placeholder
        return {
            "social_sentiment": {
                "timestamp": datetime.now().isoformat(),
                "platforms_analyzed": [],
                "sentiment_score": 0.0,
                "mentions_count": 0
            }
        }
    
    async def _agentic_analyze(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Tool-calling agentic analysis: LLM decides which sources to query."""
        llm = get_llm_client()
        if not llm.is_available:
            return None

        system_prompt = (
            f"You are a market sentiment analyst. Analyze sentiment for {symbol} on NSE.\n"
            "Use your tools to fetch news sentiment (Marketaux).\n"
            "Produce a COMPLETE JSON with: sentiment_score (-1.0 to +1.0),\n"
            "sentiment_category (Strong Buy/Buy/Neutral/Sell/Strong Sell),\n"
            "articles_count, positive_hits, negative_hits, key_themes (list),\n"
            "summary (2-3 sentence narrative), conviction (0.0-1.0).\n"
            "Output ONLY the JSON — no markdown, no explanation."
        )

        raw = await llm.generate_with_tools(
            system_prompt=system_prompt,
            tools=SENTIMENT_TOOLS,
            tool_callables=SENTIMENT_CALLABLES,
            model=llm.deep_model,
            max_tokens=1024,
            max_iterations=20,
        )
        if not raw:
            return None

        from src.autfin.utils import extract_json
        result = extract_json(raw)
        if result:
            return result
        logger.warning(f"Sentiment agentic output not JSON for {symbol}")
        return None

    async def sentiment_agent(self, state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """
        Orchestrate sentiment analysis workflow.
        
        Runs news and social sentiment analysis, updates state,
        and handles errors gracefully.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with sentiment signals
        """
        errors = state.get("errors", [])
        
        try:
            # Try agentic path first for each symbol
            symbols = state.get("holdings", [])
            if not symbols:
                symbols = [{"tradingsymbol": "RELIANCE"}]

            agentic_reports = {}
            for holding in symbols[:5]:
                symbol = holding.get("tradingsymbol", "")
                if not symbol:
                    continue
                report = await self._agentic_analyze(symbol)
                if report:
                    agentic_reports[symbol] = report

            if agentic_reports:
                scores = [r.get("sentiment_score", 0) for r in agentic_reports.values()]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                sentiment_reports = {
                    "timestamp": datetime.now().isoformat(),
                    "aggregate": {
                        "weighted_score": round(avg_score, 4),
                        "score": round(avg_score, 4),
                        "category": self._get_sentiment_category(avg_score),
                    },
                    "results": [{"symbol": k, **v} for k, v in agentic_reports.items()],
                }
                return {"sentiment_reports": sentiment_reports, "errors": errors}

            # Fall back to deterministic path
            news_result = await self.analyze_news(state)
            
            # Step 2: Analyze social sentiment (placeholder)
            social_result = await self.analyze_social(state)
            
            # Step 3: Aggregate sentiment
            sentiments = [
                r for r in news_result.get("sentiment_reports", {}).get("results", [])
                if r.get("status") == "success"
            ]
            aggregate = self.aggregate_sentiment(sentiments)
            
            # Build sentiment reports
            sentiment_reports = {
                "timestamp": datetime.now().isoformat(),
                "aggregate": aggregate,
                **news_result.get("sentiment_reports", {})
            }
            
            # Update state
            updated_state = {
                "sentiment_reports": sentiment_reports,
                "agent": "sentiment",
                "next": None,
                "errors": errors,
            }
            
            return updated_state
            
        except Exception as e:
            errors.append(f"Sentiment analysis failed: {str(e)}")
            
            return {
                "sentiment_reports": {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                },
                "agent": "sentiment",
                "next": None,
                "errors": errors,
            }


# Global sentiment agent instance for convenience
_sentiment_agent: Optional[SentimentAgent] = None


def create_sentiment_agent(api_token: Optional[str] = None) -> SentimentAgent:
    """
    Create or get global sentiment agent instance.
    
    Args:
        api_token: Optional API token override
        
    Returns:
        SentimentAgent instance
    """
    global _sentiment_agent
    
    if _sentiment_agent is None:
        _sentiment_agent = SentimentAgent(api_token=api_token)
    
    return _sentiment_agent


async def sentiment_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Convenience function to run sentiment agent.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with sentiment signals
    """
    agent = create_sentiment_agent()
    return await agent.sentiment_agent(state, config)
