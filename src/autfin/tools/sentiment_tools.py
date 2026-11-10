"""Sentiment analysis tools for AutFin.

Multi-source sentiment pipeline:
  - Marketaux (API, 0.4 weight) — structured news feed
  - Moneycontrol (browser, 0.2 weight) — Indian market news
  - Economic Times (browser, 0.2 weight) — Indian business news
  - Reddit (browser, 0.2 weight) — retail sentiment

Scraping delegated to browser_agent (Obscura MCP).
All sentiment scoring via HybridSentimentAnalyzer (FinBERT/VADER/keyword).
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from .browser_agent import get_browser_agent
from .sentiment_analyzer import get_sentiment_analyzer

logger = logging.getLogger(__name__)

# Default source weights for multi-source aggregation
DEFAULT_SOURCE_WEIGHTS = {
    "marketaux": 0.4,
    "moneycontrol": 0.2,
    "economic_times": 0.2,
    "reddit": 0.2,
}


class SentimentTools:
    """Tools for multi-source sentiment analysis."""
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize SentimentTools.
        
        Args:
            api_token: Marketaux API token. If None, loads from MARKETAUX_API_KEY env var.
        """
        self.api_token = api_token or os.environ.get("MARKETAUX_API_KEY", "")
        self.base_url = "https://api.marketaux.com/v1"
        self._client: Optional[httpx.AsyncClient] = None
        self._analyzer = get_sentiment_analyzer()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "AutFin-SentimentAgent/1.0"
                }
            )
        return self._client
    
    async def close(self):
        """Close httpx client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # ------------------------------------------------------------------
    # Browser scraping via Obscura MCP
    # ------------------------------------------------------------------

    async def _browser_scrape(self, task: str) -> str:
        """Delegate scraping to browser agent via Obscura MCP."""
        result = await get_browser_agent().run(task)
        return result.output

    @staticmethod
    def _parse_json_response(text: str) -> list:
        """Extract JSON array from browser agent response."""
        import json
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return []
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score using HybridSentimentAnalyzer (FinBERT/VADER/keyword).

        Returns score in range [-1, 1].
        """
        if not text:
            return 0.0
        result = self._analyzer.analyze(text, source_type="news")
        return result["score"]
    
    async def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using hybrid analyzer.
        
        Returns:
            Dictionary with positive, negative, neutral scores
        """
        if not text:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        result = self._analyzer.analyze(text, source_type="news")
        score = result["score"]
        
        # Convert to three-part score
        if score > 0:
            positive = min(1.0, abs(score))
            negative = 0.0
        else:
            positive = 0.0
            negative = min(1.0, abs(score))
        
        neutral = max(0.0, 1.0 - abs(score))
        
        return {
            "positive": round(positive, 4),
            "negative": round(negative, 4),
            "neutral": round(neutral, 4),
            "compound": round(score, 4),
            "method": result["method"],
        }
    
    async def score_news(self, headlines: list) -> Dict[str, Any]:
        """
        Score sentiment of news headlines.
        
        Args:
            headlines: List of news headlines
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not headlines:
            return {
                "scores": [],
                "average_sentiment": 0.0,
                "analysis_count": 0
            }
        
        scores = []
        for headline in headlines:
            score = self._calculate_sentiment_score(headline)
            scores.append({
                "headline": headline,
                "sentiment_score": round(score, 4),
                "sentiment_label": self._get_sentiment_label(score)
            })
        
        # Calculate average
        if scores:
            avg_sentiment = sum(s["sentiment_score"] for s in scores) / len(scores)
        else:
            avg_sentiment = 0.0
        
        return {
            "scores": scores,
            "average_sentiment": round(avg_sentiment, 4),
            "analysis_count": len(scores)
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label from score."""
        if score > 0.6:
            return "strong_positive"
        elif score > 0.3:
            return "positive"
        elif score > 0.1:
            return "slightly_positive"
        elif score > -0.1:
            return "neutral"
        elif score > -0.3:
            return "slightly_negative"
        elif score > -0.6:
            return "negative"
        else:
            return "strong_negative"
    
    async def get_news_sentiment(
        self,
        symbols: List[str],
        days: int = 30,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Fetch news and calculate sentiment for given symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ["RELIANCE", "TCS"])
            days: Number of days of history to fetch
            limit: Maximum number of articles to return
            
        Returns:
            Dictionary with news articles and sentiment analysis
        """
        try:
            client = await self._get_client()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Construct URL with symbols
            symbols_param = ",".join(symbols)
            url = f"{self.base_url}/news"
            params = {
                "api_token": self.api_token,
                "symbols": symbols_param,
                "published_at_min": start_date_str,
                "published_at_max": end_date_str,
                "limit": limit
            }
            
            # Make request
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"API returned status {response.status_code}",
                    "articles": [],
                    "sentiment": None
                }
            
            data = response.json()
            
            # Process articles
            articles = data.get("data", [])
            
            # Calculate sentiment for each article
            processed_articles = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("description", "")
                published_at = article.get("published_at", "")
                url = article.get("url", "")
                author = article.get("author", "")
                
                # Combine title and description for sentiment
                combined_text = f"{title} {description}"
                sentiment_score = self._calculate_sentiment_score(combined_text)
                
                processed_articles.append({
                    "title": title,
                    "description": description,
                    "url": url,
                    "author": author,
                    "published_at": published_at,
                    "sentiment_score": round(sentiment_score, 4),
                    "sentiment_label": self._get_sentiment_label(sentiment_score)
                })
            
            # Calculate aggregate sentiment
            aggregate = self._calculate_time_weighted_sentiment(
                processed_articles,
                days
            )
            
            return {
                "status": "success",
                "articles": processed_articles,
                "aggregate_sentiment": aggregate,
                "total_articles": len(processed_articles)
            }
            
        except httpx.RequestError as e:
            return {
                "status": "error",
                "message": f"Request failed: {str(e)}",
                "articles": [],
                "sentiment": None
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "articles": [],
                "sentiment": None
            }
    
    def _calculate_time_weighted_sentiment(
        self,
        articles: List[Dict],
        max_days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate time-weighted sentiment score.
        
        Recent news is weighted more heavily than older news.
        
        Args:
            articles: List of processed articles with sentiment scores
            max_days: Maximum age of articles to consider
            
        Returns:
            Dictionary with weighted sentiment analysis
        """
        if not articles:
            return {
                "weighted_score": 0.0,
                "unweighted_score": 0.0,
                "article_count": 0,
                "recent_articles": 0,
                "old_articles": 0,
                "confidence": 0.0
            }
        
        now = datetime.now()
        weighted_sum = 0.0
        total_weight = 0.0
        recent_count = 0
        old_count = 0
        
        for article in articles:
            published_at = article.get("published_at", "")
            sentiment_score = article.get("sentiment_score", 0.0)
            
            # Parse published date
            try:
                if published_at:
                    pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00").replace("+00:00", ""))
                    age_days = (now - pub_date).days
                else:
                    # If no date, assume old
                    age_days = max_days
            except (ValueError, TypeError):
                age_days = max_days
            
            # Calculate weight based on age
            if age_days <= 1:
                weight = 1.0
                recent_count += 1
            elif age_days <= 3:
                weight = 0.7
                recent_count += 1
            elif age_days <= 7:
                weight = 0.5
                recent_count += 1
            elif age_days <= 14:
                weight = 0.3
                old_count += 1
            elif age_days <= 30:
                weight = 0.1
                old_count += 1
            else:
                weight = 0.0
                old_count += 1
            
            # Add to weighted sum
            weighted_sum += sentiment_score * weight
            total_weight += weight
        
        # Calculate scores
        if total_weight > 0:
            weighted_score = weighted_sum / total_weight
        else:
            weighted_score = 0.0
        
        # Unweighted (simple average)
        if articles:
            unweighted_score = sum(a.get("sentiment_score", 0) for a in articles) / len(articles)
        else:
            unweighted_score = 0.0
        
        # Calculate confidence based on article count and weight distribution
        article_count = len(articles)
        weight_variance = 0.0
        if total_weight > 0:
            for article in articles:
                published_at = article.get("published_at", "")
                try:
                    if published_at:
                        pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00").replace("+00:00", ""))
                        age_days = (now - pub_date).days
                        if age_days <= 7:
                            weight_variance += 1.0
                except Exception:
                    logger.debug("Failed to parse article published_at date", exc_info=True)
        
        confidence = min(1.0, (article_count / 20) * (weight_variance / max(1, recent_count)))
        
        return {
            "weighted_score": round(weighted_score, 4),
            "unweighted_score": round(unweighted_score, 4),
            "article_count": article_count,
            "recent_articles": recent_count,
            "old_articles": old_count,
            "confidence": round(confidence, 4)
        }

    # ------------------------------------------------------------------
    # Multi-source sentiment
    # ------------------------------------------------------------------

    async def get_moneycontrol_sentiment(
        self, symbol: str, limit: int = 5
    ) -> Dict[str, Any]:
        """Scrape Moneycontrol news via browser agent and analyze sentiment.

        Returns:
            {source, symbol, score, confidence, articles_count, method}
        """
        try:
            text = await self._browser_scrape(
                f"Navigate to moneycontrol.com, search for '{symbol}' news. "
                f"Extract the {limit} most recent headline titles. "
                f"Return ONLY a JSON array of strings, no markdown."
            )
            headlines = self._parse_json_response(text)
            if not headlines:
                return {
                    "source": "moneycontrol",
                    "symbol": symbol,
                    "score": 0.0,
                    "confidence": 0.0,
                    "articles_count": 0,
                    "method": "none",
                }

            result = self._analyzer.analyze_batch(headlines, source_type="news")
            return {
                "source": "moneycontrol",
                "symbol": symbol,
                "score": result["score"],
                "confidence": result["confidence"],
                "articles_count": result["count"],
                "method": result["method"],
            }
        except Exception as e:
            logger.debug("Moneycontrol sentiment failed for %s: %s", symbol, e)
            return {
                "source": "moneycontrol",
                "symbol": symbol,
                "score": 0.0,
                "confidence": 0.0,
                "articles_count": 0,
                "method": "error",
                "error": str(e),
            }

    async def get_economic_times_sentiment(
        self, symbol: str, limit: int = 5
    ) -> Dict[str, Any]:
        """Scrape Economic Times news via browser agent and analyze sentiment.

        Returns:
            {source, symbol, score, confidence, articles_count, method}
        """
        try:
            text = await self._browser_scrape(
                f"Navigate to economictimes.indiatimes.com/markets. "
                f"Search for '{symbol}'. Extract the {limit} most recent headline titles. "
                f"Return ONLY a JSON array of strings, no markdown."
            )
            headlines = self._parse_json_response(text)
            if not headlines:
                return {
                    "source": "economic_times",
                    "symbol": symbol,
                    "score": 0.0,
                    "confidence": 0.0,
                    "articles_count": 0,
                    "method": "none",
                }

            result = self._analyzer.analyze_batch(headlines, source_type="news")
            return {
                "source": "economic_times",
                "symbol": symbol,
                "score": result["score"],
                "confidence": result["confidence"],
                "articles_count": result["count"],
                "method": result["method"],
            }
        except Exception as e:
            logger.debug("ET sentiment failed for %s: %s", symbol, e)
            return {
                "source": "economic_times",
                "symbol": symbol,
                "score": 0.0,
                "confidence": 0.0,
                "articles_count": 0,
                "method": "error",
                "error": str(e),
            }

    async def get_reddit_sentiment(
        self, symbol: str, subreddits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Scrape Reddit for retail sentiment via browser agent.

        Returns:
            {source, symbol, score, confidence, posts_count, method}
        """
        try:
            subs = subreddits or ["IndianStockMarket", "IndiaInvestments"]
            sub_filter = " OR ".join(f"r/{s}" for s in subs)
            text = await self._browser_scrape(
                f"Navigate to reddit.com and search {sub_filter} for '{symbol}' "
                f"sorted by new, last 7 days. Extract post titles, upvotes, and "
                f"comment counts. Return ONLY a JSON array of objects with keys: "
                f"title, upvotes, comments. No markdown."
            )
            posts = self._parse_json_response(text)

            if not posts:
                return {
                    "source": "reddit",
                    "symbol": symbol,
                    "score": 0.0,
                    "confidence": 0.0,
                    "posts_count": 0,
                    "method": "none",
                }

            # Score each post title
            titles = [p.get("title", "") for p in posts if p.get("title")]
            result = self._analyzer.analyze_batch(titles, source_type="social")

            return {
                "source": "reddit",
                "symbol": symbol,
                "score": result["score"],
                "confidence": min(1.0, result["count"] / 10),
                "posts_count": result["count"],
                "method": result["method"],
                "top_posts": posts[:5],
            }
        except Exception as e:
            logger.debug("Reddit sentiment failed for %s: %s", symbol, e)
            return {
                "source": "reddit",
                "symbol": symbol,
                "score": 0.0,
                "confidence": 0.0,
                "posts_count": 0,
                "method": "error",
                "error": str(e),
            }

    async def get_multi_source_sentiment(
        self,
        symbol: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Aggregate sentiment from all sources with weighted scoring.

        Sources:
          - Marketaux (API news) — weight 0.4
          - Moneycontrol (browser-scraped news) — weight 0.2
          - Economic Times (browser-scraped news) — weight 0.2
          - Reddit (browser-scraped retail) — weight 0.2

        Args:
            symbol: Stock symbol.
            weights: Optional override for source weights.

        Returns:
            {symbol, weighted_score, sources: [{source, score, weight,
             articles_count}], confidence, category}
        """
        if weights is None:
            weights = DEFAULT_SOURCE_WEIGHTS.copy()

        # Fetch all sources concurrently
        mc_task = self.get_moneycontrol_sentiment(symbol)
        et_task = self.get_economic_times_sentiment(symbol)
        reddit_task = self.get_reddit_sentiment(symbol)

        # Marketaux via existing path (may fail if no token)
        marketaux_result = None
        try:
            marketaux_result = await self.get_news_sentiment([symbol], days=7, limit=10)
            if marketaux_result.get("status") == "success":
                agg = marketaux_result.get("aggregate_sentiment", {})
                marketaux_source = {
                    "source": "marketaux",
                    "score": agg.get("weighted_score", 0.0),
                    "confidence": agg.get("confidence", 0.0),
                    "articles_count": marketaux_result.get("total_articles", 0),
                }
            else:
                marketaux_source = {
                    "source": "marketaux",
                    "score": 0.0,
                    "confidence": 0.0,
                    "articles_count": 0,
                }
        except Exception:
            marketaux_source = {
                "source": "marketaux",
                "score": 0.0,
                "confidence": 0.0,
                "articles_count": 0,
            }

        # Await concurrent tasks
        mc_result, et_result, reddit_result = await asyncio.gather(
            mc_task, et_task, reddit_task, return_exceptions=True
        )

        # Normalize results (handle exceptions from gather)
        sources = [marketaux_source]
        for result in [mc_result, et_result, reddit_result]:
            if isinstance(result, Exception):
                sources.append({
                    "source": "unknown",
                    "score": 0.0,
                    "confidence": 0.0,
                    "articles_count": 0,
                })
            else:
                sources.append(result)

        # Calculate weighted aggregate
        total_weight = 0.0
        weighted_sum = 0.0
        for src in sources:
            src_name = src.get("source", "unknown")
            w = weights.get(src_name, 0.0)
            score = src.get("score", 0.0)
            confidence = src.get("confidence", 0.0)
            # Only include sources that actually returned data
            effective_weight = w * (1.0 if confidence > 0 or src.get("articles_count", 0) > 0 else 0.0)
            weighted_sum += score * effective_weight
            total_weight += effective_weight

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine category
        category = self._get_sentiment_label(weighted_score)

        return {
            "symbol": symbol,
            "weighted_score": round(weighted_score, 4),
            "sources": sources,
            "weights_used": weights,
            "confidence": round(total_weight, 4),
            "category": category,
        }
