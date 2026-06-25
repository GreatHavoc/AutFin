"""Sentiment analysis tools for AutFin."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class SentimentTools:
    """Tools for sentiment analysis."""
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize SentimentTools.
        
        Args:
            api_token: Marketaux API token. If None, loads from environment.
        """
        self.api_token = api_token
        self.base_url = "https://api.marketaux.com/v1"
        self._client: Optional[httpx.AsyncClient] = None
    
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
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment score from text.
        
        Uses a simple keyword-based approach.
        Returns score in range [-1, 1].
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 negative, 0 neutral, 1 positive)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = [
            "gain", "rise", "rising", "up", " rally", "strong", "buoyant",
            "positive", "bullish", "growth", "profit", "beats", "upgrade",
            "outperform", "best", "excellent", "better", "advantage",
            "favourable", "opportunity", "潜力", "潜力", "潜力", "潜力"
        ]
        
        # Negative keywords
        negative_words = [
            "drop", "fall", "falling", "down", "plunge", "weak", "bearish",
            "negative", "sell", "downgrade", "underperform", "worst", "worse",
            "risk", "concern", "loss", "miss", "decline", "downturn",
            "recession", "crisis", " trouble", " trouble", " trouble", " trouble"
        ]
        
        # Count occurrences
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        # Calculate raw score
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Normalize to [-1, 1]
        raw_score = (positive_count - negative_count) / total
        
        # Apply severity scaling
        severity = min(1.0, (positive_count + negative_count) / 5)
        scaled_score = raw_score * (0.3 + 0.7 * severity)
        
        return max(-1.0, min(1.0, scaled_score))
    
    async def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with positive, negative, neutral scores
        """
        if not text:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        score = self._calculate_sentiment_score(text)
        
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
            "compound": round(score, 4)
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
