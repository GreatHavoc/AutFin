"""Reddit sentiment analysis for Indian stocks via PRAW.

Searches r/IndianStockMarket (331K members) and r/IndiaInvestments for stock mentions.
Uses VADER for sentiment scoring — fast, good for informal text/emoji/slang.
All free — PRAW read-only requires no API key for basic usage.
"""

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded PRAW and VADER
_reddit_client = None
_vader_analyzer = None


class RedditSentimentTool:
    """Fetch and analyze Reddit sentiment for Indian stocks."""

    def __init__(self):
        self._reddit = None
        self._vader = None

    def _load_reddit(self):
        """Lazy-init PRAW Reddit client."""
        global _reddit_client
        if _reddit_client is not None:
            self._reddit = _reddit_client
            return

        try:
            import praw
            client_id = os.environ.get("REDDIT_CLIENT_ID", "")
            client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
            user_agent = os.environ.get("REDDIT_USER_AGENT", "AutFin/1.0")

            if not client_id or not client_secret:
                logger.warning(
                    "Reddit credentials not configured (REDDIT_CLIENT_ID, "
                    "REDDIT_CLIENT_SECRET). Reddit sentiment unavailable."
                )
                self._reddit = None
                return

            _reddit_client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            self._reddit = _reddit_client
            logger.info("PRAW Reddit client initialized")
        except ImportError:
            logger.warning("praw not installed — Reddit sentiment unavailable")
            _reddit_client = None
            self._reddit = None

    def _load_vader(self):
        """Lazy-load VADER sentiment analyzer."""
        global _vader_analyzer
        if _vader_analyzer is not None:
            self._vader = _vader_analyzer
            return

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
            self._vader = _vader_analyzer
        except ImportError:
            logger.warning("vaderSentiment not installed")
            _vader_analyzer = None
            self._vader = None

    def _categorize(self, score: float) -> str:
        """Map sentiment score to category."""
        if score > 0.3:
            return "bullish"
        elif score > 0.05:
            return "slightly_bullish"
        elif score > -0.05:
            return "neutral"
        elif score > -0.3:
            return "slightly_bearish"
        else:
            return "bearish"

    def _volume_signal(self, post_count: int) -> str:
        """Map post count to volume signal."""
        if post_count >= 10:
            return "high"
        elif post_count >= 5:
            return "medium"
        else:
            return "low"

    def analyze(
        self,
        symbol: str,
        subreddits: Optional[List[str]] = None,
        limit_per_sub: int = 20,
    ) -> Dict[str, Any]:
        """Search Reddit for stock mentions and analyze sentiment.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE").
            subreddits: Subreddits to search. Defaults to Indian finance subs.
            limit_per_sub: Max posts per subreddit.

        Returns:
            {symbol, posts_found, avg_sentiment, sentiment_category,
             top_posts, volume_signal}
        """
        if subreddits is None:
            subreddits = [
                "IndianStockMarket",
                "IndiaInvestments",
                "IndianStreetBets",
            ]

        self._load_reddit()
        self._load_vader()

        if not self._reddit:
            return {
                "symbol": symbol,
                "posts_found": 0,
                "avg_sentiment": 0.0,
                "sentiment_category": "neutral",
                "top_posts": [],
                "volume_signal": "none",
                "error": "Reddit not configured",
            }

        if not self._vader:
            return {
                "symbol": symbol,
                "posts_found": 0,
                "avg_sentiment": 0.0,
                "sentiment_category": "neutral",
                "top_posts": [],
                "volume_signal": "none",
                "error": "VADER not installed",
            }

        posts = []
        symbol_upper = symbol.upper()

        for sub_name in subreddits:
            try:
                subreddit = self._reddit.subreddit(sub_name)
                for post in subreddit.search(
                    symbol_upper, limit=limit_per_sub, sort="new", time_filter="week"
                ):
                    # Skip deleted/removed posts
                    if post.selftext in ("[deleted]", "[removed]"):
                        continue

                    text = f"{post.title} {post.selftext[:500]}"
                    scores = self._vader.polarity_scores(text)

                    posts.append({
                        "title": post.title[:120],
                        "score_compound": round(scores["compound"], 4),
                        "score_pos": round(scores["pos"], 4),
                        "score_neg": round(scores["neg"], 4),
                        "upvotes": post.score,
                        "comments": post.num_comments,
                        "url": f"https://reddit.com{post.permalink}",
                        "subreddit": sub_name,
                    })
            except Exception as e:
                logger.debug("Reddit search failed for r/%s: %s", sub_name, e)
                continue

        if not posts:
            return {
                "symbol": symbol,
                "posts_found": 0,
                "avg_sentiment": 0.0,
                "sentiment_category": "neutral",
                "top_posts": [],
                "volume_signal": "none",
            }

        # Calculate average sentiment
        avg_sentiment = sum(p["score_compound"] for p in posts) / len(posts)

        # Sort by absolute sentiment (most opinionated first)
        top_posts = sorted(posts, key=lambda x: abs(x["score_compound"]), reverse=True)[:5]

        return {
            "symbol": symbol,
            "posts_found": len(posts),
            "avg_sentiment": round(avg_sentiment, 4),
            "sentiment_category": self._categorize(avg_sentiment),
            "top_posts": top_posts,
            "volume_signal": self._volume_signal(len(posts)),
        }


# Module-level singleton
_reddit_tool: Optional[RedditSentimentTool] = None


def get_reddit_sentiment_tool() -> RedditSentimentTool:
    """Get or create the global RedditSentimentTool instance."""
    global _reddit_tool
    if _reddit_tool is None:
        _reddit_tool = RedditSentimentTool()
    return _reddit_tool
