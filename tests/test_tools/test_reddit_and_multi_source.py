"""
Tests for Reddit sentiment tool and multi-source sentiment aggregation.
All tests use mocked external services — no network calls.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import src.autfin.tools.reddit_sentiment as reddit_mod
from src.autfin.tools.reddit_sentiment import RedditSentimentTool, get_reddit_sentiment_tool
from src.autfin.tools.sentiment_tools import SentimentTools, DEFAULT_SOURCE_WEIGHTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset module-level singletons before each test."""
    reddit_mod._reddit_tool = None
    reddit_mod._reddit_client = None
    reddit_mod._vader_analyzer = None
    yield
    reddit_mod._reddit_tool = None
    reddit_mod._reddit_client = None
    reddit_mod._vader_analyzer = None


@pytest.fixture
def reddit_tool():
    """Fresh RedditSentimentTool with mocked PRAW and VADER."""
    tool = RedditSentimentTool()
    mock_vader = MagicMock()
    mock_vader.polarity_scores.return_value = {
        "compound": 0.45, "pos": 0.6, "neg": 0.1, "neu": 0.3
    }
    tool._vader = mock_vader
    return tool


@pytest.fixture
def sentiment_tools():
    """Fresh SentimentTools with mocked sub-tools."""
    tools = SentimentTools(api_token="test-token")
    tools._analyzer = MagicMock()
    tools._analyzer.analyze.return_value = {"score": 0.3, "confidence": 0.7, "method": "keyword"}
    tools._analyzer.analyze_batch.return_value = {"score": 0.3, "confidence": 0.7, "count": 3, "method": "keyword"}
    return tools


# ---------------------------------------------------------------------------
# RedditSentimentTool tests
# ---------------------------------------------------------------------------

class TestRedditSentimentTool:
    """Tests for RedditSentimentTool."""

    def test_categorize(self, reddit_tool):
        assert reddit_tool._categorize(0.5) == "bullish"
        assert reddit_tool._categorize(0.15) == "slightly_bullish"
        assert reddit_tool._categorize(0.0) == "neutral"
        assert reddit_tool._categorize(-0.15) == "slightly_bearish"
        assert reddit_tool._categorize(-0.5) == "bearish"

    def test_volume_signal(self, reddit_tool):
        assert reddit_tool._volume_signal(15) == "high"
        assert reddit_tool._volume_signal(7) == "medium"
        assert reddit_tool._volume_signal(2) == "low"

    def test_analyze_no_reddit(self, reddit_tool):
        """Returns gracefully when Reddit not configured."""
        reddit_tool._reddit = None
        with patch.object(reddit_mod, "_reddit_client", None):
            result = reddit_tool.analyze("RELIANCE")
        assert result["posts_found"] == 0
        assert result["error"] == "Reddit not configured"

    def test_analyze_no_vader(self, reddit_tool):
        """Returns gracefully when VADER not installed."""
        mock_reddit = MagicMock()
        reddit_tool._reddit = mock_reddit
        reddit_tool._vader = None
        with patch.object(reddit_mod, "_reddit_client", mock_reddit):
            result = reddit_tool.analyze("RELIANCE")
        assert result["posts_found"] == 0
        assert result["error"] == "VADER not installed"

    def test_analyze_with_mock_reddit(self, reddit_tool):
        """Test with mocked PRAW client returning posts."""
        mock_reddit = MagicMock()
        mock_sub = MagicMock()

        # Create mock posts
        mock_post1 = MagicMock()
        mock_post1.title = "RELIANCE looking strong today"
        mock_post1.selftext = "Bullish on Reliance"
        mock_post1.score = 150
        mock_post1.num_comments = 45
        mock_post1.permalink = "/r/IndianStockMarket/comments/abc123/"

        mock_post2 = MagicMock()
        mock_post2.title = "RELIANCE Q3 results"
        mock_post2.selftext = "Good earnings"
        mock_post2.score = 80
        mock_post2.num_comments = 20
        mock_post2.permalink = "/r/IndiaInvestments/comments/def456/"

        mock_sub.search.return_value = [mock_post1, mock_post2]
        mock_reddit.subreddit.return_value = mock_sub
        reddit_tool._reddit = mock_reddit

        mock_vader = reddit_tool._vader
        with patch.object(reddit_mod, "_reddit_client", mock_reddit), \
             patch.object(reddit_mod, "_vader_analyzer", mock_vader):
            result = reddit_tool.analyze("RELIANCE", subreddits=["IndianStockMarket"])

        assert result["symbol"] == "RELIANCE"
        assert result["posts_found"] == 2
        assert isinstance(result["avg_sentiment"], float)
        assert result["sentiment_category"] in (
            "bullish", "slightly_bullish", "neutral",
            "slightly_bearish", "bearish"
        )
        assert len(result["top_posts"]) <= 5
        assert result["volume_signal"] in ("high", "medium", "low")

    def test_analyze_skips_deleted_posts(self, reddit_tool):
        """Deleted/removed posts should be excluded."""
        mock_reddit = MagicMock()
        mock_sub = MagicMock()

        mock_deleted = MagicMock()
        mock_deleted.title = "deleted post"
        mock_deleted.selftext = "[deleted]"

        mock_removed = MagicMock()
        mock_removed.title = "removed post"
        mock_removed.selftext = "[removed]"

        mock_real = MagicMock()
        mock_real.title = "Real post about RELIANCE"
        mock_real.selftext = "Great stock"
        mock_real.score = 50
        mock_real.num_comments = 10
        mock_real.permalink = "/r/IndianStockMarket/comments/xyz/"

        mock_sub.search.return_value = [mock_deleted, mock_removed, mock_real]
        mock_reddit.subreddit.return_value = mock_sub
        reddit_tool._reddit = mock_reddit

        mock_vader = reddit_tool._vader
        with patch.object(reddit_mod, "_reddit_client", mock_reddit), \
             patch.object(reddit_mod, "_vader_analyzer", mock_vader):
            result = reddit_tool.analyze("RELIANCE", subreddits=["IndianStockMarket"])
        assert result["posts_found"] == 1

    def test_analyze_handles_reddit_error(self, reddit_tool):
        """Handles exceptions from Reddit API gracefully."""
        mock_reddit = MagicMock()
        mock_sub = MagicMock()
        mock_sub.search.side_effect = Exception("rate limited")
        mock_reddit.subreddit.return_value = mock_sub
        reddit_tool._reddit = mock_reddit

        mock_vader = reddit_tool._vader
        with patch.object(reddit_mod, "_reddit_client", mock_reddit), \
             patch.object(reddit_mod, "_vader_analyzer", mock_vader):
            result = reddit_tool.analyze("RELIANCE", subreddits=["IndianStockMarket"])
        assert result["posts_found"] == 0
        assert result["sentiment_category"] == "neutral"

    def test_singleton(self):
        """get_reddit_sentiment_tool returns same instance."""
        t1 = get_reddit_sentiment_tool()
        t2 = get_reddit_sentiment_tool()
        assert t1 is t2


# ---------------------------------------------------------------------------
# Multi-source sentiment aggregation tests
# ---------------------------------------------------------------------------

class TestMultiSourceSentiment:
    """Tests for SentimentTools multi-source aggregation."""

    @pytest.mark.asyncio
    async def test_get_moneycontrol_sentiment_success(self, sentiment_tools):
        """Moneycontrol sentiment with mock scraper."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_moneycontrol_news = AsyncMock(return_value=[
            {"title": "RELIANCE shares rally on strong earnings"},
            {"title": "Reliance reports record profit"},
            {"title": "Reliance Jio growth accelerates"},
        ])
        sentiment_tools._scraper = mock_scraper

        # Mock analyze_batch to return count matching input
        sentiment_tools._analyzer.analyze_batch.return_value = {
            "score": 0.5, "confidence": 0.8, "count": 3, "method": "keyword"
        }

        result = await sentiment_tools.get_moneycontrol_sentiment("RELIANCE")

        assert result["source"] == "moneycontrol"
        assert result["symbol"] == "RELIANCE"
        assert result["articles_count"] == 3
        assert result["score"] != 0.0

    @pytest.mark.asyncio
    async def test_get_moneycontrol_sentiment_empty(self, sentiment_tools):
        """Moneycontrol returns no articles."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_moneycontrol_news = AsyncMock(return_value=[])
        sentiment_tools._scraper = mock_scraper

        result = await sentiment_tools.get_moneycontrol_sentiment("UNKNOWN")

        assert result["source"] == "moneycontrol"
        assert result["articles_count"] == 0
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_get_et_sentiment_success(self, sentiment_tools):
        """Economic Times sentiment with mock scraper."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_economic_times = AsyncMock(return_value=[
            {"title": "RELIANCE Q3 profit beats estimates"},
            {"title": "Reliance Industries share price target raised"},
        ])
        sentiment_tools._scraper = mock_scraper

        # Mock analyze_batch to return count matching input
        sentiment_tools._analyzer.analyze_batch.return_value = {
            "score": 0.4, "confidence": 0.7, "count": 2, "method": "keyword"
        }

        result = await sentiment_tools.get_economic_times_sentiment("RELIANCE")

        assert result["source"] == "economic_times"
        assert result["articles_count"] == 2

    @pytest.mark.asyncio
    async def test_get_et_sentiment_empty(self, sentiment_tools):
        """ET returns no articles."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_economic_times = AsyncMock(return_value=[])
        sentiment_tools._scraper = mock_scraper

        result = await sentiment_tools.get_economic_times_sentiment("FAKECO")

        assert result["source"] == "economic_times"
        assert result["articles_count"] == 0

    @pytest.mark.asyncio
    async def test_get_reddit_sentiment_success(self, sentiment_tools):
        """Reddit sentiment with mock tool."""
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = {
            "symbol": "RELIANCE",
            "posts_found": 8,
            "avg_sentiment": 0.35,
            "sentiment_category": "bullish",
            "top_posts": [{"title": "RELIANCE to the moon", "score_compound": 0.8}],
            "volume_signal": "medium",
        }
        sentiment_tools._reddit = mock_reddit

        result = await sentiment_tools.get_reddit_sentiment("RELIANCE")

        assert result["source"] == "reddit"
        assert result["score"] == 0.35
        assert result["posts_count"] == 8
        assert result["method"] == "vader"

    @pytest.mark.asyncio
    async def test_get_reddit_sentiment_no_posts(self, sentiment_tools):
        """Reddit returns no posts."""
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = {
            "symbol": "FAKECO",
            "posts_found": 0,
            "avg_sentiment": 0.0,
        }
        sentiment_tools._reddit = mock_reddit

        result = await sentiment_tools.get_reddit_sentiment("FAKECO")

        assert result["posts_count"] == 0
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_get_multi_source_sentiment(self, sentiment_tools):
        """Multi-source aggregation with all sources mocked."""
        # Mock Moneycontrol
        mock_scraper = MagicMock()
        mock_scraper.scrape_moneycontrol_news = AsyncMock(return_value=[
            {"title": "RELIANCE gains 2% on strong results"},
        ])
        mock_scraper.scrape_economic_times = AsyncMock(return_value=[
            {"title": "Reliance Q3 beats estimates"},
        ])
        sentiment_tools._scraper = mock_scraper

        # Mock Reddit
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = {
            "posts_found": 5,
            "avg_sentiment": 0.25,
            "sentiment_category": "slightly_bullish",
            "top_posts": [],
            "volume_signal": "medium",
        }
        sentiment_tools._reddit = mock_reddit

        # Mock Marketaux (get_news_sentiment)
        sentiment_tools.get_news_sentiment = AsyncMock(return_value={
            "status": "success",
            "total_articles": 3,
            "aggregate_sentiment": {
                "weighted_score": 0.4,
                "confidence": 0.6,
            },
        })

        result = await sentiment_tools.get_multi_source_sentiment("RELIANCE")

        assert result["symbol"] == "RELIANCE"
        assert isinstance(result["weighted_score"], float)
        assert len(result["sources"]) == 4
        assert "marketaux" in [s["source"] for s in result["sources"]]
        assert "moneycontrol" in [s["source"] for s in result["sources"]]
        assert "economic_times" in [s["source"] for s in result["sources"]]
        assert "reddit" in [s["source"] for s in result["sources"]]
        assert result["category"] in (
            "strong_positive", "positive", "slightly_positive",
            "neutral", "slightly_negative", "negative", "strong_negative"
        )

    @pytest.mark.asyncio
    async def test_get_multi_source_custom_weights(self, sentiment_tools):
        """Multi-source with custom weights."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_moneycontrol_news = AsyncMock(return_value=[])
        mock_scraper.scrape_economic_times = AsyncMock(return_value=[])
        sentiment_tools._scraper = mock_scraper

        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = {"posts_found": 0, "avg_sentiment": 0.0}
        sentiment_tools._reddit = mock_reddit

        sentiment_tools.get_news_sentiment = AsyncMock(return_value={"status": "error"})

        custom_weights = {"marketaux": 1.0, "moneycontrol": 0.0, "economic_times": 0.0, "reddit": 0.0}
        result = await sentiment_tools.get_multi_source_sentiment("RELIANCE", weights=custom_weights)

        assert result["weights_used"] == custom_weights

    @pytest.mark.asyncio
    async def test_get_multi_source_all_fail(self, sentiment_tools):
        """Multi-source when all sources fail — returns neutral."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_moneycontrol_news = AsyncMock(side_effect=Exception("network error"))
        mock_scraper.scrape_economic_times = AsyncMock(side_effect=Exception("timeout"))
        sentiment_tools._scraper = mock_scraper

        mock_reddit = MagicMock()
        mock_reddit.analyze.side_effect = Exception("no reddit")
        sentiment_tools._reddit = mock_reddit

        sentiment_tools.get_news_sentiment = AsyncMock(side_effect=Exception("api down"))

        result = await sentiment_tools.get_multi_source_sentiment("RELIANCE")

        assert result["weighted_score"] == 0.0
        assert result["category"] == "neutral"
