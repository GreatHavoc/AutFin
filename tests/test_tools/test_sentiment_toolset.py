"""Tests for sentiment_tools.py — browser agent mocked, Marketaux live."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from autfin.tools.sentiment_tools import SentimentTools
from autfin.tools.sentiment_analyzer import get_sentiment_analyzer


# ─── Browser agent mock helper ───

def _mock_browser_run(headlines: list):
    """Return a patch that makes browser_agent.run return headlines as JSON."""
    mock_result = AsyncMock()
    mock_result.output = json.dumps(headlines)
    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(return_value=mock_result)
    return patch("autfin.tools.sentiment_tools.get_browser_agent", return_value=mock_agent)


# ─── Tests ───

@pytest.mark.asyncio
async def test_get_news_sentiment():
    st = SentimentTools()
    result = await st.get_news_sentiment(["RELIANCE"], days=7, limit=5)

    # Works with or without API key
    assert result["status"] in ("success", "error")
    assert isinstance(result.get("total_articles", 0), int)


@pytest.mark.asyncio
async def test_get_moneycontrol_sentiment():
    fake_headlines = [
        "Reliance reports strong Q3 earnings",
        "Jio tariff hike boosts revenue outlook",
    ]
    with _mock_browser_run(fake_headlines):
        st = SentimentTools()
        result = await st.get_moneycontrol_sentiment("RELIANCE")

    assert result["source"] == "moneycontrol"
    assert result["articles_count"] == 2
    assert isinstance(result["score"], float)


@pytest.mark.asyncio
async def test_get_economic_times_sentiment():
    fake_headlines = ["Reliance Industries stock surges 5% on earnings beat"]
    with _mock_browser_run(fake_headlines):
        st = SentimentTools()
        result = await st.get_economic_times_sentiment("RELIANCE")

    assert result["source"] == "economic_times"
    assert result["articles_count"] == 1


@pytest.mark.asyncio
async def test_get_reddit_sentiment():
    fake_posts = [
        {"title": "RELIANCE looking bullish after results", "upvotes": 120, "comments": 45},
        {"title": "Should I buy more RELIANCE?", "upvotes": 30, "comments": 12},
    ]
    with _mock_browser_run(fake_posts):
        st = SentimentTools()
        result = await st.get_reddit_sentiment("RELIANCE")

    assert result["source"] == "reddit"
    assert result["posts_count"] == 2
    assert isinstance(result["score"], float)


@pytest.mark.asyncio
async def test_browser_scrape_empty_response():
    with _mock_browser_run([]):
        st = SentimentTools()
        result = await st.get_moneycontrol_sentiment("TCS")

    assert result["articles_count"] == 0
    assert result["score"] == 0.0


@pytest.mark.asyncio
async def test_analyze_text_sentiment():
    analyzer = get_sentiment_analyzer()
    result = analyzer.analyze(
        "Reliance Industries reported strong quarterly earnings with 15% profit growth"
    )

    assert "score" in result
    assert "label" in result
    assert -1.0 <= result["score"] <= 1.0


@pytest.mark.asyncio
async def test_analyze_negative_text():
    analyzer = get_sentiment_analyzer()
    result = analyzer.analyze(
        "Stock crashed after regulatory investigation and accounting scandal"
    )

    assert result["score"] < 0


def test_toolset_creation():
    from autfin.tools.sentiment_toolset import create_sentiment_toolset
    assert create_sentiment_toolset() is not None


def test_parse_json_response():
    # Valid JSON array
    assert SentimentTools._parse_json_response('["a", "b"]') == ["a", "b"]
    # JSON embedded in text
    assert SentimentTools._parse_json_response('Here is data: ["x"] done') == ["x"]
    # No JSON
    assert SentimentTools._parse_json_response("no json here") == []
