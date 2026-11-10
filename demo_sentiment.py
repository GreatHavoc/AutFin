"""Live sentiment demo — no mocks."""
import asyncio
from autfin.tools.sentiment_tools import SentimentTools
from autfin.tools.sentiment_analyzer import get_sentiment_analyzer


async def main():
    st = SentimentTools()
    analyzer = get_sentiment_analyzer()

    print("=== REDDIT (public JSON API) ===")
    reddit = await st.get_reddit_sentiment("RELIANCE")
    print(f"  posts: {reddit['posts_count']}, score: {reddit['score']}")
    for p in reddit.get("top_posts", [])[:3]:
        print(f"    [{p['score_compound']:+.3f}] {p['title'][:70]}")

    print("\n=== MARKETAUX ===")
    news = await st.get_news_sentiment(["RELIANCE"], days=7, limit=5)
    print(f"  status: {news['status']}, articles: {news.get('total_articles', 0)}")

    print("\n=== MULTI-SOURCE ===")
    combined = await st.get_multi_source_sentiment("RELIANCE")
    print(f"  weighted: {combined['weighted_score']}, category: {combined['category']}")
    for s in combined.get("sources", []):
        print(f"    {s['source']:>15}: {s['score']:+.3f}")


asyncio.run(main())
