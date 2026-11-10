import httpx
import asyncio

async def test():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Try fetching subreddit listing instead of search
    urls = [
        "https://www.reddit.com/r/IndianStockMarket/new.json?limit=5",
        "https://www.reddit.com/r/IndianStockMarket.json?limit=5",
    ]
    
    async with httpx.AsyncClient(timeout=10, headers=headers, follow_redirects=True) as client:
        for url in urls:
            resp = await client.get(url)
            print(f"{url.split('/')[-1]}: {resp.status_code}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    children = data.get("data", {}).get("children", [])
                    print(f"  Posts: {len(children)}")
                    for c in children[:2]:
                        post = c.get("data", {})
                        print(f"  - {post.get('title', '?')[:80]}")
                except:
                    print(f"  Not JSON: {resp.text[:100]}")

asyncio.run(test())
