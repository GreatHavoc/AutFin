"""Firecrawl-based web scraping tools for financial data."""

import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1/scrape"


class _TextExtractor(HTMLParser):
    """Lightweight HTML-to-text extractor for fallback scraping."""

    def __init__(self):
        super().__init__()
        self._text_parts: List[str] = []
        self._skip_tags = {"script", "style", "noscript", "iframe", "svg", "nav", "footer", "header"}
        self._skip_depth = 0
        self._tag_stack: List[str] = []

    def handle_starttag(self, tag, attrs):
        self._tag_stack.append(tag)
        if tag.lower() in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag.lower() in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._text_parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._text_parts)


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from raw HTML."""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        logger.debug("HTML parse error in _TextExtractor", exc_info=True)
    text = extractor.get_text()
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class ScraperTools:
    """Firecrawl-based web scraping tools for financial data."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ScraperTools.

        Args:
            api_key: Firecrawl API key. If None, loads from FIRECRAWL_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY", "")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": "AutFin-ScraperAgent/1.0"},
                follow_redirects=True,
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
    # Core scrape
    # ------------------------------------------------------------------

    async def scrape_page(self, url: str, format: str = "markdown") -> Dict[str, Any]:
        """
        Scrape a single URL.

        Uses Firecrawl API if key is configured, otherwise falls back to
        direct HTTP fetch and HTML-to-text extraction.

        Args:
            url: The URL to scrape.
            format: Desired output format (passed to Firecrawl as "formats").

        Returns:
            {"content": str, "metadata": dict}
        """
        try:
            if self.api_key:
                return await self._scrape_via_firecrawl(url, format)
            else:
                return await self._scrape_via_httpx(url)
        except Exception:
            logger.debug("scrape_page failed for %s", url, exc_info=True)
            return {"content": "", "metadata": {}}

    async def _scrape_via_firecrawl(self, url: str, format: str) -> Dict[str, Any]:
        """Scrape using Firecrawl API."""
        client = await self._get_client()
        response = await client.post(
            FIRECRAWL_API_URL,
            json={
                "url": url,
                "formats": [format],
                "onlyMainContent": True,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()
        content = ""
        if data.get("success") and data.get("data"):
            content = data["data"].get(format, "")
        return {
            "content": content or "",
            "metadata": data.get("data", {}).get("metadata", {}),
        }

    async def _scrape_via_httpx(self, url: str) -> Dict[str, Any]:
        """Scrape by fetching raw HTML and extracting text."""
        client = await self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        text = _extract_text_from_html(response.text)
        return {
            "content": text,
            "metadata": {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "url": str(response.url),
            },
        }

    # ------------------------------------------------------------------
    # Screener.in
    # ------------------------------------------------------------------

    async def scrape_screener_in(self, symbol: str) -> Dict[str, Any]:
        """
        Scrape Screener.in for fundamental data of a symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS").

        Returns:
            {"content": str, "symbol": str, "url": str, "metadata": dict}
        """
        symbol_upper = symbol.upper().strip()
        url = f"https://www.screener.in/company/{symbol_upper}/consolidated/"
        try:
            result = await self.scrape_page(url, format="markdown")
            result["symbol"] = symbol_upper
            result["url"] = url
            return result
        except Exception:
            logger.debug("scrape_screener_in failed for %s", symbol_upper, exc_info=True)
            return {"content": "", "symbol": symbol_upper, "url": url, "metadata": {}}

    # ------------------------------------------------------------------
    # NSE Announcements
    # ------------------------------------------------------------------

    async def scrape_nse_announcements(
        self, symbol: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Scrape NSE corporate announcements for a symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE").
            limit: Maximum announcements to return.

        Returns:
            List of dicts with keys: symbol, desc, date, attachment.
        """
        symbol_upper = symbol.upper().strip()
        url = (
            "https://www.nseindia.com/api/corporate-announcements"
            f"?index=equities&symbol={symbol_upper}"
        )
        try:
            client = await self._get_client()
            response = await client.get(url, timeout=15.0)
            if response.status_code != 200:
                logger.debug(
                    "NSE announcements HTTP %d for %s",
                    response.status_code,
                    symbol_upper,
                )
                return []

            data = response.json()
            announcements: List[Dict[str, Any]] = []
            raw_items = data if isinstance(data, list) else data.get("data", [])
            for item in raw_items[:limit]:
                announcements.append({
                    "symbol": item.get("symbol", symbol_upper),
                    "desc": item.get("desc", item.get("sm_name", "")),
                    "date": item.get("an_dt", item.get("sort_date", "")),
                    "attachment": item.get("attachment", ""),
                })
            return announcements
        except httpx.HTTPError:
            logger.debug(
                "NSE announcements HTTP error for %s", symbol_upper, exc_info=True
            )
            return []
        except Exception:
            logger.debug(
                "scrape_nse_announcements failed for %s", symbol_upper, exc_info=True
            )
            return []

    # ------------------------------------------------------------------
    # Moneycontrol News
    # ------------------------------------------------------------------

    async def scrape_moneycontrol_news(
        self, symbol: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Scrape Moneycontrol news articles for a symbol.

        Args:
            symbol: Stock symbol or company name.
            limit: Maximum articles to return.

        Returns:
            List of dicts with keys: title, url, date, snippet.
        """
        symbol_lower = symbol.lower().strip()
        url = f"https://www.moneycontrol.com/news/business/stocks/?s={symbol_lower}"
        try:
            client = await self._get_client()
            response = await client.get(url)
            if response.status_code != 200:
                logger.debug(
                    "Moneycontrol HTTP %d for %s", response.status_code, symbol
                )
                return []

            html = response.text
            articles = _parse_moneycontrol_news(html, limit)
            return articles
        except httpx.HTTPError:
            logger.debug(
                "Moneycontrol HTTP error for %s", symbol, exc_info=True
            )
            return []
        except Exception:
            logger.debug(
                "scrape_moneycontrol_news failed for %s", symbol, exc_info=True
            )
            return []

    # ------------------------------------------------------------------
    # Economic Times
    # ------------------------------------------------------------------

    async def scrape_economic_times(
        self, topic: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Scrape Economic Times for news on a topic.

        Args:
            topic: News topic or keyword.
            limit: Maximum articles to return.

        Returns:
            List of dicts with keys: title, url, date, snippet.
        """
        topic_slug = re.sub(r"[^a-zA-Z0-9]+", "-", topic.lower().strip()).strip("-")
        url = f"https://economictimes.indiatimes.com/topic/{topic_slug}"
        try:
            client = await self._get_client()
            response = await client.get(url)
            if response.status_code != 200:
                logger.debug(
                    "Economic Times HTTP %d for %s", response.status_code, topic
                )
                return []

            html = response.text
            articles = _parse_economic_times(html, limit)
            return articles
        except httpx.HTTPError:
            logger.debug(
                "Economic Times HTTP error for %s", topic, exc_info=True
            )
            return []
        except Exception:
            logger.debug(
                "scrape_economic_times failed for %s", topic, exc_info=True
            )
            return []


# ------------------------------------------------------------------
# HTML parsing helpers for Moneycontrol & Economic Times
# ------------------------------------------------------------------

def _parse_moneycontrol_news(html: str, limit: int) -> List[Dict[str, Any]]:
    """Extract news articles from Moneycontrol HTML."""
    articles: List[Dict[str, Any]] = []
    seen_urls: set = set()

    # Match <a> tags containing /news/ in href with a title attribute
    link_pattern = re.compile(
        r'<a[^>]*href\s*=\s*["\'](/news/[^"\']*)["\'][^>]*title\s*=\s*["\']([^"\']*)["\']',
        re.IGNORECASE,
    )
    for match in link_pattern.finditer(html):
        if len(articles) >= limit:
            break
        href = match.group(1)
        title = match.group(2)
        full_url = f"https://www.moneycontrol.com{href}" if href.startswith("/") else href
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        articles.append({
            "title": title.strip(),
            "url": full_url,
            "date": "",
            "snippet": "",
        })

    # If the above yields nothing, try finding <li> list items with links
    if not articles:
        li_pattern = re.compile(
            r'<li[^>]*>.*?<a[^>]*href\s*=\s*["\'](/news/[^"\']*)["\'][^>]*>(.*?)</a>.*?</li>',
            re.DOTALL | re.IGNORECASE,
        )
        for match in li_pattern.finditer(html):
            if len(articles) >= limit:
                break
            href = match.group(1)
            inner_html = match.group(2)
            title = re.sub(r"<[^>]+>", "", inner_html).strip()
            full_url = f"https://www.moneycontrol.com{href}" if href.startswith("/") else href
            if full_url in seen_urls or not title:
                continue
            seen_urls.add(full_url)
            articles.append({
                "title": title,
                "url": full_url,
                "date": "",
                "snippet": "",
            })

    return articles


def _parse_economic_times(html: str, limit: int) -> List[Dict[str, Any]]:
    """Extract news articles from Economic Times HTML."""
    articles: List[Dict[str, Any]] = []
    seen_urls: set = set()

    # ET topic pages often embed article data in JSON-like script blocks
    json_pattern = re.compile(
        r'"title":"([^"]*)".*?"url":"([^"]*)".*?"datePublished":"([^"]*)"',
        re.IGNORECASE,
    )
    for match in json_pattern.finditer(html):
        if len(articles) >= limit:
            break
        title = match.group(1).replace("\\/", "/")
        url = match.group(2).replace("\\/", "/")
        date = match.group(3).replace("\\/", "/")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        articles.append({
            "title": title,
            "url": url,
            "date": date,
            "snippet": "",
        })

    # Fallback: look for standard article links
    if not articles:
        link_pattern = re.compile(
            r'<a[^>]*href\s*=\s*["\'](https?://economictimes\.indiatimes\.com/[^"\']*/articleshow/\d+\.cms[^"\']*)["\'][^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        for match in link_pattern.finditer(html):
            if len(articles) >= limit:
                break
            url = match.group(1)
            inner = match.group(2)
            title = re.sub(r"<[^>]+>", "", inner).strip()
            if url in seen_urls or not title:
                continue
            seen_urls.add(url)
            articles.append({
                "title": title,
                "url": url,
                "date": "",
                "snippet": "",
            })

    return articles
