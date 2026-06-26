"""Firecrawl-based web scraping tools for financial data.

Also includes nselib-powered NSE data functions for corporate actions
and event calendar (complements the HTTP-based announcement scraper).
"""

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
    """Firecrawl-based web scraping tools for financial data.

    Also includes nselib-powered NSE data functions for corporate actions
    and event calendar (complements the HTTP-based announcement scraper).
    """

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

        NSE blocks direct API calls without cookies. This method first visits
        the main page to establish a session, then fetches the announcements API.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE").
            limit: Maximum announcements to return.

        Returns:
            List of dicts with keys: symbol, desc, date, attachment.
        """
        symbol_upper = symbol.upper().strip()
        try:
            import httpx

            # NSE requires cookies from the main page before API calls
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            async with httpx.AsyncClient(
                timeout=15.0,
                headers=headers,
                follow_redirects=True,
            ) as session:
                # Step 1: Visit main page to get cookies
                try:
                    await session.get("https://www.nseindia.com", timeout=10.0)
                except Exception:
                    logger.debug("NSE main page fetch failed, trying API anyway")

                # Step 2: Fetch announcements API with cookies
                api_url = (
                    "https://www.nseindia.com/api/corporate-announcements"
                    f"?index=equities&symbol={symbol_upper}"
                )
                response = await session.get(api_url, timeout=15.0)

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
    # NSE via nselib (corporate actions, event calendar)
    # ------------------------------------------------------------------

    def get_nse_corporate_actions(
        self, symbol: str, period: str = "1M"
    ) -> List[Dict[str, Any]]:
        """Get corporate actions (dividends, splits, bonuses) via nselib.

        Uses nselib's bulk API — fetches all actions for the period,
        then filters for the requested symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE").
            period: Lookback period — "1W", "1M", "3M", "6M", "1Y".

        Returns:
            List of dicts with keys: symbol, subject, ex_date, rec_date.
        """
        symbol_upper = symbol.upper().strip()
        try:
            from nselib import capital_market
            df = capital_market.corporate_actions_for_equity(period=period)
            if df is None or df.empty:
                return []

            # Filter for the requested symbol
            sym_col = "symbol" if "symbol" in df.columns else " SYMBOL"
            filtered = df[df[sym_col].str.upper() == symbol_upper]

            results = []
            for _, row in filtered.iterrows():
                results.append({
                    "symbol": symbol_upper,
                    "subject": row.get("subject", ""),
                    "ex_date": row.get("exDate", ""),
                    "rec_date": row.get("recDate", ""),
                    "face_val": row.get("faceVal", ""),
                    "series": row.get("series", ""),
                })
            return results
        except ImportError:
            logger.debug("nselib not installed, skipping corporate actions")
            return []
        except Exception:
            logger.debug(
                "get_nse_corporate_actions failed for %s", symbol_upper, exc_info=True
            )
            return []

    def get_nse_event_calendar(
        self, symbol: str, period: str = "1M"
    ) -> List[Dict[str, Any]]:
        """Get event calendar (board meetings, fund raising) via nselib.

        Uses nselib's bulk API — fetches all events for the period,
        then filters for the requested symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE").
            period: Lookback period — "1W", "1M", "3M", "6M", "1Y".

        Returns:
            List of dicts with keys: symbol, company, purpose, date, description.
        """
        symbol_upper = symbol.upper().strip()
        try:
            from nselib import capital_market
            df = capital_market.event_calendar_for_equity(period=period)
            if df is None or df.empty:
                return []

            filtered = df[df["symbol"].str.upper() == symbol_upper]

            results = []
            for _, row in filtered.iterrows():
                results.append({
                    "symbol": symbol_upper,
                    "company": row.get("company", ""),
                    "purpose": row.get("purpose", ""),
                    "date": row.get("date", ""),
                    "description": row.get("bm_desc", ""),
                })
            return results
        except ImportError:
            logger.debug("nselib not installed, skipping event calendar")
            return []
        except Exception:
            logger.debug(
                "get_nse_event_calendar failed for %s", symbol_upper, exc_info=True
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
