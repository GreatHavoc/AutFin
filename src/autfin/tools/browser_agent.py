"""Browser agent — connects to Obscura MCP for web scraping.

Provides a shared MCPToolset (headless Chrome via Obscura) that can be:
  1. Used by SentimentTools for Reddit/Moneycontrol/ET scraping
  2. Attached to the research agent for Trendlyne/Screener/SEBI scraping

Usage:
    from autfin.tools.browser_agent import browser_agent, browser_toolset

    # Option A: call the agent directly (for sentiment_tools.py)
    result = await browser_agent.run("Scrape Reddit for RELIANCE")
    print(result.output)

    # Option B: attach the toolset to another agent (for research.py)
    research_agent = Agent("...", toolsets=[browser_toolset, ...])
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset

def _obscura_endpoint() -> str:
    """Resolve the MCP endpoint from env, with the documented reader's defaults.

    Named exports win first, then BROWSER_MCP_ENDPOINT/PORT — the settings
    browser_client.py reads. This toolset used to default to a hardcoded
    :9222 while browser_client.py defaulted to :8080 and .env.example said
    :9333, so a stock deployment never connected to the Obscura it started.
    """
    explicit = os.getenv("OBSCURA_MCP_URL")
    if explicit:
        return explicit
    from autfin.mcp.browser_client import BrowserMCPSettings

    return BrowserMCPSettings.from_env().endpoint


_obscura_url = _obscura_endpoint()

_browser_toolset: MCPToolset | None = None
_browser_agent: Agent | None = None


def get_browser_toolset() -> MCPToolset:
    """Lazy singleton — creates MCPToolset on first call."""
    global _browser_toolset
    if _browser_toolset is None:
        from pydantic_ai.mcp import MCPToolset
        _browser_toolset = MCPToolset(_obscura_url)
    return _browser_toolset


def get_browser_agent() -> Agent:
    """Lazy singleton — creates browser Agent on first call."""
    global _browser_agent
    if _browser_agent is None:
        from pydantic_ai import Agent
        _browser_agent = Agent(
            "openai:gpt-4.1-mini",
            toolsets=[get_browser_toolset()],
            instructions=(
                "You are a web scraping agent. You control a headless Chrome browser.\n"
                "Navigate to pages, extract text content, handle popups and consent banners.\n"
                "Always return scraped data as clean structured text.\n"
                "Never hallucinate content — if a page fails to load, say so.\n"
                "When asked for JSON, return ONLY the JSON, no markdown fences."
            ),
        )
    return _browser_agent
