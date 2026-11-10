"""AutFin MCP integration layer.

.. deprecated-removed:: v5.6
    Per *V5_6_BROWSER_MCP_PLAN*, the previous ``tools/scraper_tools.py``,
    Firecrawl key, and httpx-MC/ET regex paths are retired in favour of the
    long-lived Obscura MCP server described in :mod:`autfin.mcp.browser_client`.

Hard runtime dependency on Obscura (Apache-2.0 Rust binary, ~30 MB RAM).
Download instructions and transport config: see ``.env.example``.
"""
from __future__ import annotations

from .browser_client import (
    BrowserMCPConnectionError,
    BrowserMCPNotInstalledError,
    BrowserMCPSettings,
    BrowserMCPStartupError,
    autostart_observa,
    browser_mcp_register_tools,
    get_browser_mcp_client,
    managed_observa,
    reset_browser_mcp_client,
)

__all__ = [
    "BrowserMCPConnectionError",
    "BrowserMCPNotInstalledError",
    "BrowserMCPStartupError",
    "BrowserMCPSettings",
    "autostart_observa",
    "browser_mcp_register_tools",
    "get_browser_mcp_client",
    "managed_observa",
    "reset_browser_mcp_client",
]
