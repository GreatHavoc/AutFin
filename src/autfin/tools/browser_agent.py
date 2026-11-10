"""Browser MCP for AutFin — Obscura lifecycle + the pydantic-ai toolset.

One module instead of the former `mcp/browser_client.py` (a langchain
singleton) + a hardcoded URL in this file. Settings and the subprocess
lifecycle now live here, next to the toolset that uses them, so the agent path
spawns Obscura itself and dials the same port it started.

What it provides:
  - `get_browser_toolset()`  — shared MCPToolset for the research agent, lazy
  - `get_browser_agent()`    — standalone scraping agent, lazy
  - `autostart_obscura()`    — spawn `obscura mcp --http` and wait for the port
  - `managed_obscura()`      — context manager, terminates on exit
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPToolset

logger = logging.getLogger(__name__)


class ObscuraNotInstalledError(RuntimeError):
    """Obscura binary is missing on PATH."""


class ObscuraStartupError(RuntimeError):
    """`obscura mcp` came up but never started listening."""


@dataclass
class BrowserMCPSettings:
    transport: str          # "streamable_http" | "stdio"
    host: str               # e.g. "127.0.0.1"
    port: int               # e.g. 8080
    endpoint: str           # e.g. "http://127.0.0.1:8080/mcp"
    autostart: bool         # spawn obscura subprocess if endpoint not serving
    binary: str             # explicit obscura path; "" → search PATH

    @classmethod
    def from_env(cls) -> "BrowserMCPSettings":
        transport = (os.getenv("BROWSER_MCP_TRANSPORT", "streamable_http") or "").strip().lower()
        host = (os.getenv("BROWSER_MCP_HOST", "127.0.0.1") or "127.0.0.1").strip()
        port = int(os.getenv("BROWSER_MCP_PORT", "8080") or "8080")
        endpoint = (
            os.getenv("BROWSER_MCP_ENDPOINT")
            or f"http://{host}:{port}/mcp"
        ).strip()
        autostart = (os.getenv("BROWSER_MCP_AUTOSTART", "true") or "").strip().lower() in (
            "1", "true", "yes", "on"
        )
        binary = (os.getenv("BROWSER_MCP_BINARY") or "").strip()
        return cls(
            transport=transport,
            host=host,
            port=port,
            endpoint=endpoint,
            autostart=autostart,
            binary=binary,
        )


def _find_obscura_binary(settings: BrowserMCPSettings) -> str:
    """Locate the obscura executable, with install instructions on failure."""
    if settings.binary:
        if os.path.isfile(settings.binary) and os.access(settings.binary, os.X_OK):
            return settings.binary
        raise ObscuraNotInstalledError(
            f"BROWSER_MCP_BINARY={settings.binary!r} does not point to an executable. "
            "Download Obscura from https://github.com/h4ckf0r0day/obscura/releases/latest "
            "and either set BROWSER_MCP_BINARY to the absolute path or add it to PATH."
        )

    candidate = shutil.which("obscura")
    if candidate:
        return candidate

    raise ObscuraNotInstalledError(
        "Obscura binary not found on PATH.\n"
        "  Install: download a release `.zip`/`.tar.gz` from "
        "https://github.com/h4ckf0r0day/obscura/releases/latest,\n"
        "  extract, and add the folder to your PATH.\n"
        "  Verify with `obscura --version`.\n"
        "  Then either:\n"
        "    - run `obscura mcp --http --port 8080` yourself, OR\n"
        "    - set BROWSER_MCP_AUTOSTART=true (default) and AutFin will spawn it."
    )


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_endpoint(host: str, port: int, deadline_s: float = 10.0) -> bool:
    """Block until host:port accepts TCP, or time out."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if _port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def obscura_endpoint() -> str:
    """Resolve the MCP endpoint. `OBSCURA_MCP_URL` overrides the settings."""
    explicit = os.getenv("OBSCURA_MCP_URL")
    if explicit:
        return explicit
    return BrowserMCPSettings.from_env().endpoint


_obscura_url = obscura_endpoint()

_obscura_process: subprocess.Popen | None = None
_browser_toolset: "MCPToolset | None" = None
_browser_agent: "Agent | None" = None


def autostart_obscura(settings: BrowserMCPSettings | None = None) -> subprocess.Popen:
    """Spawn `obscura mcp --http` and wait until its port accepts TCP.

    Returns the Popen handle. Raises ObscuraNotInstalledError when the binary
    is missing and ObscuraStartupError when it never listens — never a silent
    httpx fallback.
    """
    settings = settings or BrowserMCPSettings.from_env()
    binary = _find_obscura_binary(settings)
    cmd = [binary, "mcp", "--http", "--host", settings.host, "--port", str(settings.port)]
    logger.info("Auto-starting Obscura: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        raise ObscuraNotInstalledError(str(exc)) from exc
    except OSError as exc:
        raise ObscuraStartupError(f"Failed to spawn {binary!r}: {exc}") from exc

    if not _wait_for_endpoint(settings.host, settings.port):
        proc.terminate()
        raise ObscuraStartupError(
            f"Obscura did not start listening on {settings.host}:{settings.port} within 10s. "
            f"Manual step: `{binary} mcp --http --port {settings.port}` and check for crashes."
        )
    logger.info("Obscura up at %s:%d (pid=%d)", settings.host, settings.port, proc.pid)
    return proc


def ensure_obscura_running() -> subprocess.Popen | None:
    """Make sure an Obscura MCP server is reachable, starting one if needed.

    Returns the Popen handle when this process started the server, else None
    (already running, or autostart disabled).
    """
    global _obscura_process

    settings = BrowserMCPSettings.from_env()
    if _port_open(settings.host, settings.port, timeout=0.4):
        return None  # someone else is serving

    if not settings.autostart:
        logger.warning(
            "No Obscura on %s:%d and BROWSER_MCP_AUTOSTART is off — browser tools will fail",
            settings.host, settings.port,
        )
        return None

    if _obscura_process is not None and _obscura_process.poll() is None:
        return _obscura_process  # ours, still alive

    _obscura_process = autostart_obscura(settings)
    return _obscura_process


def stop_obscura() -> None:
    """Terminate the Obscura subprocess this process started (if any)."""
    global _obscura_process
    if _obscura_process is not None and _obscura_process.poll() is None:
        _obscura_process.terminate()
    _obscura_process = None


@contextmanager
def managed_obscura(settings: BrowserMCPSettings | None = None):
    """Autostart on enter, terminate on exit — for scripts and the daemon."""
    handle = autostart_obscura(settings)
    try:
        yield handle
    finally:
        if handle.poll() is None:
            handle.terminate()


def get_browser_toolset() -> "MCPToolset":
    """Lazy singleton — best-effort Obscura startup, then the toolset.

    Autostart failures are logged, not raised: this is called at module import
    (research_agent builds its toolset list there), so raising would make the
    whole autfin package unimportable on a machine without Obscura. The MCP
    connection itself still fails loudly when an agent calls a browser tool.
    """
    global _browser_toolset
    if _browser_toolset is None:
        from pydantic_ai.mcp import MCPToolset

        try:
            ensure_obscura_running()
        except (ObscuraNotInstalledError, ObscuraStartupError) as exc:
            logger.error("Obscura unavailable — browser tools will fail: %s", exc)
        _browser_toolset = MCPToolset(obscura_endpoint())
    return _browser_toolset


def get_browser_agent() -> "Agent":
    """Lazy singleton — creates the browser Agent on first call."""
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
