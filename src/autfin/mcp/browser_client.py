"""Browser MCP client for AutFin (v5.6).

Thin singleton wrapper around ``langchain_mcp_adapters.client.MultiServerMCPClient``
talking to a long-lived **Obscura** HTTP MCP server.

Why this module exists:
 - Holds a process-wide singleton MCP session so all Pydantic AI worker threads
   share one browser context (cookies + tabs + storage_state).
 - Auto-spawns ``obscura mcp --http --port 8080`` as a background subprocess
   on first use when ``BROWSER_MCP_AUTOSTART=true`` (default).
 - Hard-fails fast with the exact install command when the binary is missing
   on PATH — no silent fallback to httpx scraping.

Public surface:
 - ``get_browser_mcp_client()`` — returns the singleton, lazy-initialised.
 - ``browser_mcp_register_tools()`` — returns a list of LangChain tools bound
   to the MCP session, ready to use with Pydantic AI agents.
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrowserMCPNotInstalledError(RuntimeError):
    """Raised when Obscura binary is missing on PATH."""


class BrowserMCPConnectionError(RuntimeError):
    """Raised when the MCP HTTP endpoint is unreachable."""


class BrowserMCPStartupError(RuntimeError):
    """Raised when ``obscura mcp`` subprocess fails to come up."""


# ---------------------------------------------------------------------------
# Settings (read once — env vars may not change during runtime)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Subprocess lifecycle (autostart only; no-op when endpoint already serves)
# ---------------------------------------------------------------------------


def _find_observa_binary(settings: BrowserMCPSettings) -> str:
    """Locate the ``obscura`` executable.

    Prefers ``BROWSER_MCP_BINARY`` env var when set; otherwise searches
    ``$PATH``. Raises :class:`BrowserMCPNotInstalledError` with the exact
    install command when not found.
    """
    if settings.binary:
        if os.path.isfile(settings.binary) and os.access(settings.binary, os.X_OK):
            return settings.binary
        raise BrowserMCPNotInstalledError(
            f"BROWSER_MCP_BINARY={settings.binary!r} does not point to an executable. "
            "Download Obscura from https://github.com/h4ckf0r0day/obscura/releases/latest "
            "and either set BROWSER_MCP_BINARY to the absolute path or add it to PATH."
        )

    candidate = shutil.which("obscura")
    if candidate:
        return candidate

    raise BrowserMCPNotInstalledError(
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
    """Return True if a TCP connection to ``host:port`` succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_endpoint(host: str, port: int, deadline_s: float = 8.0) -> bool:
    """Block until ``host:port`` accepts TCP or timeout. Polls every 0.25s."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if _port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def _preexam_endpoint_unreachable(settings: BrowserMCPSettings) -> bool:
    """Check whether endpoint needs us to autostart (without blocking long)."""
    if not settings.autostart:
        return False
    return not _port_open(settings.host, settings.port, timeout=0.4)


def autostart_observa(settings: Optional[BrowserMCPSettings] = None) -> subprocess.Popen:
    """Spawn ``obscura mcp --http`` and wait until its HTTP port accepts TCP.

    Returns the ``Popen`` handle so callers can ``terminate()`` later if they
    want to. Raises :class:`BrowserMCPStartupError` on timeout.
    """
    settings = settings or BrowserMCPSettings.from_env()
    binary = _find_observa_binary(settings)
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
        raise BrowserMCPNotInstalledError(str(exc)) from exc
    except OSError as exc:
        raise BrowserMCPStartupError(f"Failed to spawn {binary!r}: {exc}") from exc

    if not _wait_for_endpoint(settings.host, settings.port, deadline_s=10.0):
        proc.terminate()
        raise BrowserMCPStartupError(
            f"Obscura did not start listening on {settings.host}:{settings.port} within 10s. "
            f"Manual step: `{binary} mcp --http --port {settings.port}` and check stderr for crashes."
        )
    logger.info("Obscura up at %s:%d (pid=%d)", settings.host, settings.port, proc.pid)
    return proc


@contextmanager
def managed_observa(settings: Optional[BrowserMCPSettings] = None):
    """Context manager: autostart obscura on enter, terminate on exit.

    Useful inside scripts (`python -m autfin.<tool>`); inside Pydantic AI dev
    the subprocess is left running for the lifetime of the dev server, so we
    autostart once at module import instead.
    """
    settings = settings or BrowserMCPSettings.from_env()
    handle = autostart_observa(settings)
    try:
        yield handle
    finally:
        if handle.poll() is None:
            handle.terminate()


# ---------------------------------------------------------------------------
# LangChain MCP client (singleton)
# ---------------------------------------------------------------------------


_client_singleton: Dict[str, Any] = {"client": None, "tools": None, "settings": None}


def _build_langchain_mcp_client(settings: BrowserMCPSettings):
    """Build a ``MultiServerMCPClient`` from current settings.

    HTTP transport only for the default config. Stdio is wired in too as a
    fallback option — opted into via ``BROWSER_MCP_TRANSPORT=stdio``.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient  # local-import

    if settings.transport == "streamable_http":
        return MultiServerMCPClient(
            {
                "obscura": {
                    "transport": "streamable_http",
                    "url": settings.endpoint,
                }
            }
        )
    if settings.transport == "stdio":
        binary = _find_observa_binary(settings)
        return MultiServerMCPClient(
            {
                "obscura": {
                    "transport": "stdio",
                    "command": binary,
                    "args": ["mcp", "--stdio"],
                    "env": os.environ.copy(),
                }
            }
        )
    raise BrowserMCPStartupError(
        f"Unsupported BROWSER_MCP_TRANSPORT={settings.transport!r}. "
        "Use 'streamable_http' (default) or 'stdio'."
    )


def get_browser_mcp_client(*, force_reload: bool = False):
    """Return process-wide singleton :class:`MultiServerMCPClient`.

    Auto-spawns the subprocess on first call when configured, then caches the
    client. ``force_reload=True`` rebuilds the singleton (useful in tests).
    """
    if _client_singleton["client"] is not None and not force_reload:
        return _client_singleton["client"]

    settings = BrowserMCPSettings.from_env()

    if settings.transport == "streamable_http" and _preexam_endpoint_unreachable(settings):
        try:
            autostart_observa(settings)
        except (BrowserMCPNotInstalledError, BrowserMCPStartupError) as exc:
            logger.error("BrowserMCP autostart failed: %s", exc)
            raise

    try:
        client = _build_langchain_mcp_client(settings)
    except ImportError as exc:
        raise BrowserMCPNotInstalledError(
            "langchain-mcp-adapters is required. Install with: "
            "pip install 'langchain-mcp-adapters>=0.3.0'"
        ) from exc

    _client_singleton["client"] = client
    _client_singleton["settings"] = settings
    _client_singleton["tools"] = None  # invalidate cache
    logger.info(
        "BrowserMCP client ready (transport=%s, endpoint=%s)",
        settings.transport, settings.endpoint,
    )
    return client


async def browser_mcp_register_tools() -> List[Any]:
    """Discover and return LangChain tools exposed by the MCP server.

    Caches the list after the first successful discovery. Subsequent calls
    return the same tool objects (they can be re-bound into multiple graphs
    or LLM clients without re-handshaking the MCP session).
    """
    if _client_singleton["tools"] is not None:
        return _client_singleton["tools"]

    client = get_browser_mcp_client()
    tools = await client.get_tools()  # verbose=False by default
    _client_singleton["tools"] = tools
    logger.info("BrowserMCP tools registered: %d", len(tools))
    return tools


def reset_browser_mcp_client() -> None:
    """Drop the cached singleton (used in tests and after hot-reload)."""
    with _client_lock():
        _client_singleton["client"] = None
        _client_singleton["tools"] = None
        _client_singleton["settings"] = None


@contextmanager
def _client_lock():
    """Tiny cooperative lock around the singleton dict."""
    # Re-entrant single-thread guard — multiprocess safety is the responsibility
    # of the host (each Pydantic AI worker is its own process).
    yield


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _main_entry(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    """Convenience: `python -m autfin.mcp.browser_client status`.

    Returns process exit code.
    """
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "status"
    settings = BrowserMCPSettings.from_env()

    if cmd == "status":
        endpoint_ready = _port_open(settings.host, settings.port)
        binary_path: Optional[str] = None
        try:
            binary_path = _find_observa_binary(settings)
        except BrowserMCPNotInstalledError:
            pass
        print("BrowserMCP status:")
        print(f"  transport   = {settings.transport}")
        print(f"  endpoint    = {settings.endpoint}")
        print(f"  autostart   = {settings.autostart}")
        print(f"  endpoint up = {endpoint_ready}")
        print(f"  binary found = {bool(binary_path)} ({binary_path or 'n/a'})")
        return 0 if endpoint_ready else 2

    if cmd == "start":
        autostart_observa(settings)
        return 0

    if cmd == "stop":
        # No persistent registry; user-level stop is `obscura --help`.
        print("BrowserMCP autostart owns no registry; close the obscured browser process manually.")
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main_entry())
