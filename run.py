"""AutFin daemon — starts Obscura MCP, runs morning routine daily.

Usage:
    python run.py                # paper mode (default)
    python run.py --mode live    # live trading
    python run.py --once         # run once and exit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import logfire

logfire.configure(service_name="autfin")
logfire.instrument_pydantic_ai()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("autfin.log")],
)
log = logging.getLogger("autfin")

_stop = False

OBSCURA_DIR = Path(os.getenv("OBSCURA_DIR", r"C:\Users\Prakhar\Downloads\obscura-x86_64-windows"))
OBSCURA_PORT = int(os.getenv("OBSCURA_PORT", "9222"))

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def _handle_signal(sig, frame):
    global _stop
    log.info("Shutdown signal received, stopping...")
    _stop = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _start_obscura() -> subprocess.Popen | None:
    """Start Obscura MCP server if not already running."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(("127.0.0.1", OBSCURA_PORT))
        sock.close()
        log.info("Obscura already running on port %d", OBSCURA_PORT)
        return None
    except ConnectionRefusedError:
        pass

    exe = OBSCURA_DIR / "obscura.exe"
    if not exe.exists():
        log.warning("Obscura not found at %s — browser agent will be unavailable", exe)
        return None

    log.info("Starting Obscura MCP on port %d...", OBSCURA_PORT)
    proc = subprocess.Popen(
        [str(exe), "mcp", "--http", "--port", str(OBSCURA_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # Wait for it to come up
    for _ in range(30):
        time.sleep(0.5)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", OBSCURA_PORT))
            sock.close()
            log.info("Obscura ready on port %d", OBSCURA_PORT)
            return proc
        except ConnectionRefusedError:
            continue
    log.warning("Obscura failed to start within 15s")
    return proc


def _is_market_hours() -> bool:
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Asia/Kolkata")).time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def _next_market_open() -> datetime:
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    target = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now.time() >= MARKET_OPEN:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


async def _run_morning(mode: str):
    from autfin.agents.supervisor import supervisor_agent
    from autfin.types import AgentDeps

    log.info("Starting morning routine (mode=%s)", mode)
    deps = AgentDeps(user_id="daemon", mode=mode)

    try:
        result = await supervisor_agent.run(
            "Run the complete morning routine: market snapshot, analyze holdings, research, risk, conference",
            deps=deps,
        )
        log.info("Morning routine: %s", result.output)
        return result.output
    except Exception:
        log.exception("Morning routine failed")
        return None


async def _run_quarterly(mode: str):
    from autfin.agents.supervisor import supervisor_agent
    from autfin.types import AgentDeps

    log.info("Starting quarterly review (mode=%s)", mode)
    deps = AgentDeps(user_id="daemon", mode=mode)

    try:
        result = await supervisor_agent.run(
            "Run quarterly review: deep analysis, sector rotation, rebalance",
            deps=deps,
        )
        log.info("Quarterly review: %s", result.output)
        return result.output
    except Exception:
        log.exception("Quarterly review failed")
        return None


async def daemon(mode: str, once: bool = False):
    log.info("AutFin daemon starting (mode=%s, once=%s)", mode, once)

    # Start Obscura MCP
    obscura_proc = _start_obscura()

    last_quarterly = datetime.now()

    # Run immediately on startup
    await _run_morning(mode)

    if once:
        log.info("Single run complete, exiting.")
        if obscura_proc:
            obscura_proc.terminate()
        return

    try:
        while not _stop:
            next_open = _next_market_open()
            wait_secs = (next_open - datetime.now(next_open.tzinfo)).total_seconds()
            if wait_secs > 0:
                log.info("Next run at %s (sleeping %.0fs)", next_open.strftime("%Y-%m-%d %H:%M IST"), wait_secs)
                slept = 0
                while slept < wait_secs and not _stop:
                    chunk = min(60, wait_secs - slept)
                    await asyncio.sleep(chunk)
                    slept += chunk

            if _stop:
                break

            if not _is_market_hours() and datetime.now().weekday() < 5:
                log.info("Outside market hours, waiting...")
                continue

            await _run_morning(mode)

            if (datetime.now() - last_quarterly).days >= 90:
                await _run_quarterly(mode)
                last_quarterly = datetime.now()
    finally:
        if obscura_proc:
            log.info("Stopping Obscura...")
            obscura_proc.terminate()

    log.info("AutFin daemon stopped.")


def main():
    parser = argparse.ArgumentParser(description="AutFin autonomous financial system")
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], default="paper")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    asyncio.run(daemon(mode=args.mode, once=args.once))


if __name__ == "__main__":
    main()
