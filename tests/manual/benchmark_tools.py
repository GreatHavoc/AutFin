"""
Manual benchmark of all 10 v5.4 tools.

Tests against real symbols:
  - Indian:   ETERNAL  (Eternal Limited / Zomato, NSE ticker was ZOMATO,
              renamed to ETERNAL after Blinkit / Hyperpure re-organization)
  - Indian:   RELIANCE (fallback to verify flows)
  - Global:   AAPL     (for sentiment / cross-market reference)

Outputs input → output → time for each tool.

Run from project root:
    .\\.venv\\Scripts\\python.exe tests\\manual\\benchmark_tools.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ETERNAL is the NSE ticker for Eternal Limited (formerly Zomato).
# Both refer to the same company — yfinance streams ETERNAL.NS.
ETERNAL = "ETERNAL"     # ex-Zomato
RELIANCE = "RELIANCE"
APPLE = "AAPL"


def banner(stage: str) -> None:
    print("\n" + "=" * 78)
    print(stage)
    print("=" * 78)


async def bench(label: str, coro) -> None:
    """Run an async coroutine and pretty-print input/output/time."""
    start = time.monotonic()
    try:
        result = await coro
        elapsed = time.monotonic() - start
        success = True
        error = None
    except Exception as e:
        elapsed = time.monotonic() - start
        success = False
        error = repr(e)
        result = None

    print(f"\n--- {label}  [{elapsed:.2f}s]")
    if not success:
        print(f"  EXCEPTION: {error}")
        return

    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            pretty = json.dumps(parsed, indent=2, default=str)
        except Exception:
            pretty = result
    elif isinstance(result, dict):
        pretty = json.dumps(result, indent=2, default=str)
    else:
        pretty = str(result)

    if len(pretty) > 1600:
        pretty = pretty[:1600] + "\n  ... [truncated] ..."
    print("  " + pretty.replace("\n", "\n  "))


async def main():
    banner("AUTFIN v5.4 — TOOL BENCHMARK")
    print(f"  Indian symbol #1: {ETERNAL}  (Eternal Limited, ex-Zomato)")
    print(f"  Indian symbol #2: {RELIANCE} (fallback to see real flows)")
    print(f"  Global symbol   : {APPLE}")

    from src.autfin.agents.agent_tools import (
        fundamentals_tool,
        technical_tool,
        sentiment_tool,
        macro_tool,
        research_tool,
        risk_tool,
        distortion_tool,
        regulatory_tool,
        valuation_tool,
        portfolio_tool,
    )

    # ── 1. fundamentals ──────────────────────────────────────────────────
    banner("1. fundamentals_analysis")
    await bench(f"INPUT ({ETERNAL}, standard)", fundamentals_tool(ETERNAL, "standard"))
    await bench(f"INPUT ({RELIANCE}, standard)", fundamentals_tool(RELIANCE, "standard"))

    # ── 2. technical ─────────────────────────────────────────────────────
    banner("2. technical_analysis")
    await bench(f"INPUT ({ETERNAL}, standard)", technical_tool(ETERNAL, "standard"))
    await bench(f"INPUT ({RELIANCE}, standard)", technical_tool(RELIANCE, "standard"))

    # ── 3. sentiment ─────────────────────────────────────────────────────
    banner("3. sentiment_analysis")
    await bench(f"INPUT ({ETERNAL}, Indian)", sentiment_tool(ETERNAL))
    await bench(f"INPUT ({APPLE}, Global)", sentiment_tool(APPLE))

    # ── 4. macro ─────────────────────────────────────────────────────────
    banner("4. macro_analysis (no symbol — global macro)")
    await bench("INPUT ()", macro_tool())
    await bench("INPUT () again (cache warm)", macro_tool())

    # ── 5. distortion ────────────────────────────────────────────────────
    banner("5. distortion_audit")
    await bench(f"INPUT ({ETERNAL}, standard)", distortion_tool(ETERNAL, "standard"))
    await bench(f"INPUT ({RELIANCE}, standard)", distortion_tool(RELIANCE, "standard"))

    # ── 6. regulatory ────────────────────────────────────────────────────
    banner("6. regulatory_analysis")
    await bench(f"INPUT ({ETERNAL}, standard)", regulatory_tool(ETERNAL, "standard"))
    await bench(f"INPUT ({RELIANCE}, standard)", regulatory_tool(RELIANCE, "standard"))

    # ── 7. valuation ─────────────────────────────────────────────────────
    banner("7. valuation_analysis")
    await bench(f"INPUT ({ETERNAL}, standard)", valuation_tool(ETERNAL, "standard"))
    await bench(f"INPUT ({RELIANCE}, standard)", valuation_tool(RELIANCE, "standard"))

    # ── 8. risk ───────────────────────────────────────────────────────────
    banner("8. risk_assessment")
    trade_e = json.dumps({"symbol": ETERNAL, "action": "BUY", "quantity": 50, "price": 280.0})
    trade_r = json.dumps({"symbol": RELIANCE, "action": "BUY", "quantity": 50, "price": 1300.0})
    await bench(f"INPUT {trade_e}", risk_tool(trade_e))
    await bench(f"INPUT {trade_r}", risk_tool(trade_r))

    # ── 9. portfolio ──────────────────────────────────────────────────────
    banner("9. portfolio_allocation")
    await bench("INPUT ()", portfolio_tool())

    # ── 10. research synthesis (LLM, expensive) ──────────────────────────
    banner("10. research_synthesis (LLM-driven — Bull/Bear debate)")
    print(f"\n  Skipping in benchmark. Would call:")
    print(f"    research_tool('{ETERNAL},{RELIANCE}', 'standard')")
    print(f"  Expected cost: ~15-20s on Modal vLLM, returns Bull/Bear debate JSON.")

    banner("BENCHMARK COMPLETE")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark cancelled.")
        sys.exit(0)
