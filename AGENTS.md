# AGENTS.md — AutFin

Pydantic AI multi-agent trading system for NSE/BSE. Python 3.13 in `.venv`, managed by `uv` (lockfile committed).

## Commands

```bash
uv sync                                              # install
uv run pytest tests/ -q                              # full suite, ~4.5 min
uv run pytest tests/test_tools/test_technical_toolset.py -q   # single file, ~12 s
uv run ruff check <paths>                            # line 100, E/F/I
uv run pyright                                       # NOTE: no pyrightconfig.json exists
uv run python run.py --once                          # daemon; --mode paper|live|backtest
```

- **Always pass `tests/`.** Bare `uv run pytest` fails collection: root `test_validation.py` imports `validate_durable_execution`, deleted along with `agents/durable.py`. Root `test_*.py` / `demo_sentiment.py` are ad-hoc scripts that hit live network at import.
- Full-suite time is dominated by sentiment tests loading FinBERT (transformers → HF cache). ~1.5 min warm, ~4.5 min cold.

## The lint/type baseline is already red — do not "fix" it repo-wide

`ruff check .` = 119 errors, `pyright` = ~400 errors. Fix only files you touch. One of those is a genuine bug, not style: `agents/base.py` uses PEP 695 generics (`def create_agent[OutputT]`), i.e. needs ≥3.12 even though `requires-python = ">=3.11"` and `ruff target-version = "py311"`. Ruff reads it as invalid syntax; don't "fix" by de-genericizing.

## What is actually wired

Entry graph: `agents/supervisor.py` → `create_agent()` in `agents/base.py` → sub-agents `market_data, analysis, research, risk, debate, conference, memory` (+ `quant`). Only market_data/analysis/research/risk/quant get a toolset (from `tools/__init__.py`); debate/conference/memory are tool-less. Shared state is `types.AgentDeps` (deps), structured outputs and model IDs are in `types.py`. Live backtesting is `src/autfin/backtest/`; `src/autfin/risk/` (kelly, vol_targeting) is imported lazily from `agents/risk.py`.

Dead / legacy — read-only, do not extend, do not assume it runs. These are kept only because they are the **sole implementation** of a feature (walk-forward backtests, conviction calibration, tax lots), not because they work:
- `backtesting/` (walk-forward + alpha decay — `backtest/` has **no** walk-forward), `learning/` (calibration, online learning, A/B tests), `portfolio/` (tax engine), `universe/` + `tools/universe.py` (sector map + bulk fetcher that `tools/screening.py` lacks).
- `memory/` (SQLite trade history + Chroma vectors), `llm/`, `observability/`, `config/settings.py` — all kept only because `learning/` and `memory/` import them. Nothing in the agent graph reads `config/settings.py`; models come from `types.ModelConfig`.
- `agents/moat.py`, `agents/regulatory.py` (SEBI checks), `risk/anomaly.py`, `risk/regime.py`, `evals/`, `validation.py`, `mcp/browser_client.py` — unwired, no live replacement.
- `agents/guardrails.py` is exported but attached to **no** agent; guardrails are not enforced despite what the README says.
- Removed in the cleanup commit: `agents/{agent_tools,tools,conferences,memory_system,schemas,peer_compare}.py`, `tools/{supervisor_tools,fundamental_wrapper,technical_wrapper,memory_tools}.py`, `state/`. If you see a stale import of one, it was already dead.

## Traps

- Two import styles coexist: `autfin.*` and `src.autfin.*`. The `src.` form resolves only with the repo root on `sys.path`; it breaks anywhere else. New code: `autfin.*`.
- LLM calls go through `create_nvidia_model()` → `OpenAIChatModel` against `https://integrate.api.nvidia.com/v1` with `NVIDIA_NIM_API_KEY` (pydantic-ai has no native NIM provider). Model IDs get swapped frequently because NIM rejects names/patterns (e.g. xgrammar chokes on `Decimal`); verify with a real call before "upgrading" a model string.
- Prices/quantities are `float` on purpose — don't reintroduce `Decimal`.
- `Memory(FileStore("./data/memory"))` is CWD-relative: run from the repo root or state lands elsewhere. Tests mutate the **tracked** `data/memory/.memory-store.sqlite3`; don't commit that diff.
- Browser scraping has mismatched defaults: `tools/browser_agent.py` connects to `OBSCURA_MCP_URL` (default `http://localhost:9222/mcp`), while `mcp/browser_client.py` and `.env.example` use `BROWSER_MCP_PORT` (8080 / 9333). They must be aligned by hand. `get_browser_agent()` also hardcodes `openai:gpt-4.1-mini`, needing `OPENAI_API_KEY`, which `.env.example` does not list.
- `run.py` hardcodes a Windows Obscura path (`OBSCURA_DIR` default `C:\Users\Prakhar\Downloads\...`) and installs SIGTERM/SIGINT handlers at import.
