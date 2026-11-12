# AGENTS.md — AutFin

Pydantic AI multi-agent trading system for NSE/BSE. Python 3.13 in `.venv`, managed by `uv` (lockfile committed).

## Versions — upgraded in `fcc62b3`

`pydantic-ai` / `pydantic-ai-slim` / `pydantic-evals` **2.68.0**, `pydantic-ai-harness` **0.68.0**.

- The migration checklist (`pydantic-ai-migration.html`) was written against 2.37.0 and re-verified after the bump. Section C lists what the upgrade unlocked: `Caching`, `ModelFallback`, `LocalWorkspace`, `conversation_store`, `workspaces`, `pydantic_graph`, plus the harness safety layer (`DetectLoops`, `AskUser`, `Permissions`, `ApprovalReviewer`, `ToolCallJudge`, `TrajectoryJudge`, `ArtifactScanning`, `SmartSearch`).
- Capability extras installed as base deps (`uv sync` covers them): `researcher`, `playwright` (+Chromium), `ipython`, `coder`, `repair-tool-arguments`, `prompt-injection-defender`, `temporal`. `browser-use` is **not installable** (pins `pydantic==2.12.5`, conflicts with pydantic-ai-slim 2.68.0); `exa`/`youdotcom` need paid keys.
- `FileSystem`/workspace capabilities are **dropped entirely**: 2.68 requires an attached workspace, `LocalWorkspace` is POSIX-only, and `DockerSandbox` hard-codes that same POSIX subprocess runner on the client — so there is no supported local workspace on this Windows box. No agent exposes file/workspace/shell tools, so nothing was lost. Podman 5.8.2 works for containers generally; it is the *client-side* runner that blocks it.
- `pytest`, `pytest-asyncio`, `ruff` are now explicit `[dependency-groups] dev` deps: they used to arrive transitively via a pydantic-ai extra, and `uv sync` dropped them when that extra changed — which made `uv run pytest` fall back to a system interpreter with no `pydantic_ai`.

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
- `tests/test_tools/test_technical_toolset.py` calls live Yahoo on purpose ("no mocks"); its last close can come back `nan` when Yahoo rate-limits, which fails `test_sma_reliance` on `result.iloc[-1] > 0`. Re-run before blaming a change.

## Housekeeping for every code change

**`autfin-architecture.html` (repo root) is the living architecture page and must be updated in the same commit as the change it describes.** Edit the affected section only — it is a single self-contained file (no build, no CDN), so a stale number there is misinformation an agent will act on. Sections 1–7: runtime flow, agent grid, data sources, learning tools, model routing, live-vs-legacy, verification commands. Same rule for `AGENTS.md`.

## The lint/type baseline is already red — do not "fix" it repo-wide

`ruff check .` = 85 errors, `pyright` = ~400 errors. Fix only files you touch. One of those is a genuine bug, not style: `agents/base.py` uses PEP 695 generics (`def create_agent[OutputT]`), i.e. needs ≥3.12 even though `requires-python = ">=3.11"` and `ruff target-version = "py311"`. Ruff reads it as invalid syntax; don't "fix" by de-genericizing.

## What is actually wired

Entry graph: `agents/supervisor.py` → `create_agent()` in `agents/base.py` → sub-agents `market_data, analysis, research, risk, debate, conference, memory` (+ `quant`). Toolsets (from `tools/__init__.py`): `market_data`→kite, `analyze`→fundamental/technical/sentiment, `research`→screening/fundamental/browser, `risk`→kite/technical/**learning**, `memory`→**learning**, `supervisor`→**learning**; `debate` and `conference` are tool-less. Shared state is `types.AgentDeps` (deps), structured outputs and model IDs are in `types.py`. Both backtest suites work now: `src/autfin/backtest/` (vectorbt) and `src/autfin/backtesting/` (walk-forward + alpha decay), exported together from `autfin.backtest`. `src/autfin/risk/` (kelly, vol_targeting) is imported lazily from `agents/risk.py`. `autfin/learning/` (calibration, retrieval, online learner) is fixed and covered by `tests/test_learning/`; `tools/learning_toolset.py` exposes it as agent tools attached to `supervisor`, `risk` and `memory`, so the LLM decides when to pull calibration — `execute_trade` is the only supervisor tool with side effects left; the imperative `run_morning_routine` fan-out was deleted in favour of the already-attached `DynamicWorkflow` capability. Two deterministic pre-work modules are fixed but uninvoked: `autfin/universe/` (WatchlistCurator) and `autfin/portfolio/` (tax engine).

Dead / legacy — unwired because no live replacement exists, do not extend, do not assume it runs:
- `agents/moat.py`, `agents/regulatory.py` (SEBI), `risk/anomaly.py`, `risk/regime.py`, `evals/` — all fixed and covered by `tests/test_unwired/`, still called by nothing in the agent graph. the browser MCP layer lives in `tools/browser_agent.py` (the old `mcp/` package was deleted).
- `llm/`, `observability/`, `config/settings.py` — unwired, but they run and are tested in `tests/test_memory/`. Model names now come from `types.MODEL_CONFIG` everywhere; `memory/{retrieval,consolidation}.py` were deleted (remnants of a removed store, zero importers).
- `mcp/` was deleted; its `BrowserMCPSettings` + Obscura lifecycle now live in `tools/browser_agent.py`.
- `agents/guardrails.py` **is enforced now**: `create_agent()` in `agents/base.py` appends `get_guardrails_for_agent(name)` to every agent's capabilities, so each agent carries the set its name maps to. The tool guard also runs `stay_in_workspace` alongside `validate_trade_args`.
- `create_agent()` also attaches `Instrumentation()` (default settings: the global tracer provider, no-op until one is configured) so spans come out of every entrypoint, not just `run.py`'s `logfire.instrument_pydantic_ai()` call. Note a test in `tests/test_guardrails/` imports `run.py`, which calls `logfire.configure()` at import — after that a *global* `TracerProvider` override is refused, so span assertions use logfire's own `capfire` fixture.
- The memory agent carries `ConversationSearch` over the same step store as the supervisor's `StepPersistence` (`scope="conversation"`), so it can recall earlier turns of the daemon's thread with BM25. That store is shared through `agents/persistence.py::get_step_store()`; giving a capability its own copy of the store is the failure mode there (the spend limits comment in `base.py` documents the same shape).
- Conversations persist in SQLite via `src/autfin/conversation/` (a `ConversationStore` on the existing
  `data/autfin.db`): `run.py`'s morning routine loads and saves `daemon:morning`, so usage and
  `conversation_id` carry across daily runs instead of every run starting cold. Dataset serialisation is the
  framework's (`Dataset.to_file`/`from_file`, with `custom_evaluator_types` — the old hand-rolled saver
  flattened evaluators to strings).
- Crash recovery for the supervisor run is the harness's `StepPersistence` (`src/autfin/agents/persistence.py`, a `SqliteStepStore` on the same `data/autfin.db`): it snapshots the message history at every settled step, appends a step-event trail, and keeps a tool-effect ledger, so after a crash `continue_run`/`fork_run` resume from the last settled step and a tool still marked `started` reads as `unknown_after_crash`. The dead `runs` table plus `record_run_start()`/`record_run_end()` that preceded it are gone — they had zero callers and recorded only start/end. Only the supervisor carries the capability; sub-agents would just fill the store with noise.
- Evals: `evals/dataset.py` keeps only the deterministic evaluators (field presence, thresholds, numeric checks). The judgemental ones moved to the framework — `LLMJudge` rubrics and nine `HasMatchingSpan` evaluators that were sitting unused.
- `agents/moat.py` is now an agent: the five moat sources are a judgement, so the model scores them against a Pydantic-enforced contract, with ratio extraction kept deterministic. `agents/distortion.py` (volume-ratio arithmetic) and `agents/regulatory.py` (controlled-vocabulary keywords) are deliberately left deterministic — a judge would be worse at both.
- Removed in the cleanup commit: `agents/{agent_tools,tools,conferences,memory_system,schemas,peer_compare}.py`, `tools/{supervisor_tools,fundamental_wrapper,technical_wrapper,memory_tools}.py`, `state/`. If you see a stale import of one, it was already dead.

## The learning journal and its database

- `memory/database.py` owns `data/autfin.db` (gitignored, local-only). `get_connection()` runs `PRAGMA quick_check` and **quarantines a corrupt file as `data/autfin.db.malformed-<ts>`** instead of handing it out — every learning function swallows `Exception`, so without that check the engine silently returns empty.
- If a `.malformed-*` file exists, that trade history needs `sqlite3 .recover`; the journal restarted empty.
- `pre_trade_context()` / `post_trade_learn()` in `autfin/learning/feedback_loop.py` are the entry points for trading context (similar setups, calibration, patterns, conviction adjustment). **`tools/learning_toolset.py` exposes them to agents as `get_calibration`, `get_learning_context`, `get_learning_summary`, `calibrate_conviction` and `record_trade_outcome`, and that toolset is attached to `supervisor`, `risk` and `memory`.** The LLM decides when to pull context.
- `retrieve_learnings()` claims semantic search but `sentence-transformers` is not a dependency, so it silently uses sha256 "hash embeddings" — similarity scores are noise, not meaning.

## Traps

- Two import styles coexist: `autfin.*` and `src.autfin.*`. The `src.` form resolves only with the repo root on `sys.path`; it breaks anywhere else. New code: `autfin.*`.
- LLM calls go through `create_nvidia_model()` → `OpenAIChatModel` against `https://integrate.api.nvidia.com/v1` with `NVIDIA_NIM_API_KEY` (pydantic-ai has no native NIM provider). Model IDs get swapped frequently because NIM rejects names/patterns (e.g. xgrammar chokes on `Decimal`); verify with a real call before "upgrading" a model string.
- Prices/quantities are `float` on purpose — don't reintroduce `Decimal`.
- `Memory(FileStore("./data/memory"))` is CWD-relative: run from the repo root or state lands elsewhere. Tests mutate the **tracked** `data/memory/.memory-store.sqlite3`; don't commit that diff.
- Browser MCP: `tools/browser_agent.py` owns the Obscura settings and lifecycle (`autostart_obscura`, `managed_obscura`, `ensure_obscura_running`), and `run.py` delegates to them after writing `BROWSER_MCP_PORT`. A named `OBSCURA_MCP_URL` still overrides the endpoint. `get_browser_toolset()` is called at import time (research_agent), so autostart failures are logged, not raised — otherwise the whole package fails to import on a machine without Obscura. Set `BROWSER_MCP_PORT` (or `BROWSER_MCP_ENDPOINT`) and both sides agree. `get_browser_agent()` still hardcodes `openai:gpt-4.1-mini`, needing `OPENAI_API_KEY`, which `.env.example` does not list.
- `run.py` hardcodes a Windows Obscura path (`OBSCURA_DIR` default `C:\Users\Prakhar\Downloads\...`) and installs SIGTERM/SIGINT handlers at import. It syncs `BROWSER_MCP_PORT` to the port it starts Obscura on and delegates the spawn to `tools/browser_agent.autostart_obscura`, so daemon and agent path run the same server the same way.
