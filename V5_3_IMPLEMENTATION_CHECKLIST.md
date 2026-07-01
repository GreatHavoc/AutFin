# v5.3 Implementation Checklist

> Step-by-step implementation of LangGraph Subagents pattern

---

## Step 1: Create `src/autfin/agents/tools.py`

**Goal:** Agent tool wrappers — each wraps an agent as a `@tool` that makes its own LLM call

**Why:** Context isolation — each agent gets clean context window, not one big blob

**Create:**
- `fundamentals_tool(symbol, depth)` — calls fundamentals agent with own LLM
- `technical_tool(symbol, depth)` — calls technical agent with own LLM
- `sentiment_tool(symbol)` — calls sentiment agent with own LLM
- `macro_tool()` — calls global macro agent with own LLM
- `research_tool(symbols, depth)` — calls researcher agent with own LLM
- `risk_tool(trade)` — calls risk manager with own LLM
- `portfolio_tool()` — calls portfolio manager with own LLM

**Each tool:**
- Lazy imports inside function body
- Makes its own `llm.generate()` call with agent-specific system prompt
- Returns structured result (text or JSON)
- `SUPERVISOR_TOOLS` list exported for supervisor to use

---

## Step 2: Create `src/autfin/agents/schemas.py`

**Goal:** JSON response schemas for structured output from each agent

**Why:** Predictable JSON from each subagent, easier synthesis by supervisor

**Create schemas:**
- `FUNDAMENTALS_SCHEMA` — piotroski_score, magic_formula_rank, dcf_value, financial_health, conviction
- `TECHNICAL_SCHEMA` — signal, conviction, indicators, patterns
- `SENTIMENT_SCHEMA` — score, sources, catalysts
- `MACRO_SCHEMA` — impact_signal, sector_matrix, narrative
- `RESEARCH_SCHEMA` — bull_case, bear_case, synthesis, rating, conviction
- `RISK_SCHEMA` — verdict, checks, approved_size, rationale
- `PORTFOLIO_SCHEMA` — trades, allocation, rationale

---

## Step 3: Update `src/autfin/graph/supervisor.py`

**Goal:** Accept `store` parameter, update system prompt for agent tools

**Changes:**
- `compile_supervisor_graph(checkpointer, store)` — accept store param
- Update `_build_context_prompt()` — describe agent tools instead of 20 raw tools
- Keep backward compatibility — old tools still work if agent tools not available
- `build_supervisor_graph()` — no changes needed (same 2 nodes)

**System prompt update:**
```
AVAILABLE AGENT TOOLS:
- fundamentals_analysis(symbol, depth) — Piotroski, Magic Formula, DCF
- technical_analysis(symbol, depth) — RSI, MACD, support/resistance
- sentiment_analysis(symbol) — news, social, analyst sentiment
- macro_analysis() — global macro impact on India
- research_synthesis(symbols, depth) — Bull/Bear debate
- risk_assessment(trade) — 14 risk checks with veto power
- portfolio_allocation() — capital allocation, trade prioritization
Call multiple agents in parallel when possible.
```

---

## Step 4: Create `src/autfin/graph/builder.py`

**Goal:** Extract graph compilation into a builder that accepts checkpointer + store

**Why:** Clean separation, store gets passed to graph compilation

**Contents:**
- `build_graph(checkpointer=None, store=None)` — builds and compiles graph
- Attaches observability handler
- Returns compiled graph

---

## Step 5: Update `src/autfin/llm/client.py`

**Goal:** Add `_execute_tools_parallel()` for agent tools (each = own LLM call)

**Why:** Agent tools need to run in parallel when supervisor calls multiple

**Changes:**
- `_execute_tools_parallel(tool_calls, tool_map, state)` — execute multiple tool calls via `asyncio.gather`
- Already exists in `generate_with_tools()` via `asyncio.gather` — verify it works with agent tools
- Add structured output support via `response_format={"type": "json_object"}`

---

## Step 6: Update `src/autfin/cli/main.py`

**Goal:** Initialize store, pass to graph compilation via builder

**Changes:**
- Import `create_store()` from memory setup
- Pass store to `build_graph(checkpointer, store)`
- Ensure store is created within running event loop (async init)

---

## Step 7: Update `pyproject.toml`

**Goal:** Add langgraph-store dependency if needed

**Check:** Is `langgraph-store` needed separately or included in `langgraph>=1.0.0`?
- If separate: add `langgraph-store>=0.1.0`
- If included: no change needed

---

## Step 8: Create `tests/test_graphs/test_v5_3_integration.py`

**Goal:** Tests for agent tools, structured output, parallel execution, store integration

**Tests:**
- `test_supervisor_calls_fundamentals_tool()` — verify tool call works
- `test_parallel_agent_execution()` — verify multiple agents run in parallel
- `test_structured_output_schemas()` — verify JSON schemas are valid
- `test_store_initialization()` — verify store gets created and passed
- `test_backward_compatibility()` — old tools still work alongside new

---

## Step 9: Run tests + verify

**Commands:**
```bash
.\.venv\Scripts\python.exe -m pytest tests/ -x -q
.\.venv\Scripts\python.exe -m pytest tests/test_graphs/test_v5_3_integration.py -v
```

**Verify:**
- All existing tests pass
- New v5.3 tests pass
- LangSmith traces show individual agent calls (not one blob)
- `langgraph dev` shows agent tools in Studio

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `src/autfin/agents/tools.py` | CREATE | Agent tool wrappers (7 tools) |
| `src/autfin/agents/schemas.py` | CREATE | Response schemas for structured output |
| `src/autfin/graph/builder.py` | CREATE | Graph builder with store support |
| `src/autfin/graph/supervisor.py` | MODIFY | Accept store, update system prompt |
| `src/autfin/llm/client.py` | MODIFY | Verify parallel execution, add structured output |
| `src/autfin/cli/main.py` | MODIFY | Initialize store, pass to graph |
| `pyproject.toml` | MODIFY | Add langgraph-store if needed |
| `tests/test_graphs/test_v5_3_integration.py` | CREATE | New tests for v5.3 |

**Total:** 4 new files + 4 modified files

---

## Success Criteria

1. ✅ LangSmith shows individual agent traces (not one blob)
2. ✅ Each agent returns structured JSON
3. ✅ Supervisor can call multiple agents in parallel
4. ✅ Long-term memory persists across sessions (Store)
5. ✅ All existing tests pass
6. ✅ New v5.3 tests pass
7. ✅ No performance regression
