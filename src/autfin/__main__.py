"""python -m autfin — AutFin entry point with Logfire instrumentation."""
from __future__ import annotations

import logfire

logfire.configure(service_name="autfin")
logfire.instrument_pydantic_ai()

try:
    logfire.instrument_httpx(capture_all=True)
except Exception:
    pass

try:
    logfire.instrument_sqlite3()
except Exception:
    pass
