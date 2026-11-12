"""Spend ceilings for the agent graph (B2).

Pydantic AI's `UsageLimits` caps one run; it does not cover a day, a per-tenant
share, or a counter shared by several workers. A morning routine fans out to seven
sub-agents that each retry, so a runaway loop has nothing between it and the
quota.

Two things that are easy to get wrong:
  - the default `InMemorySpendStore` is built *per capability*, so one
    `SpendLimits` per agent would give every agent its own daily ceiling — nine
    agents means nine times the budget. One shared instance is the point.
  - a token-only budget does not refuse in this harness version, so the USD
    ceiling is what enforces; `price` supplies the amount because genai-prices
    has no entry for the NIM model ids.
"""
from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget

logger = logging.getLogger(__name__)

# USD ceilings. Env-overridable because the right number depends on the account
# and on which NIM model each tier resolves to that month.
DAILY_USD = Decimal(os.getenv("AUTFIN_SPEND_USD_DAY", "5"))
RUN_USD = Decimal(os.getenv("AUTFIN_SPEND_USD_RUN", "1"))
WARN_AT = float(os.getenv("AUTFIN_SPEND_WARN_AT", "0.8"))

# Charged per model request. genai-prices has no entry for the NIM model ids, so
# without this every response reads as unpriced and a USD ceiling never trips.
PER_REQUEST_USD = Decimal(os.getenv("AUTFIN_SPEND_USD_REQUEST", "0.002"))

# Per-tenant partitions: each `user_id` gets its own ceiling rather than one
# shared pool, which is what a single shared counter would otherwise give.
TENANT_SCOPE = lambda ctx: ctx.deps.user_id  # noqa: E731


def _price(_response: Any) -> Decimal:
    return PER_REQUEST_USD


_limits: SpendLimits | None = None


def get_spend_limits() -> SpendLimits:
    """The one shared spend limiter for the whole process."""
    global _limits
    if _limits is None:
        _limits = SpendLimits(
            budgets=[
                Budget(usd=RUN_USD, window="run", name="autfin-run"),
                Budget(usd=DAILY_USD, window="day", name="autfin-day"),
                Budget(
                    usd=DAILY_USD,
                    window="day",
                    name="autfin-tenant",
                    scope=TENANT_SCOPE,
                    warn_at=WARN_AT,
                ),
            ],
            price=_price,
        )
        logger.info(
            "Spend ceilings: run=$%s day=$%s per-tenant=$%s @ %s/request",
            RUN_USD, DAILY_USD, DAILY_USD, PER_REQUEST_USD,
        )
    return _limits
