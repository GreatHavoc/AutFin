"""Investable universe: symbol list, sector map, bulk fetcher, watchlist curation.

The watchlist runs BEFORE the supervisor so the agent graph starts from a
deterministic candidate list instead of an LLM guess.
"""
from __future__ import annotations

from autfin.universe.curator import WatchlistCurator, curate_watchlist

__all__ = [
    "WatchlistCurator",
    "curate_watchlist",
]
