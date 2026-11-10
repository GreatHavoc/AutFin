"""Indian tax engine: STCG/LTCG classification, exemption limits, tax-loss and
gain harvesting, and XIRR.

State comes from the SQLite `trades` table (see ``autfin.memory.database``).
"""
from __future__ import annotations

from autfin.portfolio.tax_engine import (
    HOLDING_PERIOD_DAYS,
    LTCG_EXEMPTION_LIMIT,
    LTCG_TAX_RATE,
    STCG_TAX_RATE,
    TaxEngine,
)

__all__ = [
    "TaxEngine",
    "STCG_TAX_RATE",
    "LTCG_TAX_RATE",
    "LTCG_EXEMPTION_LIMIT",
    "HOLDING_PERIOD_DAYS",
]
