"""The risk agent's stop-loss tool imported a module that never existed."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_calculate_stop_loss_computes_an_atr_stop(mock_run_context):
    """`from autfin.agents.technical import get_atr` raised ModuleNotFoundError
    the moment the tool ran without an explicit atr — a function-local import,
    so importing the package never caught it."""
    from autfin.agents.risk import calculate_stop_loss

    result = await calculate_stop_loss(mock_run_context, "RELIANCE", 2500.0, "buy", atr=25.0)
    assert result["atr"] == 25.0
    assert result["stop_loss"] == max(2500.0 - 2.5 * 25.0, 2500.0 * 0.92)


@pytest.mark.asyncio
async def test_without_an_atr_it_falls_back_to_the_toolset(mock_run_context, fake_yf_history):
    """The fallback path calls the technical toolset's own get_atr; with the
    yfinance call patched this proves the call resolves instead of raising."""
    from unittest.mock import MagicMock, patch

    ticker = MagicMock()
    ticker.history.return_value = fake_yf_history
    with patch("autfin.tools.technical_toolset.yf.Ticker", return_value=ticker):
        from autfin.agents.risk import calculate_stop_loss

        result = await calculate_stop_loss(mock_run_context, "RELIANCE", 2500.0, "buy")

    assert result["atr"] > 0
    assert result["stop_loss"] == round(max(2500.0 - 2.5 * result["atr"], 2500.0 * 0.92), 2)
