"""Tests for v4.1 macro indicator additions (India 10Y, new global indices, Bank Nifty regime)."""

import pytest
from unittest.mock import patch, MagicMock


# ===========================================================================
# 4.1.1 — India 10Y G-Sec via FRED
# ===========================================================================

class TestIndia10YFRED:
    """Tests for India 10Y bond yield via FRED API."""

    def test_fred_series_includes_india_10y(self):
        from src.autfin.agents.global_macro import FRED_SERIES
        assert "india_10y_yield" in FRED_SERIES
        assert FRED_SERIES["india_10y_yield"] == "INDIRLTLT01STM"

    def test_fred_labels_include_india_10y(self):
        from src.autfin.agents.global_macro import fetch_fred_indicators
        import src.autfin.agents.global_macro as gm
        # Verify the labels dict has india_10y_yield
        # We can't easily test the internal labels dict, but we can verify
        # the FRED_SERIES constant is correct
        assert "india_10y_yield" in gm.FRED_SERIES

    @patch("src.autfin.agents.global_macro._fetch_fred_indicator")
    def test_fetch_fred_includes_india_10y(self, mock_fetch):
        from src.autfin.agents.global_macro import fetch_fred_indicators
        mock_fetch.side_effect = lambda sid: {
            "us_gdp": 2.1,
            "us_cpi": 300.0,
            "fed_funds_rate": 5.25,
            "india_10y_yield": 6.78,
        }.get(sid)
        result = fetch_fred_indicators()
        assert "india_10y_yield" in result
        assert result["india_10y_yield"]["value"] == 6.78
        assert result["india_10y_yield"]["unit"] == "percent"

    @patch("src.autfin.agents.global_macro._fetch_fred_indicator")
    def test_fetch_fred_india_10y_unavailable(self, mock_fetch):
        from src.autfin.agents.global_macro import fetch_fred_indicators
        mock_fetch.side_effect = lambda sid: {
            "us_gdp": 2.1,
            "us_cpi": 300.0,
            "fed_funds_rate": 5.25,
        }.get(sid)  # india_10y_yield not in dict → returns None
        result = fetch_fred_indicators()
        assert result["india_10y_yield"]["value"] is None


# ===========================================================================
# 4.1.2 — 5 new global indices
# ===========================================================================

class TestNewGlobalIndices:
    """Tests for Shanghai, Hang Seng, DAX, CAC 40, Euro Stoxx 50."""

    def test_sector_impact_matrix_has_all_new_indices(self):
        from src.autfin.agents.global_macro import SECTOR_IMPACT_MATRIX
        new_indices = [
            "shanghai_composite", "hang_seng", "dax", "cac_40", "euro_stoxx_50",
            "india_10y_yield",
        ]
        for idx in new_indices:
            assert idx in SECTOR_IMPACT_MATRIX, f"Missing: {idx}"
            assert "positive" in SECTOR_IMPACT_MATRIX[idx]
            assert "negative" in SECTOR_IMPACT_MATRIX[idx]
            assert "affected_sectors" in SECTOR_IMPACT_MATRIX[idx]["positive"]
            assert "reasoning" in SECTOR_IMPACT_MATRIX[idx]["positive"]

    def test_sector_impact_has_all_original_indices(self):
        from src.autfin.agents.global_macro import SECTOR_IMPACT_MATRIX
        original = [
            "crude_oil_wti", "inr_usd", "fed_funds_rate", "gold_futures",
            "vix", "dxy", "nifty_50", "us_10y_yield",
        ]
        for idx in original:
            assert idx in SECTOR_IMPACT_MATRIX, f"Missing original: {idx}"

    def test_sector_impact_total_count(self):
        from src.autfin.agents.global_macro import SECTOR_IMPACT_MATRIX
        # 8 original + 6 new = 14 total
        assert len(SECTOR_IMPACT_MATRIX) == 14

    def test_new_indices_sector_counts(self):
        from src.autfin.agents.global_macro import SECTOR_IMPACT_MATRIX
        # Verify new indices have reasonable sector counts
        assert len(SECTOR_IMPACT_MATRIX["shanghai_composite"]["positive"]["affected_sectors"]) >= 2
        assert len(SECTOR_IMPACT_MATRIX["india_10y_yield"]["positive"]["affected_sectors"]) >= 2
        assert len(SECTOR_IMPACT_MATRIX["dax"]["positive"]["affected_sectors"]) >= 2


# ===========================================================================
# 4.1.3 — Env var consistency (MARKETAUX_API_KEY)
# ===========================================================================

class TestEnvVarConsistency:
    """Tests that MARKETAUX_API_KEY is used consistently."""

    def test_sentiment_uses_api_key_not_token(self):
        import inspect
        from src.autfin.agents.sentiment import SentimentAgent
        source = inspect.getsource(SentimentAgent.__init__)
        assert "MARKETAUX_API_KEY" in source
        assert "MARKETAUX_API_TOKEN" not in source


# ===========================================================================
# 4.1.7 — Bank Nifty regime detection
# ===========================================================================

class TestBankNiftyRegime:
    """Tests for Bank Nifty SMA-based regime detection."""

    def test_detect_market_regime_returns_bank_nifty(self):
        """Verify detect_market_regime now returns bank_nifty_regime."""
        from src.autfin.tools.screening import detect_market_regime
        import asyncio
        result = asyncio.run(detect_market_regime())
        assert "bank_nifty_regime" in result
        assert result["bank_nifty_regime"] in ("bull", "bear", "neutral")

    def test_detect_market_regime_has_all_fields(self):
        from src.autfin.tools.screening import detect_market_regime
        import asyncio
        result = asyncio.run(detect_market_regime())
        expected_fields = ["regime", "nifty_price", "sma_50", "sma_200", "confidence", "bank_nifty_regime"]
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"


# ===========================================================================
# 4.1.8 — get_market_status returns new indices
# ===========================================================================

class TestMarketStatusNewIndices:
    """Tests that get_market_status returns Bank Nifty and global indices."""

    @pytest.mark.asyncio
    async def test_market_status_has_bank_nifty(self):
        from src.autfin.tools.supervisor_tools import get_market_status
        result = await get_market_status()
        assert "bank_nifty_level" in result
        assert "bank_nifty_change_pct" in result

    @pytest.mark.asyncio
    async def test_market_status_has_global_indices(self):
        from src.autfin.tools.supervisor_tools import get_market_status
        result = await get_market_status()
        assert "global_indices" in result
        # At minimum these should be present (may have 0.0 if fetch fails)
        expected = ["S&P 500", "NASDAQ", "Nikkei 225", "FTSE 100",
                     "Shanghai Composite", "Hang Seng", "DAX", "CAC 40",
                     "Euro Stoxx 50", "VIX", "USD Index", "USD/INR"]
        for name in expected:
            assert name in result["global_indices"], f"Missing: {name}"
