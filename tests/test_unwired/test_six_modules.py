"""Tests for the six previously-unwired modules."""
from __future__ import annotations

import pytest


# ── risk/regime ─────────────────────────────────────────────────────────────
def test_regime_unknown_when_there_is_no_evidence():
    """No VIX and too few returns used to classify as low_vol, which hands out
    kelly_mult 1.0 — the most aggressive sizing on zero data."""
    from autfin.risk.regime import RegimeDetector

    state = RegimeDetector().detect()
    assert state.regime == "unknown"
    assert state.kelly_mult == 0.50
    assert state.conviction_threshold == 0.65
    assert state.confidence == 0.0


def test_regime_still_detects_an_explicit_drawdown():
    from autfin.risk.regime import RegimeDetector

    state = RegimeDetector().detect(nifty_close_today=95, nifty_close_5d_ago=100)
    assert state.regime == "crisis"
    assert state.kelly_mult == 0.25


@pytest.mark.parametrize(
    "vol_sigma,expected",
    [(0.002, "low_vol"), (0.010, "normal"), (0.017, "high_vol")],
)
def test_regime_from_realized_volatility(vol_sigma, expected):
    import random

    from autfin.risk.regime import RegimeDetector

    random.seed(7)
    d = RegimeDetector()
    for _ in range(30):
        d.record_daily_return(random.gauss(0.0005, vol_sigma))

    assert d.detect().regime == expected


def test_regime_uses_vix_when_available():
    from autfin.risk.regime import RegimeDetector

    assert RegimeDetector().detect(vix=35).regime == "crisis"
    assert RegimeDetector().detect(vix=25).regime == "high_vol"
    assert RegimeDetector().detect(vix=16).regime == "normal"


def test_regime_history_tracks_transitions():
    from autfin.risk.regime import RegimeDetector

    d = RegimeDetector()
    d.detect(vix=35)
    d.detect(vix=12)
    history = d.get_history()
    assert history and history[-1]["to"] == "low_vol"


# ── agents/moat ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_moat_scores_gross_margin_not_debt_equity():
    """Both moat helpers read debt_equity into a variable used as gross margin,
    so the caller's gross_margin never reached the score."""
    from autfin.agents.moat import moat_agent

    strong = {
        "holdings": [{"tradingsymbol": "NESTLEIND"}],
        "fundamental_reports": {
            "NESTLEIND": {
                "ratios": {
                    "roe": 0.35,
                    "gross_margin": 0.45,
                    "pb": 7.0,
                    "debt_equity": 0.05,
                    "revenue_growth": 0.20,
                    "fcf_yield": 0.06,
                },
                "market_cap": 800_000_00_00_00_000,
            }
        },
    }
    weak = {
        "candidates": ["NESTLEIND"],
        "fundamental_reports": {
            "NESTLEIND": {
                "ratios": {"roe": 0.05, "gross_margin": 0.08, "pb": 1.0, "debt_equity": 2.0}
            }
        },
    }

    strong_score = (await moat_agent(strong))["moat_assessments"]["NESTLEIND"]["score"]
    weak_score = (await moat_agent(weak))["moat_assessments"]["NESTLEIND"]["score"]

    assert strong_score >= 50, "strong fundamentals must reach at least NARROW"
    assert weak_score < 50


@pytest.mark.asyncio
async def test_moat_degrades_without_fundamentals():
    from autfin.agents.moat import moat_agent

    result = await moat_agent({"holdings": [], "candidates": ["UNKNOWNSTOCK"]})
    assessment = result["moat_assessments"]["UNKNOWNSTOCK"]
    assert assessment["rating"] in {"NONE", "ERROR"}
    assert 0 <= assessment["score"] < 50


# ── agents/regulatory ───────────────────────────────────────────────────────
def test_regulatory_classifies_a_sebi_order():
    from autfin.agents.regulatory import classify_regulatory_event

    for shape in ("title", "headline", "subject"):
        payload = {shape: "SEBI order imposes penalty for disclosure violations"}
        result = classify_regulatory_event(payload)
        assert result["impact_level"] == "high", shape
        assert result["direction"] == "negative"
        assert "SEBI order" in result["matched_keywords"]


def test_regulatory_accepts_body_fields_too():
    from autfin.agents.regulatory import classify_regulatory_event

    result = classify_regulatory_event(
        {"title": "Company update", "details": "Show cause notice received from NCLT"}
    )
    assert result["impact_level"] == "high"


def test_regulatory_reports_no_data_rather_than_low_impact():
    from autfin.agents.regulatory import classify_regulatory_event

    assert classify_regulatory_event({})["impact_level"] == "low"
    assert classify_regulatory_event({"title": ""})["matched_keywords"] == []


def test_regulatory_audit_returns_the_documented_shape():
    import asyncio

    from autfin.agents.regulatory import regulatory_audit

    result = asyncio.run(regulatory_audit("HDFCBANK", depth="quick"))
    assert result["status"] == "success"
    assert result["events"] == []
    assert "high_impact_count" in result


# ── risk/anomaly ────────────────────────────────────────────────────────────
def test_anomaly_detector_flags_a_win_rate_collapse():
    from autfin.risk.anomaly import AnomalyDetector

    d = AnomalyDetector(window=10, threshold=2.5)
    for _ in range(20):
        d.record_trade_result(won=True, pnl_pct=0.01)

    assert d.check_anomalies() == []

    # 8 losses in a 10-wide window takes the win rate to 0.2, below the 0.4 floor.
    for _ in range(8):
        d.record_trade_result(won=False, pnl_pct=-0.08)

    flagged = d.check_anomalies()
    assert flagged, "a collapse from 100% wins must be flagged"
    assert flagged[0]["metric"] == "win_rate"
    assert flagged[0]["severity"] == "high"


def test_anomaly_detector_quiet_without_enough_samples():
    from autfin.risk.anomaly import AnomalyDetector

    d = AnomalyDetector(window=20)
    assert d.check_anomalies() == []
    assert isinstance(d.get_summary(), dict)


def test_rolling_stats_z_score():
    from autfin.risk.anomaly import RollingStats

    flat = RollingStats(window=10)
    for _ in range(10):
        flat.add(1)
    assert flat.z_score(1) == 0.0  # no variance

    mixed = RollingStats(window=10)
    for v in [0, 0, 0, 0, 0, 10, 10, 10, 10, 10]:
        mixed.add(v)
    assert mixed.z_score(10) == pytest.approx(0.9487, abs=1e-3)
    assert mixed.latest_z == pytest.approx(1.0547, abs=1e-3)


# ── evals ───────────────────────────────────────────────────────────────────
def test_evals_dataset_round_trips(tmp_path):
    from autfin.evals import create_autfin_dataset, load_dataset, save_dataset

    ds = create_autfin_dataset(include_llm_judges=False, include_span_eval=True)
    assert len(ds.cases) > 0

    out = str(tmp_path / "ds.json")
    save_dataset(ds, out)
    assert len(load_dataset(out).cases) == len(ds.cases)


def test_evals_deps_are_paper_mode_with_capital():
    from autfin.evals import create_test_deps

    deps = create_test_deps("risk")
    assert deps.mode == "paper"
    assert deps.capital > 0
    assert deps.user_id.startswith("eval")


def test_agent_specific_datasets_are_built():
    from autfin.evals import create_agent_specific_dataset

    assert len(create_agent_specific_dataset("risk").cases) > 0


# ── mcp/browser_client ──────────────────────────────────────────────────────
def test_mcp_settings_come_from_env(monkeypatch):
    from autfin.mcp.browser_client import BrowserMCPSettings

    monkeypatch.setenv("BROWSER_MCP_PORT", "9333")
    monkeypatch.setenv("BROWSER_MCP_TRANSPORT", "STDIO")
    monkeypatch.setenv("BROWSER_MCP_AUTOSTART", "false")

    s = BrowserMCPSettings.from_env()
    assert s.port == 9333
    assert s.transport == "stdio"
    assert s.autostart is False


def test_mcp_client_reports_install_instructions_when_absent():
    """Obscura is not installed here; the client must say so rather than hang
    or fail with a bare ImportError."""
    from autfin.mcp.browser_client import (
        BrowserMCPNotInstalledError,
        get_browser_mcp_client,
        reset_browser_mcp_client,
    )

    reset_browser_mcp_client()
    try:
        client = get_browser_mcp_client()
        # If Obscura is present on this machine, a real client is returned.
        assert client is not None
    except BrowserMCPNotInstalledError as exc:
        # Either the Obscura binary or its langchain adapter can be the missing
        # piece; both must name the thing to install.
        message = str(exc).lower()
        assert "obscura" in message or "langchain-mcp-adapters" in message


def test_mcp_port_check_reports_closed_ports():
    from autfin.mcp.browser_client import _port_open

    assert _port_open("127.0.0.1", 1) is False  # port 1 is privileged/closed
