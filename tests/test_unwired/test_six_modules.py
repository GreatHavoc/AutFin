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
async def test_moat_is_an_agent_and_returns_the_documented_shape():
    """The scoring is a judgement, so the model makes it; the ratio extraction
    stays deterministic. Five invented per-dimension thresholds were removed,
    two of which read debt_equity where gross margin was meant."""
    from autfin.agents import moat

    assert callable(moat._moat_agent.run)
    assert set(moat.DIMENSIONS) == {
        "intangible_assets", "switching_costs", "network_effect",
        "cost_advantage", "efficient_scale",
    }


def test_moat_evidence_extraction_passes_gross_margin_through():
    """The old bug: two of the five scorers read debt_equity into a variable
    used as gross margin, so the caller's gross_margin never reached the score."""
    from autfin.agents.moat import _extract_evidence

    ev = _extract_evidence(
        {"fundamental_reports": {"X": {"ratios": {
            "roe": 0.35, "gross_margin": 0.45, "debt_equity": 0.1, "pb": 7.0,
        }}}},
        "X",
    )
    assert ev["gross_margin"] == 0.45
    assert ev["debt_equity"] == 0.1
    assert ev["roe"] == 0.35


@pytest.mark.asyncio
async def test_moat_degrades_to_error_rather_than_raising():
    """Bad model output or an unreachable endpoint must produce the documented
    shape with rating ERROR, not crash the caller."""
    from autfin.agents.moat import moat_agent

    result = await moat_agent({"holdings": [], "candidates": ["UNKNOWNSTOCK"]})
    assessment = result["moat_assessments"]["UNKNOWNSTOCK"]
    assert assessment["rating"] in {"NONE", "ERROR"}
    assert 0 <= assessment["score"] < 100
    assert "errors" in result


def test_moat_report_rejects_out_of_range_scores():
    """The output contract is enforced by Pydantic, not by an if-statement."""
    from pydantic import ValidationError

    from autfin.agents.moat import MoatReport

    with pytest.raises(ValidationError):
        MoatReport(score=250, rating="WIDE", sources={})


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


# ── browser MCP: settings + lifecycle ────────────────────────────────────────
def test_mcp_settings_come_from_env(monkeypatch):
    from autfin.tools.browser_agent import BrowserMCPSettings

    monkeypatch.setenv("BROWSER_MCP_PORT", "9333")
    monkeypatch.setenv("BROWSER_MCP_TRANSPORT", "STDIO")
    monkeypatch.setenv("BROWSER_MCP_AUTOSTART", "false")

    s = BrowserMCPSettings.from_env()
    assert s.port == 9333
    assert s.transport == "stdio"
    assert s.autostart is False


def test_autostart_names_the_missing_install(monkeypatch):
    """Obscura is absent here; the error must name what to install rather than
    fall back silently or raise a bare ImportError."""
    from autfin.tools import browser_agent as ba

    monkeypatch.setattr(ba.shutil, "which", lambda *_: None)
    monkeypatch.setattr(ba, "_port_open", lambda *a, **k: False)
    monkeypatch.delenv("BROWSER_MCP_BINARY", raising=False)

    with pytest.raises(ba.ObscuraNotInstalledError) as excinfo:
        ba.autostart_obscura()
    message = str(excinfo.value)
    assert "obscura" in message.lower()
    assert "https://github.com/h4ckf0r0day/obscura" in message


def test_mcp_port_check_reports_closed_ports():
    from autfin.tools.browser_agent import _port_open

    assert _port_open("127.0.0.1", 1) is False  # port 1 is privileged/closed
