"""Offline tests: synthetic data only, no keys or network.  Run:  pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macronav import data as D  # noqa: E402
from macronav.assets import label, score_assets  # noqa: E402
from macronav.indicators import build_panel, recession_probability  # noqa: E402
from macronav.regime import compute_flags, risk_score, run_state_machine  # noqa: E402


@pytest.fixture(scope="module")
def base():
    raw = D.make_demo_data()
    universe = D.load_yaml("universe.yaml")
    th = D.load_yaml("thresholds.yaml")
    P = build_panel(raw, universe)
    return raw, universe, th, P


def calm(P):
    """Overwrite the last ~2 years with a quiet, disinflationary, rising market."""
    P = P.copy()
    w = P.index[-520:]
    P.loc[w, "core_pce_yoy"], P.loc[w, "core_cpi_3m_ann"], P.loc[w, "brent_yoy"] = 2.2, 2.3, 0.0
    P.loc[w, "ff_change_3m"], P.loc[w, "ff_change_6m"], P.loc[w, "ff_change_12m"] = 0.0, 0.0, 0.0
    P.loc[w, "two_y_minus_ff"], P.loc[w, "real_ff"] = -0.2, 1.0
    P.loc[w, ["sahm", "claims_vs_low", "philly", "permits_yoy"]] = [0.05, 0.03, 10.0, 5.0]
    P.loc[w, ["hy_bp", "hy_min_126", "hy_max_126", "hy_10d_ago", "nfci", "cp_ff_bp"]] = [320, 300, 340, 320, -0.5, 10]
    P.loc[w, ["vix", "vix_max_60", "vix_avg_5", "vix_avg_20", "vix_ratio", "vix_ratio_min_3d", "vix_days_above_25"]] = [14, 16, 14, 14, 0.85, 0.85, 0]
    P.loc[w, "trend_down"], P.loc[w, "drawdown_pct"], P.loc[w, "months_above_sma10"] = False, -2.0, 6
    P.loc[w, "curve_pattern"], P.loc[w, "kre_rel_20d"] = "bear steepener", 0.0
    P.loc[w, "spx"] = np.linspace(4000, 5000, len(w))
    return P


def run(P, th):
    F = compute_flags(P, th, {})
    return F, run_state_machine(P, F, th)


def test_calm_is_goldilocks(base):
    _, _, th, P = base
    _, R = run(calm(P), th)
    assert R.current == 1 and not R.amber


def test_hot_inflation_becomes_hawkish(base):
    _, _, th, P = base
    P = calm(P)
    P.loc[P.index[-130]:, "core_pce_yoy"] = 3.4
    _, R = run(P, th)
    assert R.current == 2


def test_crisis_triggers_breakdown_immediately(base):
    _, _, th, P = base
    P = calm(P)
    P.loc[P.index[-3]:, "vix"] = 45
    _, R = run(P, th)
    assert R.current == 3


def test_pivot_washout_healing_gives_recovery(base):
    _, _, th, P = base
    P = calm(P)
    crash, heal = P.index[-90], P.index[-20]
    P.loc[crash:, ["hy_bp", "hy_max_126", "hy_10d_ago"]] = [800, 800, 800]
    P.loc[crash:, "vix"], P.loc[crash:, "vix_max_60"], P.loc[crash:, "vix_avg_5"] = 50, 50, 50
    P.loc[heal:, "ff_change_3m"] = -0.75                 # pivot
    P.loc[heal:, "vix_avg_5"], P.loc[heal:, "vix"] = 30, 30  # release (30 <= 0.75 x 50)
    P.loc[heal:, "hy_bp"], P.loc[heal:, "hy_10d_ago"] = 600, 700  # healing
    _, R = run(P, th)
    assert R.current == 4
    assert 3 in R.weekly.regime.values


def test_risk_score_bounds(base):
    _, _, _, P = base
    S = risk_score(P)
    assert S["Total"].between(0, 100).all()
    assert set(np.unique(S.drop(columns="Total").to_numpy())) <= {0, 5, 10}


def test_asset_scores_bounded(base):
    raw, universe, th, P = base
    df = score_assets(raw.prices, universe, P, 2, True, 38.0)
    ok = df[df.status == "ok"]
    assert len(ok) > 30
    for col in ("short_score", "medium_score"):
        assert ok[col].dropna().between(-100, 100).all()
    assert label(50) == "Positive" and label(-50) == "Negative" and label(0) == "Neutral"


def test_recession_probability_shape():
    assert recession_probability(-1.0) > recession_probability(0.0) > recession_probability(2.0)
    assert abs(recession_probability(0.0) - 0.297) < 0.01


def test_pipeline_demo_snapshot(tmp_path, monkeypatch):
    from macronav import pipeline
    monkeypatch.setattr(pipeline, "SNAPSHOT", tmp_path / "s.json")
    monkeypatch.setattr(pipeline, "REPORT", tmp_path / "r.md")
    monkeypatch.setattr(pipeline, "HISTORY", tmp_path / "h.csv")
    snap = pipeline.build_snapshot(demo=True)
    pipeline.write_outputs(snap)
    loaded = json.loads((tmp_path / "s.json").read_text())
    for key in ("regime", "score", "narrative", "assets", "history", "readings"):
        assert key in loaded
    assert loaded["regime"]["code"] in (1, 2, 3, 4)
    assert "## Sectors and indices" in (tmp_path / "r.md").read_text()


def test_dashboard_runs_in_demo(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("MACRONAV_DEMO", "1")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    at.button[1].click().run()   # "Show demo data"
    assert not at.exception, at.exception


def test_yahoo_close_parsing_handles_both_layouts():
    idx = pd.date_range("2026-01-01", periods=3)
    multi = pd.DataFrame(np.arange(12.0).reshape(3, 4), index=idx,
                         columns=pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "QQQ"]]))
    assert list(D._close_from_yf(multi, ["SPY", "QQQ"]).columns) == ["SPY", "QQQ"]
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Open": [1.0, 2.0, 3.0]}, index=idx)
    assert list(D._close_from_yf(flat, ["SPY"]).columns) == ["SPY"]


def test_archive_merge_keeps_old_history(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "ARCHIVE", tmp_path)
    old = pd.Series([3.0, 3.1], index=pd.to_datetime(["2020-01-01", "2020-01-02"]), name="BAMLH0A0HYM2")
    old.to_frame().to_csv(tmp_path / "BAMLH0A0HYM2.csv")
    fresh = pd.Series([3.2, 3.3], index=pd.to_datetime(["2020-01-02", "2026-11-20"]), name="BAMLH0A0HYM2")
    merged = D.merge_archive("BAMLH0A0HYM2", fresh)
    assert len(merged) == 3 and merged.loc["2020-01-02"] == 3.2 and merged.index[0] == pd.Timestamp("2020-01-01")


def test_fred_parsing(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"observations": [{"date": "2026-10-01", "value": "3.1"}, {"date": "2026-11-01", "value": "."}]}

    class Sess:
        def get(self, *a, **k):
            return Resp()

    s = D.fetch_fred("PCEPILFE", "key", "2026-01-01", Sess())
    assert len(s) == 1 and s.iloc[0] == 3.1


def test_research_scoring_maps_modules():
    from macronav.research import _score
    cfg = D.load_yaml("research.yaml")
    item = {"title": "The yield curve and recession risk", "summary": "We study credit spreads and the business cycle."}
    score, module, hits = _score(item, cfg["modules"])
    assert score >= cfg["min_score"] and module == "recession_risk" and "yield curve" in hits


def test_trigger_context_splits_episodes_by_economy():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import engine
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    spx = pd.Series(100.0, index=idx)
    sahm = pd.Series(0.0, index=idx)
    un = pd.Series(4.0, index=idx)
    vix_flag = pd.Series(0.0, index=idx)
    # episode 1 (month 60): economy deteriorating (Sahm 0.8 published the month before), then a 30% fall
    sahm.iloc[55:75] = 0.8
    vix_flag.iloc[60:62] = 1
    spx.iloc[61:73] = np.linspace(95, 70, 12)
    spx.iloc[73:] = 70.0
    # episode 2 (month 150): economy stable, then a 25% rise
    vix_flag.iloc[150:152] = 1
    spx.iloc[151:163] = np.linspace(72, 87.5, 12)
    spx.iloc[163:] = 87.5
    cpi = pd.Series(100 * 1.02 ** (np.arange(240) / 12), index=idx)
    cpi.iloc[50:66] = cpi.iloc[50] * 1.06 ** (np.arange(16) / 12) * 1.03     # inflation above 4% around episode 1
    P = pd.DataFrame({"fred:SAHMREALTIME": sahm, "fred:UNRATE": un, "px:^GSPC": spx, "fred:CPIAUCSL": cpi})
    S = {"Volatility stress (VIX 30+ at month-end)": (vix_flag, "bullish", "")}
    tc = engine.trigger_context(S, spx, P)
    g = tc["triggers"]["Volatility stress (VIX 30+ at month-end)"]["groups"]
    eps = tc["triggers"]["Volatility stress (VIX 30+ at month-end)"]["episodes"]
    assert [e["start"] for e in eps] == ["Jan 2005", "Jul 2012"]
    assert g["deteriorating"]["n"] == 1 and g["deteriorating"]["avg12"] < -0.2 and g["deteriorating"]["dip"] < -0.2
    assert g["stable"]["n"] == 1 and g["stable"]["avg12"] > 0.15
    assert tc["now"]["deteriorating"] is False
    bi = tc["triggers"]["Volatility stress (VIX 30+ at month-end)"]["by_inflation"]
    assert bi["high"]["n"] == 1 and bi["high"]["avg12"] < 0 and bi["low"]["n"] == 1 and bi["low"]["avg12"] > 0


def test_pooled_transfer_detects_stocks_that_follow_the_sp500_trend(tmp_path):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import fetch_all as fa
    import research
    rng = np.random.default_rng(5)
    idx = pd.date_range("1985-01-31", "2025-12-31", freq="ME")
    # S&P with regimes: long rises, occasional falls
    spx_r = np.where((np.arange(len(idx)) // 30) % 4 == 3, -0.015, 0.012) + rng.normal(0, 0.03, len(idx))
    spx = pd.Series(100 * np.exp(np.cumsum(spx_r)), index=idx)
    broken = (spx < spx.rolling(10).mean()).shift(0).fillna(False).values
    days = pd.bdate_range(idx[0] - pd.offsets.MonthBegin(1), idx[-1])
    cols = {}
    for k in range(40):                                   # stocks do worse in the months after the S&P trend breaks
        r = np.where(np.roll(broken, 1), -0.02, 0.015) + rng.normal(0, 0.06, len(idx))
        m = pd.Series(50 * np.exp(np.cumsum(r)), index=idx)
        cols[f"S{k:02d}"] = m.reindex(days, method="bfill")
    store = tmp_path / "store"; store.mkdir()
    fa.save(pd.DataFrame(cols), store / "universe_prices")
    P = pd.DataFrame({"px:^GSPC": spx})
    out = research.pooled_transfer(store, P, np.random.default_rng(1))
    by_h = {r["horizon"]: r for r in out}
    assert set(by_h) == {3, 12}
    assert by_h[3]["late_skill"] > 0 and by_h[3]["verdict"] in ("Solid", "Not proven")


def test_pooled_transfer_counts_only_months_in_the_index(tmp_path, monkeypatch):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import fetch_all as fa
    import research, earnings
    idx = pd.date_range("1990-01-31", "2025-12-31", freq="ME")
    rng = np.random.default_rng(2)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.006, 0.04, len(idx)))), index=idx)
    days = pd.bdate_range(idx[0] - pd.offsets.MonthBegin(1), idx[-1])
    cols = {f"S{k:02d}": pd.Series(50 * np.exp(np.cumsum(rng.normal(0.006, 0.06, len(idx)))), index=idx).reindex(days, method="bfill")
            for k in range(40)}
    store = tmp_path / "store"; store.mkdir()
    fa.save(pd.DataFrame(cols), store / "universe_prices")
    mem = pd.DataFrame({"ticker": list(cols), "start": pd.Timestamp("2005-01-01"), "end": pd.Timestamp("2100-01-01")})
    monkeypatch.setattr(earnings, "load_membership", lambda path: mem)
    out = research.pooled_transfer(store, pd.DataFrame({"px:^GSPC": spx}), np.random.default_rng(1))
    assert out and all("counted only while in the S&P 500" in r["reference"] for r in out)
