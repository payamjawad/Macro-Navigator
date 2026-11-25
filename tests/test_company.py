import sys, pathlib, json
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import company, composite, engine


def _facts():
    rows = []
    for y in (2024, 2025):
        k = 1.0 if y == 2024 else 1.2
        q = pd.Timestamp(f"{y}-09-30")
        f = q + pd.Timedelta(days=35)
        rows += [("revenue", q, f, 100 * k), ("gross_profit", q, f, 40 * k + (5 if y == 2025 else 0)), ("op_income", q, f, 20 * k),
                 ("net_income", q, f, 10 * k), ("rnd", q, f, 8 * k)]
        a = pd.Timestamp(f"{y}-06-30")
        fa = a + pd.Timedelta(days=60)
        rows += [("revenue_a", a, fa, 400 * k), ("net_income_a", a, fa, 40 * k), ("cfo_a", a, fa, 60 * k), ("capex_a", a, fa, 20 * k),
                 ("dividends_a", a, fa, 5), ("buybacks_a", a, fa, 15)]
        rows += [("assets", q, f, 1000), ("liabilities", q, f, 600), ("cash", q, f, 100), ("equity", q, f, 400),
                 ("shares", q, f, 100 if y == 2024 else 98)]
    rows.append(("revenue", pd.Timestamp("2025-09-30"), pd.Timestamp("2026-03-01"), 999.0))   # a later restatement: ignored
    return pd.DataFrame(rows, columns=["key", "end", "filed", "val"])


def test_summarize_first_reported_figures():
    f = _facts()
    f = f.sort_values(["key", "end", "filed"]).groupby(["key", "end"], as_index=False).first()   # as parse_companyfacts does
    s = company.summarize(f, pd.Timestamp("2026-04-01"))
    assert s["quarter_end"] == "2025-09-30" and s["revenue"] == 120
    assert s["revenue_growth"] == pytest.approx(0.2)
    assert s["gross_margin"] == pytest.approx(53 / 120) and s["gross_margin_chg"] == pytest.approx(53 / 120 - 0.4)
    assert s["free_cash_flow_year"] == pytest.approx(48) and s["fcf_margin"] == pytest.approx(0.1)
    assert s["share_change"] == pytest.approx(-0.02) and s["liabilities_share"] == pytest.approx(0.6)


def test_share_split_is_not_reported_as_issuance():
    f = _facts()
    f.loc[(f.key == "shares") & (f.end == pd.Timestamp("2025-09-30")), "val"] = 400                  # 4-for-1 split
    s = company.summarize(f, pd.Timestamp("2026-01-01"))
    assert s["share_change"] is None


def test_next_date_from_last_year():
    react = pd.to_datetime(["2024-01-30", "2024-04-30", "2024-07-30", "2024-10-29", "2025-01-29", "2025-04-29", "2025-07-29", "2025-10-28"])
    d, how = company.next_date(react, pd.Timestamp("2025-11-10"))
    assert abs((d - pd.Timestamp("2026-01-28")).days) <= 3 and "52 weeks" in how
    d2, _ = company.next_date(react, pd.Timestamp("2026-03-15"))                                     # estimate passed: next quarter
    assert d2 > pd.Timestamp("2026-03-12")


def test_earnings_block_from_filings(tmp_path):
    from test_earnings import _daily_store
    tick = _daily_store(tmp_path, effect=0.0)
    C = pd.read_parquet(tmp_path / "earn_close.parquet")
    e = company.earnings_block(tmp_path, tick[:5], C, C["SPY"], budget_s=0, today=C.index[-1])
    assert len(e) == 5
    x = e[tick[0]]
    assert len(x["recent"]) == 8 and 0.04 < x["typical_move"] < 0.15 and x["next"]["date"]


def test_trend_rule_costs_and_decades():
    idx = pd.date_range("1950-01-31", "2025-12-31", freq="ME")
    rng = np.random.default_rng(3)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.006, 0.04, len(idx)))), index=idx)
    P = pd.DataFrame({"fred:TB3MS": 4.0}, index=idx)
    out = engine.trend_rule(spx, P)
    assert out["cost_per_switch"] == 0.001 and len(out["decades"]) >= 7
    assert all(p["switches_per_year"] > 0 for p in out["periods"])
    engine.SWITCH_COST = 0.0
    free = engine.trend_rule(spx, P)
    engine.SWITCH_COST = 0.001
    assert out["periods"][1]["rule"]["cagr"] < free["periods"][1]["rule"]["cagr"]


def test_forward_test_uses_only_months_after_registration(tmp_path):
    from test_stocksignals import _world
    C = _world(persist=0.0002, seed=5, n=80)
    reg = tmp_path / "composite.json"
    reg.write_text(json.dumps({"registered": "2023-01-01", "rule": "test",
                               "features": [{"feature": "ret_12m_skip1m", "sign": 1, "family": "Returns"},
                                            {"feature": "vol_12m", "sign": -1, "family": "Risk"},
                                            {"feature": "gaps_12m", "sign": 1, "family": "Gaps"}]}))          # not computable: dropped
    members = [c for c in C.columns if c != "SPY"]
    out = composite.forward_test(tmp_path, C, members, None, None, reg_path=reg)
    assert out["registered"] and out["features"] == 2 and out["computed"] == 2
    assert out["months"] >= 30 and all(m["formed"] >= "2023-01-01" for m in out["history"])
    assert 6 <= len(out["picks_now"]) <= 10 and out["status"].startswith("Collecting") is (out["months"] < 36)
    none = composite.forward_test(tmp_path, C, members, None, None, reg_path=tmp_path / "missing.json")
    assert none["registered"] is False
