import sys, pathlib
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import outlook as ol


def _px(seed=0, years=60, drift=0.0003):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end="2026-11-27", periods=int(years * 252))
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(days)))), index=days)


def test_market_outlook_shapes_and_labels():
    px = _px()
    idx = pd.date_range("1880-01-31", "2026-10-31", freq="ME")
    sp = pd.Series(np.exp(np.linspace(0, 6, len(idx))), index=idx)
    P = pd.DataFrame({"shiller:P": sp, "shiller:D": sp * 0.03, "shiller:CAPE": np.linspace(10, 40, len(idx))}, index=idx)
    o = ol.market_outlook("S&P 500", px, px, P, {"in_stocks": True}, {"in_stocks": False}, "Low", "1970-01-01")
    for k in ("3 months", "12 months", "5 years a year"):
        assert o[k]["p10"] <= o[k]["median"] <= o[k]["p90"] and o[k]["basis"]
    assert o["5 years a year"]["cape_now"] >= 30 and "30 or more" in o["5 years a year"]["basis"]
    opts = o["options"]
    assert opts["months"] > 30 and {"invest_now", "stage_3_months", "wait_for_trend"} <= set(opts)
    assert o["best"]["option"] == "Stage in over 3 months"                     # in stocks at the close, below the average today


def test_options_backtest_arithmetic():
    idx = pd.date_range("2000-01-31", periods=30, freq="ME")
    px = pd.Series(np.arange(100, 130, dtype=float), index=idx)              # steady rise: investing now beats staging
    it = pd.Series(1.0, index=idx)
    res = ol.options_backtest(px, it, pd.Series(True, index=idx))
    assert res["invest_now"]["median"] > res["stage_3_months"]["median"]
    assert res["wait_for_trend"]["median"] == pytest.approx(res["invest_now"]["median"])     # trend already in: same as investing now
    it2 = pd.Series(0.0, index=idx); it2.iloc[5:] = 1.0
    res2 = ol.options_backtest(px, it2, pd.Series([True] + [False] * 29, index=idx))
    assert res2["wait_for_trend"]["median"] == pytest.approx(px.iloc[12] / px.iloc[1 + 4] - 1)   # bought when the trend turned


def test_best_option_rules():
    assert ol.best_option({"in_stocks": False}, None, -0.2, "Low")["option"].startswith("Wait")
    assert ol.best_option({"in_stocks": True}, {"in_stocks": True}, -0.02, "Low")["option"] == "Invest normally"
    assert "cash buffer" in ol.best_option({"in_stocks": True}, {"in_stocks": True}, -0.02, "Watch")["why"]


def test_stock_outlook_scales_with_beta_and_sizes_risk():
    spx = _px(1)
    stock = pd.Series(spx.values ** 1.0, index=spx.index) * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.02, len(spx))))
    o = ol.stock_outlook(stock, spx, 2.0, 0.09, 0.04)
    assert o["12 months"]["central"] == pytest.approx(0.04 + 2.0 * (0.09 - 0.04))
    assert 0 < o["same_risk_as_10pct_sp500"] < 0.10 and o["review_fall_3m"] < 0
