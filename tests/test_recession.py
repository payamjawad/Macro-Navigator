import sys, pathlib
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import fetch_all as fa
import recession as rc

PEAKS = ["1953-07", "1957-08", "1960-04", "1969-12", "1973-11", "1980-01", "1981-07", "1990-07", "2001-03", "2007-12", "2020-02"]
TROUGHS = ["1954-05", "1958-04", "1961-02", "1970-11", "1975-03", "1980-07", "1982-11", "1991-03", "2001-11", "2009-06", "2020-04"]


def _store(tmp_path):
    store = tmp_path / "store"
    (store / "fred").mkdir(parents=True)
    (store / "fred_first_release").mkdir()
    m = pd.period_range("1948-01", "2026-10", freq="M")
    rec = pd.Series(0.0, index=m)
    for p, t in zip(PEAKS, TROUGHS):
        rec.loc[pd.Period(p) + 1: pd.Period(t)] = 1
    starts = rec[(rec == 1) & (rec.shift(1, fill_value=0) == 0)].index
    ts = m.to_timestamp()
    rng = np.random.default_rng(0)
    tb3 = pd.Series(4.0, index=m)
    for s in starts:
        tb3.loc[s - 10: s - 1] = 6.5                                     # short rate above the 10-year before each recession
    series = {"USREC": rec, "GS10": pd.Series(5.0, index=m), "TB3MS": tb3, "GS1": tb3, "BAA": pd.Series(6.5, index=m) + rng.normal(0, 0.05, len(m)),
              "UMCSENT": pd.Series(90.0, index=m), "UNRATE": pd.Series(5.0, index=m), "SAHMREALTIME": pd.Series(0.0, index=m),
              "PAYEMS": pd.Series(np.linspace(40000, 160000, len(m)), index=m), "INDPRO": pd.Series(np.linspace(20, 105, len(m)), index=m),
              "PERMIT": pd.Series(1500.0, index=m), "HTRUCKSSAAR": pd.Series(400.0, index=m), "AWHMAN": pd.Series(41.0, index=m)}
    for sid, s in series.items():
        fa.save(pd.DataFrame({sid: s.values}, index=ts), store / "fred" / sid)
    for sid in ("PAYEMS", "INDPRO"):
        v = series[sid]
        fa.save(pd.DataFrame({"value": v.values, "first_release": (ts + pd.offsets.MonthEnd(0) + pd.Timedelta(days=7))}, index=ts),
                store / "fred_first_release" / sid)
    days = pd.bdate_range("1950-01-02", "2026-11-27")
    r = rng.normal(0.0003, 0.01, len(days))
    dm = days.to_period("M")
    for s in starts:
        r[(dm >= s - 3) & (dm <= s + 8)] -= 0.0015
    r[-60:] -= 0.0025                                                    # a dip right now
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=days)
    D = pd.DataFrame({"px:^GSPC": px, "px:^NDX": px * 1.2})
    return store, D, px


def test_watch_finds_planted_warnings_and_odds(tmp_path):
    store, D, _ = _store(tmp_path)
    w, G = rc.recession_watch(store, pd.DataFrame(), D)
    assert w and w["recessions"] >= 10 and w["warned"] >= 10                 # the curve warned before each recession
    assert w["odds"]["Watch"]["p12"] > w["odds"]["Low"]["p12"] or w["odds"].get("High", {}).get("p12", 0) > w["odds"]["Low"]["p12"]
    assert w["curve"]["state"] in ("Inverted", "Re-steepening after an inversion", "Normal")
    names = [s["name"] for s in w["signals"]]
    assert "Yield curve inverted in the last 18 months" in names and "Payrolls down over 3 months (first report)" in names


def test_situation_and_stock_after_dips(tmp_path):
    store, D, px = _store(tmp_path)
    w, G = rc.recession_watch(store, pd.DataFrame(), D)
    sit = rc.situation(D, G, w["level"])
    spx = sit["indices"]["S&P 500"]
    assert spx["from_high"] < -0.03 and spx["episodes"] >= 5
    o = spx["outcomes"]["6 months"]
    assert o["n"] >= 5 and 0 <= o["up"] <= 1 and o["p10"] <= o["median"] <= o["p90"]
    prices = pd.DataFrame({"ABC": px * 2})
    a = rc.stock_after_dips(prices, "ABC", D["px:^GSPC"], spx["event_dates"])
    assert a["episodes"] >= 5 and a["6 months"]["mean"] == pytest.approx(a["6 months index"]["mean"], abs=1e-9)   # same path as the index


def test_first_report_counts_only_after_publication():
    months = pd.period_range("2010-01", "2010-04", freq="M")
    idx = pd.to_datetime(["2009-12-01", "2010-01-01", "2010-02-01", "2010-03-01"])
    fp = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "published": pd.to_datetime(["2010-01-08", "2010-02-05", "2010-03-05", "2010-04-02"])}, index=idx)
    s, share = rc.first_print_signal(fp, lambda v: v, pd.Series(dtype=float), months)
    assert s[pd.Period("2010-01", "M")] == 1.0 and s[pd.Period("2010-02", "M")] == 2.0 and share == 1.0
