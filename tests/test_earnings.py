import json, os, sys, pathlib
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import earnings as ea


def _fake_sec(store, filings: dict):
    """filings: ticker -> list of acceptance strings. Writes fresh cache files so no network is used."""
    sec = store / "sec"; (sec / "subs").mkdir(parents=True, exist_ok=True)
    tick = {str(i): {"cik_str": 1000 + i, "ticker": t, "title": t} for i, t in enumerate(filings)}
    (sec / "company_tickers.json").write_text(json.dumps(tick))
    for i, (t, accs) in enumerate(filings.items()):
        recent = {"form": ["8-K"] * len(accs) + ["10-Q"], "acceptanceDateTime": accs + [accs[0]],
                  "items": ["2.02,9.01"] * len(accs) + [""]}
        (sec / "subs" / f"{1000 + i}.json").write_text(json.dumps({"filings": {"recent": recent, "files": []}}))


def test_announcements_timezone_and_timing(tmp_path):
    _fake_sec(tmp_path, {"AAA": ["2020-01-30T16:05:00.000Z", "2020-04-30T07:00:00.000Z", "2020-07-30T16:10:00.000Z",
                                 "2020-10-29T16:05:00.000Z"]})
    ann = ea.announcements(tmp_path, ["AAA"])
    assert len(ann) == 4 and ann.ts.iloc[0].hour == 16                  # Eastern clock time kept
    days = pd.bdate_range("2020-01-01", "2020-12-31")
    ev = ea.classify(ann, days)
    assert list(ev.timing) == ["amc", "bmo", "amc", "amc"]
    assert ev.reaction.iloc[0] == pd.Timestamp("2020-01-31") and ev.reaction.iloc[1] == pd.Timestamp("2020-04-30")


def _daily_store(tmp_path, effect: float, pre: float = 0.0):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2001-01-02", "2025-06-30")
    n = len(days)
    tickers = [f"S{k:02d}" for k in range(40)]
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    O, C, V = {"SPY": spy * (1 + rng.normal(0, 0.001, n))}, {"SPY": spy}, {"SPY": np.full(n, 1e8)}
    filings = {}
    for t in tickers:
        r = rng.normal(0.0002, 0.015, n)
        opens_gap = np.zeros(n)
        accs = []
        for rp in range(70 + rng.integers(0, 50), n - 70, 65):            # about 91 calendar days, like real quarters
            sgn = rng.choice([-1, 1])
            opens_gap[rp] = sgn * 0.08                                      # big earnings gap at the open: +8% or -8%,
            r[rp] += np.log1p(sgn * 0.08)                                     # a fair bet (no built-in premium)
            r[rp + 1: rp + 6] += sgn * effect / 5                           # planted drift over the next 5 days
            r[rp - 5: rp] += pre / 5                                          # planted run-up into the announcement
            accs.append((days[rp - 1] + pd.Timedelta(hours=16, minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        c = 50 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * (1 + opens_gap)
        O[t], C[t], V[t] = o, c, np.full(n, 1e6)
        filings[t] = accs
    for k, d in (("open", O), ("close", C), ("volume", V)):
        pd.DataFrame(d, index=days).to_parquet(tmp_path / f"earn_{k}.parquet")
    _fake_sec(tmp_path, filings)
    return tickers


def _cell(res, entry, exit_, cond, strat):
    return next(c for c in res["daily"]["cells"] if c["entry"] == entry and c["exit"] == exit_ and c["condition"] == cond and c["strategy"] == strat)


def test_daily_study_finds_a_planted_drift_and_not_a_fake_one(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPACA_KEY_ID", raising=False)
    tick = _daily_store(tmp_path, effect=0.03)
    res = ea.run(tmp_path, tick, budget_s=60)
    assert res["daily"]["events"] > 2000
    c = _cell(res, "close", "5d", "big_up", "continuation")
    assert c["late"]["mean"] > 0.015 and c["verdict"] in ("Solid", "Not proven")
    r = _cell(res, "close", "5d", "big_up", "reversal")
    assert r["verdict"] == "Fails"
    assert "Minutes and hours" in res["intraday"]["summary"]
    tmp2 = tmp_path / "none"; tmp2.mkdir()
    tick = _daily_store(tmp2, effect=0.0)
    res0 = ea.run(tmp2, tick, budget_s=60)
    c0 = _cell(res0, "close", "5d", "big_up", "continuation")
    assert c0["verdict"] != "Solid"


def test_intraday_points_for_an_after_close_release():
    def bars(sym_base, day0, day1):
        out = []
        for t in pd.date_range(f"{day0} 15:55", f"{day0} 19:59", freq="min"):
            price = sym_base if t.hour < 16 else sym_base * 1.10          # +10% after the 16:05 release
            out.append({"t": t.tz_localize(ea.ET).tz_convert("UTC").isoformat(), "o": price, "c": price})
        for t in pd.date_range(f"{day1} 09:30", f"{day1} 15:59", freq="min"):
            k = (t - pd.Timestamp(f"{day1} 09:30")).seconds // 60
            price = sym_base * 1.10 * (1 + 0.001 * k)                       # keeps rising 0.1% a minute
            out.append({"t": t.tz_localize(ea.ET).tz_convert("UTC").isoformat(), "o": price, "c": price})
        return out
    b = {"AAA": bars(100.0, "2024-08-01", "2024-08-02"), "SPY": [{"t": x["t"], "o": 500.0, "c": 500.0} for x in bars(500.0, "2024-08-01", "2024-08-02")]}
    row = pd.Series({"reaction": pd.Timestamp("2024-08-02"), "prevday": pd.Timestamp("2024-08-01"), "timing": "amc",
                     "ts": pd.Timestamp("2024-08-01 16:05:00")})
    p = ea.intraday_points(row, b, "AAA")
    assert p["prev"] == pytest.approx(100.0) and p["a5"] == pytest.approx(110.0)
    assert p["r0"] == pytest.approx(110.0) and p["r5"] == pytest.approx(110.0 * 1.004)


def test_intraday_fetch_batches_symbols_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_KEY_ID", "k"); monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    calls = []
    def fake_bars(symbols, start, end, H):
        calls.append(symbols)
        out = {}
        for sym in symbols.split(","):
            bars = []
            for t in list(pd.date_range(f"{start.date()} 15:55", f"{start.date()} 19:59", freq="min")) + \
                     list(pd.date_range(f"{end.date()} 09:30", f"{end.date()} 15:59", freq="min")):
                bars.append({"t": t.tz_localize(ea.ET).tz_convert("UTC").isoformat(), "o": 10.0, "c": 10.0})
            out[sym] = bars
        return out
    monkeypatch.setattr(ea, "_bars", fake_bars)
    ev = pd.DataFrame({"ticker": [f"T{i}" for i in range(10)] + ["BRK-B"],
                       "reaction": pd.Timestamp("2024-08-02"), "prevday": pd.Timestamp("2024-08-01"),
                       "timing": "amc", "ts": pd.Timestamp("2024-08-01 16:05")})
    I = ea.intraday_fetch(tmp_path, ev, budget_s=60)
    assert len(I) == 11 and len(calls) == 2                                # 11 companies in 2 requests
    assert any("BRK.B" in c for c in calls) and all(c.endswith(",SPY") for c in calls)
    calls.clear()
    I2 = ea.intraday_fetch(tmp_path, ev, budget_s=60)                     # second run: nothing left to fetch
    assert len(I2) == 11 and calls == []


def test_membership_at_the_time():
    mem = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"], "start": pd.to_datetime(["2005-01-01", "2015-01-01", "2010-01-01"]),
                        "end": pd.to_datetime(["2008-01-01", "2100-01-01", "2012-01-01"])})
    df = pd.DataFrame({"ticker": ["AAA", "AAA", "AAA", "BBB", "CCC"],
                       "reaction": pd.to_datetime(["2006-05-01", "2010-05-01", "2020-05-01", "2013-01-01", "2020-01-01"])})
    assert list(ea.member_at(df, mem)) == [True, False, True, False, False]


def test_control_difference_must_clear_luck():
    late = {"mean": 0.006, "se": 0.0015, "t": 4.0}
    assert ea.verdict({"mean": 0.01}, late, {"mean": 0.004, "se": 0.0015}) != "Solid"     # +0.2% over control, t ~ 0.9
    assert ea.verdict({"mean": 0.01}, late, {"mean": 0.000, "se": 0.0010}) == "Solid"


def test_announcement_premium_finds_a_planted_run_up_and_not_a_fake_one(tmp_path):
    def run(pre, sub):
        d = tmp_path / sub; d.mkdir()
        tick = _daily_store(d, effect=0.0, pre=pre)
        return ea.premium_study(d, tick)
    res = run(0.01, "pre")
    c = next(c for c in res["cells"] if c["version"] == "actual date" and c["entry_days_before"] == 5 and c["exit"] == "just before the news")
    assert c["late"]["mean"] > 0.005 and c["verdict"] == "Solid"
    s = next(c for c in res["cells"] if c["version"] == "expected date" and c["entry_days_before"] == 5 and c["exit"] == "just before the news")
    assert s["late"]["mean"] > 0.003                                       # predicted dates catch most of it (regular schedule)
    res0 = run(0.0, "null")
    assert not any(c["verdict"] == "Solid" for c in res0["cells"])
    assert res0["by_year"]["expected date"]


def test_expected_dates_ignore_extra_results_filings():
    days = pd.bdate_range("2009-01-01", "2014-12-31")
    rng = np.random.default_rng(1)
    C = pd.DataFrame({"AAA": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))),
                      "SPY": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))}, index=days)
    ts = [pd.Timestamp(f"{y}-{m:02d}-{d:02d} 16:05") for y in range(2009, 2015) for m, d in ((2, 2), (5, 3), (8, 2), (11, 2))]
    ts += [pd.Timestamp("2011-03-15 16:05"), pd.Timestamp("2012-09-20 16:05")]           # extra results filings
    ann = pd.DataFrame({"ticker": "AAA", "ts": sorted(ts)})
    R = ea.premium_rows(ann, {"close": C})
    s = R[(R.version == "expected date") & (R.k == 1) & (R.exit == "after the reaction day")]
    regular = s[~s.reaction.isin(pd.to_datetime(["2011-03-16", "2012-09-21"]))]
    assert len(regular) >= 15 and (regular.date_error.abs() <= 3).mean() > 0.9 and regular.days_held.max() <= 5
