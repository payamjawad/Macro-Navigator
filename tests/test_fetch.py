"""Offline tests for scripts/fetch_all.py and scripts/evidence_check.py.

Every source is simulated in its real response format (FRED JSON and CSV, Ken French zip,
Shiller workbook, Fed CSV, NY Fed workbook, BoE CSV, ECB CSV, yfinance frames)."""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evidence_check as ev  # noqa: E402
import fetch_all as fa  # noqa: E402

RNG = np.random.default_rng(3)
END = pd.Timestamp("2026-11-20")


class Resp:
    def __init__(self, status=200, text=None, content=None, ctype="text/plain"):
        self.status_code = status
        self.text = text if text is not None else (content or b"").decode("latin-1", errors="ignore")
        self.content = content if content is not None else self.text.encode()
        self.headers = {"content-type": ctype}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.text)


def fred_obs(sid, start=None):
    freq = fa.FRED[sid][1]
    first = pd.Timestamp("1950-01-01")
    rule = {"D": "B", "W": "W-FRI", "M": "MS", "Q": "QS"}[freq]
    idx = pd.date_range(first, END, freq=rule)
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    base = {"GS10": 5.0, "TB3MS": 4.0, "BAA": 6.5, "FEDFUNDS": 4.0, "UNRATE": 5.0, "CPIAUCSL": 100.0, "WTISPLC": 20.0}.get(sid, 2.0)
    t = np.arange(len(idx))
    vals = base + np.sin(t / 37) + RNG.normal(0, 0.05, len(idx))
    if sid == "CPIAUCSL":
        vals = 25 * np.exp(0.003 * t + 0.02 * np.sin(t / 50))
    if sid == "WTISPLC":
        vals = 20 * np.exp(0.003 * t + 0.4 * np.sin(t / 30))
    obs = [{"date": d.strftime("%Y-%m-%d"), "value": f"{v:.4f}"} for d, v in zip(idx, vals)]
    if obs:
        obs[-1]["value"] = "."   # FRED marks missing values with a dot
    return obs


def french_zip(header):
    lines = ["This file was created by CMPT_ME_BEME_RETS using the 202610 CRSP database.", "",
             "  Average Value Weighted Returns -- Monthly", header]
    for d in pd.date_range("1926-07-31", "2026-10-31", freq="ME"):
        n = header.count(",")
        lines.append(f"{d:%Y%m}," + ",".join(f"{RNG.normal(1, 4):6.2f}" for _ in range(n)))
    lines += ["", "  Average Equal Weighted Returns -- Monthly", header, "192607,  9.99"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.CSV", "\n".join(lines))
    return buf.getvalue()


def shiller_xlsx():
    dates = pd.date_range("1871-01-01", "2026-10-01", freq="MS")
    rows = [["Stock market data"], ["Robert Shiller"], [], [], [], [],
            ["Date", "P", "D", "E", "CPI", "Date Fraction", "Rate GS10", "Price", "Dividend", "Total Return Price",
             "Earnings", "Scaled Earnings", "CAPE", "", "TR CAPE"]]
    for i, d in enumerate(dates):
        code = float(f"{d.year}.{d.month:02d}")
        rows.append([code, 4 * np.exp(0.0045 * i), 0.2, 0.3, 12 + 0.02 * i, 0, 4.0, 0, 0, 0, 0, 0,
                     15 + 15 * np.sin(i / 200) + (i > 1500) * 15, "", 20])
    rows.append(["", "end"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="Data", header=False, index=False)
    return buf.getvalue()


def gscpi_xlsx():
    d = pd.date_range("1997-09-30", "2026-10-31", freq="ME")
    rows = [["Global Supply Chain Pressure Index"], [], ["Date", "GSCPI"]] + [[x, float(np.sin(i / 12))] for i, x in enumerate(d)]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="GSCPI Monthly Data", header=False, index=False)
    return buf.getvalue()


def fake_get(self, url, params=None, timeout=None, **kw):
    params = params or {}
    if url == fa.FRED_API:
        sid = params["series_id"]
        if sid == "TEDRATE":
            return Resp(400, text=json.dumps({"error_message": "Bad Request. The series does not exist."}), ctype="application/json")
        if params.get("output_type") == 4:
            obs = [{"date": o["date"], "value": o["value"], "realtime_start": "2000-01-05"} for o in fred_obs(sid)[-24:]]
            return Resp(text=json.dumps({"observations": obs}), ctype="application/json")
        return Resp(text=json.dumps({"observations": fred_obs(sid, params.get("observation_start"))}), ctype="application/json")
    if url == fa.FRED_CSV:
        sid = params["id"]
        if sid == "TEDRATE":
            return Resp(404, text="series not found")
        obs = fred_obs(sid, params.get("cosd"))
        csv = "observation_date," + sid + "\n" + "\n".join(f"{o['date']},{o['value']}" for o in obs)
        return Resp(text=csv)
    if "shillerdata.com" in url:
        return Resp(text='<a href="https://img1.wsimg.com/blobby/go/x/downloads/ie_data.xls?ver=1">ie_data</a>')
    if "ie_data.xls" in url:
        return Resp(content=shiller_xlsx())
    if "12_Industry" in url:
        return Resp(content=french_zip(",NoDur,Durbl,Manuf,Enrgy,Chems,BusEq,Telcm,Utils,Shops,Hlth ,Money,Other"))
    if "Research_Data_Factors" in url:
        return Resp(content=french_zip(",Mkt-RF,SMB,HML,RF"))
    if "Momentum" in url:
        return Resp(content=french_zip(",Mom   "))
    if "ebp_csv" in url:
        d = pd.date_range("1973-01-01", "2026-10-01", freq="MS")
        return Resp(text="date,gz_spread,ebp,est_prob\n" + "\n".join(f"{x:%Y-%m-%d},1.5,{np.sin(i / 20):.3f},0.2" for i, x in enumerate(d)))
    if "gscpi" in url:
        return Resp(content=gscpi_xlsx())
    if "bankofengland" in url:
        return Resp(text="<html><body>Access denied</body></html>")
    if "ecb.europa.eu" in url:
        d = pd.date_range("1999-01-01", END, freq="B")
        return Resp(text="KEY,FREQ,TIME_PERIOD,OBS_VALUE\n" + "\n".join(f"FM.D,D,{x:%Y-%m-%d},2.0" for x in d))
    return Resp(404, text="not found")


def fake_download(tickers, period=None, start=None, **kw):
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    idx = pd.bdate_range("1950-01-03" if period == "max" else start, END)
    data = {}
    for i, t in enumerate(tickers):
        if t == "BADTICKER":
            continue
        drift = 0.0003 if t != "^VIX" else 0.0
        path = 100 * np.exp(np.cumsum(np.full(len(idx), drift)) + 0.1 * np.sin(np.arange(len(idx)) / 300 + i))
        if t == "^VIX":
            path = 18 + 12 * np.abs(np.sin(np.arange(len(idx)) / 400))
        data[("Close", t)] = path
        data[("Open", t)] = path
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import requests
    import yfinance
    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(fa.time, "sleep", lambda s: None)
    monkeypatch.setattr(fa, "YAHOO", ["^GSPC", "^VIX", "SPY", "XLE", "BADTICKER"])
    monkeypatch.setenv("FRED_API_KEY", "test")
    return tmp_path / "store"


def test_full_fetch_builds_panels_and_catalog(store):
    assert fa.main(["--out", str(store), "--vintages", "--inprocess"]) == 0
    cat = pd.read_csv(store / "catalog.csv")
    fred = cat[cat.source == "fred"]
    assert (fred.status.str.startswith("ok")).sum() >= len(fa.FRED) - 1      # TEDRATE simulated as discontinued
    assert cat[cat.name == "TEDRATE"].status.iloc[0].startswith("failed")
    assert cat[cat.name == "boe_bank_rate"].status.iloc[0].startswith("failed")  # HTML page detected, not parsed
    assert cat[cat.name == "BADTICKER"].status.iloc[0] == "no data"
    assert (store / "fred_first_release" / "PAYEMS.parquet").exists()
    m = pd.read_parquet(store / "panel_monthly.parquet")
    for c in ("fred:GS10", "px:^GSPC", "shiller:CAPE", "ind:NoDur", "ff:Mkt-RF", "ebp:ebp", "gscpi:GSCPI", "ecb:DFR"):
        assert c in m.columns and m[c].notna().sum() > 100, c
    assert m.index.min() <= pd.Timestamp("1871-01-31")         # Shiller history reaches 1871
    assert m["shiller:CAPE"].notna().sum() > 1800
    d = pd.read_parquet(store / "panel_daily.parquet")
    assert "fred:DGS10" in d.columns and "px:SPY" in d.columns


def test_incremental_run_adds_only_new_rows_and_keeps_archive(store):
    fa.main(["--out", str(store), "--inprocess"])
    hy = pd.read_parquet(store / "fred" / "BAMLH0A0HYM2.parquet").iloc[:, 0]
    old_rows = len(hy)
    # pretend the archive holds older history FRED no longer serves
    ancient = pd.Series([9.9], index=pd.to_datetime(["1940-01-02"]), name="BAMLH0A0HYM2")
    pd.concat([ancient, hy]).rename_axis("date").to_frame().to_parquet(store / "fred" / "BAMLH0A0HYM2.parquet")
    fa.main(["--out", str(store), "--full", "--inprocess"])
    hy2 = pd.read_parquet(store / "fred" / "BAMLH0A0HYM2.parquet").iloc[:, 0]
    assert hy2.index.min() == pd.Timestamp("1940-01-02") and len(hy2) >= old_rows   # archive survived --full
    cat = pd.read_csv(store / "catalog.csv")
    fa.main(["--out", str(store), "--inprocess"])
    cat2 = pd.read_csv(store / "catalog.csv")
    assert cat2[cat2.source == "files"].status.str.startswith("cached").sum() >= 5    # files not re-downloaded
    assert len(cat) == len(cat2)


def test_parsers_on_real_formats():
    s = fa.parse_shiller(shiller_xlsx())
    assert s.index[0] == pd.Timestamp("1871-01-31") and s.index[9] == pd.Timestamp("1871-10-31")   # 1871.1 = October
    f = fa.parse_french(french_zip(",Mkt-RF,SMB,HML,RF"))
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"] and f.index[0] == pd.Timestamp("1926-07-31")
    assert len(f) > 1000                                    # stops before the equal-weighted section
    g = fa.parse_gscpi(gscpi_xlsx())
    assert g.index[0] == pd.Timestamp("1997-09-30")


def test_evidence_check_runs_on_fetched_panel(store, capsys):
    fa.main(["--out", str(store), "--inprocess"])
    assert ev.main(["--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "Evidence check" in out and "Yield curve inverted" in out
    sc = pd.read_csv(store / "evidence_scorecard.csv")
    assert len(sc) == 14 and {"verdict", "episodes", "now"} <= set(sc.columns)
    assert (store / "evidence_report.md").read_text().startswith("# Evidence check")



def test_crashed_group_is_retried_in_safe_mode(tmp_path, monkeypatch, capsys):
    """A segmentation fault in one group must not stop the others, and gets one safe-mode retry."""
    store = tmp_path / "store"
    writer = ("import sys,pandas as pd; g=sys.argv[1]; out=sys.argv[2]; "
              "pd.DataFrame([{'source':g,'name':g+'-x','description':'','first':None,'last':None,"
              "'rows':1,'status':'ok','seconds':0.1}]).to_csv(out+'/catalog_part_'+g+'.csv', index=False); print(g+': 1/1 ok')")
    crash = "import os,signal; print('loading compiled library', flush=True); os.kill(os.getpid(), signal.SIGSEGV)"
    calls = []

    def fake_cmd(group, args, safe):
        calls.append((group, safe))
        if group == "yahoo" and not safe:
            return [sys.executable, "-c", crash]
        return [sys.executable, "-c", writer, group, args.out]

    monkeypatch.setattr(fa, "child_cmd", fake_cmd)
    assert fa.main(["--out", str(store), "--only", "yahoo,fred"]) in (0, 1)
    out = capsys.readouterr().out
    assert "yahoo crashed: segmentation fault" in out and "retrying it in safe mode" in out
    assert ("yahoo", True) in calls and ("fred", True) not in calls        # only the crashed group retries
    cat = pd.read_csv(store / "catalog.csv")
    assert set(cat.name) == {"yahoo-x", "fred-x"}                          # both groups delivered
    assert not list(store.glob("catalog_part_*.csv"))                      # parts cleaned up


def test_engine_runs_and_uses_only_past_data(store):
    import engine
    fa.main(["--out", str(store), "--inprocess"])
    assert engine.main(["--store", str(store)]) == 0
    res = json.loads((store / "engine_results.json").read_text())
    assert 0 <= res["call"]["prob_up_12m"] <= 1
    assert res["track"]["Historical analogs"]["from"] >= "1975-01"
    assert (store / "engine_report.html").read_text().startswith("<!doctype html>")
    # analog neighbours for the latest month must all be at least 12 months old (outcome known)
    import numpy as np
    Z = np.random.default_rng(0).normal(size=(300, 7))
    d = engine.distances(Z, 299, 299 - 12)
    assert len(d) == 288


def test_no_future_rows_and_today_is_last_price(store):
    import engine
    fa.main(["--out", str(store), "--inprocess"])
    m = pd.read_parquet(store / "panel_monthly.parquet")
    assert m.index.max() <= pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
    engine.main(["--store", str(store)])
    res = json.loads((store / "engine_results.json").read_text())
    assert res["data_to"] == m["px:^GSPC"].last_valid_index().strftime("%Y-%m-%d")
    assert res["analog"]["vars_used"] >= 5
    for k in ("Historical analogs", "Combination forecast"):
        t = res["track"][k]
        if res["call"]["weights"][k] > 0:
            assert t["skill"] > 0 and t["top"] > t["bottom"]



def test_criteria_and_triggers_present(store):
    import engine
    fa.main(["--out", str(store), "--inprocess"])
    engine.main(["--store", str(store)])
    res = json.loads((store / "engine_results.json").read_text())
    cr = res["criteria"]
    assert cr["level"] in ("Recovery window", "High risk", "Elevated risk", "Normal")
    assert len(cr["triggers"]) == 2
    html = (store / "engine_report.html").read_text()
    assert "Re-entry triggers to watch" in html and "Macro Navigator" in html


def _run_pipeline(store):
    import engine
    import evidence_check
    fa.main(["--out", str(store), "--inprocess"])
    evidence_check.main(["--store", str(store)])
    engine.main(["--store", str(store)])


def test_site_build_offline(store, tmp_path):
    import build_site
    _run_pipeline(store)
    out = tmp_path / "deploy"
    build_site.main(["--store", str(store), "--out", str(out), "--offline"])
    pub = out / "public"
    s = json.loads((pub / "data" / "summary.json").read_text())
    assert s["economy"]["regime"] and len(s["economy"]["pillars"]) == 4
    assert s["market"]["criteria"]["level"] in ("Recovery window", "High risk", "Elevated risk", "Normal")
    assert any(c["label"] == "S&P 500" for c in s["strip"])
    uni = json.loads((pub / "data" / "universe.json").read_text())
    syms = {u["s"] for u in uni}
    assert {"^GSPC", "SPY", "XLE"} <= syms
    t = json.loads((pub / "data" / "tickers" / "_GSPC.json").read_text())
    assert t["layers"] and len(t["horizons"]) == 3 and t["metrics"]["price"] > 0
    assert (pub / "index.html").exists() and (out / "functions" / "api" / "chat.js").exists()


def test_site_stock_view_with_fundamentals(store, tmp_path, monkeypatch):
    import build_site
    _run_pipeline(store)
    base = fa.load(store / "yahoo_prices")
    idx = base.index
    rng = np.random.default_rng(3)
    test_px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, len(idx)))), index=idx, name="TEST")
    uni = [{"s": "TEST", "n": "Test Energy Corp", "k": "stock", "sector": "Energy", "sub": "Oil & Gas"}] + \
          [{"s": s, "n": n, "k": k, "sector": None, "sub": None} for s, (n, k) in build_site.BASE_ASSETS.items()]
    monkeypatch.setattr(build_site, "load_universe", lambda store: uni)
    monkeypatch.setattr(build_site.fa, "fetch_yahoo", lambda need, store, full, name="x": (test_px.to_frame(), {}, 0.1, 0))
    monkeypatch.setattr(build_site, "sec_fundamentals", lambda store: ({"TEST": {"growth": 0.25, "margin": 0.18, "period": "Q3 2026"}}, "SEC"))
    out = tmp_path / "deploy"
    build_site.main(["--store", str(store), "--out", str(out)])
    t = json.loads((out / "public" / "data" / "tickers" / "TEST.json").read_text())
    names = [l["name"] for l in t["layers"]]
    assert len(t["layers"]) == 5 and names[2].startswith("Sector") and names[4].startswith("Market link")
    assert t["fundamentals"]["growth"] == 0.25 and t["sector_etf"] == "XLE"
    assert t["evidence"]["level"] in ("High", "Medium", "Low") and t["market_link"]["by_trend"]["intact"]
    assert t["metrics"]["beta"] is not None and t["headline"]



def test_rejected_fred_key_falls_back_to_public_csv(store, monkeypatch):
    import requests
    def rejecting_get(self, url, params=None, timeout=None, **kw):
        if url == fa.FRED_API:
            return Resp(400, text=json.dumps({"error_message": "Bad Request. The value for variable api_key is not registered."}),
                        ctype="application/json")
        return fake_get(self, url, params, timeout, **kw)
    monkeypatch.setattr(requests.Session, "get", rejecting_get)
    monkeypatch.setenv("FRED_API_KEY", "  wrong-key\n")
    fa.main(["--out", str(store), "--inprocess"])
    cat = pd.read_csv(store / "catalog.csv")
    fred = cat[cat.source == "fred"]
    assert (fred.status.str.startswith("ok")).sum() >= len(fa.FRED) - 1
    assert fred.status.str.contains("via public CSV").any()


def test_yahoo_retries_tickers_missing_from_the_batch(store, monkeypatch):
    import yfinance
    def flaky(tickers, period=None, start=None, **kw):
        if not isinstance(tickers, str):                      # batch call: Yahoo throttles ^GSPC
            tickers = [t for t in tickers if t != "^GSPC"]
        return fake_download(tickers, period=period, start=start, **kw)
    monkeypatch.setattr(yfinance, "download", flaky)
    fa.main(["--out", str(store), "--inprocess"])
    px = fa.load(store / "yahoo_prices")
    assert "^GSPC" in px.columns and px["^GSPC"].notna().sum() > 1000


def test_trend_rule_and_research_lab_reach_the_app(store, tmp_path):
    import build_site
    import research
    _run_pipeline(store)
    res = json.loads((store / "engine_results.json").read_text())
    tr = res["trend_rule"]
    assert tr["periods"] and {"rule", "buy_hold", "out_share"} <= set(tr["periods"][0]) and "in_stocks" in tr["now"]
    assert "inflation_check" in res["criteria"]
    lab = research.run_if_stale(store, force=True)
    names = {r["signal"] for r in lab["results"]}
    assert "Trend: S&P below its 10-month average" in names and len(lab["results"]) >= 16
    assert all(r["verdict"] in ("Solid", "Not proven", "Fails", "Too little data") for r in lab["results"])
    out = tmp_path / "deploy"
    build_site.main(["--store", str(store), "--out", str(out), "--offline"])
    s = json.loads((out / "public" / "data" / "summary.json").read_text())
    assert s["market"]["research"]["results"] and s["market"]["trend_rule"]["now"]


def test_engine_separates_confirmed_month_end_from_todays_reading(store):
    import engine
    fa.main(["--out", str(store), "--inprocess"])
    engine.main(["--store", str(store)])
    res = json.loads((store / "engine_results.json").read_text())
    c, tr = res["criteria"], res["trend_rule"]
    assert c["month_complete"] in (True, False) and c["confirmed_month"]
    assert set(c["live"]) >= {"level", "differs", "changing"}
    assert c["live"]["differs"] == (c["live"]["level"] != c["level"])
    if not c["month_complete"]:
        assert all(t["provisional"] for t in c["triggers"])
        assert tr.get("live") and tr["now"]["month"] != tr["live"]["month"]
    assert "total_return" in tr and all("switches_per_year" in p for p in tr["periods"])
