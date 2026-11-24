#!/usr/bin/env python3
"""
build_site.py - builds the Macro Navigator web app for Cloudflare Pages.

    python build_site.py --store data/store --out deploy

Run after fetch_all.py, evidence_check.py and engine.py. Writes:
    deploy/public/index.html (+ icons, manifest)    the app
    deploy/public/data/summary.json                 economy, market strip, S&P 500 stance and evidence
    deploy/public/data/universe.json                every searchable ticker
    deploy/public/data/tickers/<SYMBOL>.json        one ready-made view per stock, index or ETF
    deploy/functions/api/*.js                       chat (Gemini), refresh and watchlist endpoints
Universe: S&P 500 + Nasdaq 100 members (Wikipedia), config/watchlist.txt, main indices and ETFs.
Fundamentals: SEC XBRL "frames" (one request per concept and quarter covers every filer).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE
sys.path.insert(0, str(HERE))
import fetch_all as fa  # noqa: E402
from evidence_check import build_signals, col, episodes, load_panel, worst_dip_12m  # noqa: E402
from engine import finite, latest, load_daily  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) macro-navigator-build/1.0"}
SEC_UA = {"User-Agent": "Macro Navigator personal research tool " + (os.environ.get("SEC_CONTACT") or "macro-navigator@users.noreply.github.com").strip(),
          "Accept-Encoding": "gzip, deflate"}
BASE_ASSETS = {
    "^GSPC": ("S&P 500", "index"), "^NDX": ("Nasdaq 100", "index"), "^RUT": ("Russell 2000", "index"), "^DJI": ("Dow Jones", "index"),
    "SPY": ("SPDR S&P 500 ETF", "etf"), "QQQ": ("Invesco QQQ (Nasdaq 100)", "etf"), "IWM": ("iShares Russell 2000", "etf"),
    "DIA": ("SPDR Dow Jones ETF", "etf"), "XLK": ("Technology sector", "etf"), "XLF": ("Financials sector", "etf"),
    "XLV": ("Health care sector", "etf"), "XLY": ("Consumer discretionary sector", "etf"), "XLP": ("Consumer staples sector", "etf"),
    "XLE": ("Energy sector", "etf"), "XLI": ("Industrials sector", "etf"), "XLB": ("Materials sector", "etf"),
    "XLU": ("Utilities sector", "etf"), "XLRE": ("Real estate sector", "etf"), "XLC": ("Communication services sector", "etf"),
    "SMH": ("Semiconductors", "etf"), "GLD": ("Gold", "etf"), "TLT": ("Long Treasuries", "etf"),
    "^FTSE": ("FTSE 100", "index"),
}
# Well-known stocks outside the S&P 500 and Nasdaq 100 (mostly US-listed foreign companies)
EXTRA_STOCKS = {"TSM": ("Taiwan Semiconductor (TSMC)", "Information Technology", "Semiconductors"),
                "ASML": ("ASML Holding", "Information Technology", "Semiconductor Materials & Equipment"),
                "ARM": ("Arm Holdings", "Information Technology", "Semiconductors"),
                "NVO": ("Novo Nordisk", "Health Care", "Pharmaceuticals"), "SAP": ("SAP", "Information Technology", "Application Software"),
                "BABA": ("Alibaba", "Consumer Discretionary", "Broadline Retail"), "SONY": ("Sony", "Consumer Discretionary", "Consumer Electronics"),
                "TM": ("Toyota", "Consumer Discretionary", "Automobile Manufacturers"), "SHOP": ("Shopify", "Information Technology", "Internet Services")}
GICS_ETF = {"Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV", "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP", "Energy": "XLE", "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU",
            "Real Estate": "XLRE", "Communication Services": "XLC"}
REV_TAGS = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax"]


def fnum(x, d=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(x) else round(x, d)


def file_name(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "_", sym)


# --------------------------------------------------------------------------- market strip and economy

def last(s: pd.Series):
    s = s.dropna()
    return (float(s.iloc[-1]), s.index[-1]) if len(s) else (None, None)


def strip_block(D: pd.DataFrame, P: pd.DataFrame) -> list[dict]:
    out = []
    spx = D["px:^GSPC"].dropna() if "px:^GSPC" in D else pd.Series(dtype=float)
    if len(spx):
        hi = spx.tail(252).max()
        out.append({"label": "S&P 500", "value": f"{spx.iloc[-1]:,.0f}", "note": f"{(spx.iloc[-1] / hi - 1) * 100:+.1f}% from high",
                    "tone": "bad" if spx.iloc[-1] / hi - 1 <= -0.05 else "neutral", "date": str(spx.index[-1].date())})
    v, d = last(D["px:^VIX"]) if "px:^VIX" in D else (None, None)
    if v is not None:
        out.append({"label": "VIX", "value": f"{v:.1f}", "note": "Calm" if v < 18 else ("Elevated" if v < 30 else "Stress"),
                    "tone": "good" if v < 18 else ("warn" if v < 30 else "bad"), "date": str(d.date())})
    v, d = last(D["fred:DGS10"]) if "fred:DGS10" in D else (None, None)
    if v is not None:
        out.append({"label": "US 10-year", "value": f"{v:.2f}%", "note": "Yield", "tone": "neutral", "date": str(d.date())})
    v, d = last(D["fred:DCOILBRENTEU"]) if "fred:DCOILBRENTEU" in D else (None, None)
    if v is not None:
        out.append({"label": "Brent", "value": f"${v:.1f}", "note": "Crude oil", "tone": "neutral", "date": str(d.date())})
    v, d = last(D["fred:BAMLH0A0HYM2"]) if "fred:BAMLH0A0HYM2" in D else (None, None)
    if v is not None:
        bp = v * 100
        out.append({"label": "HY spread", "value": f"{bp:.0f} bp", "note": "Calm" if bp < 400 else ("Watch" if bp < 500 else "Stress"),
                    "tone": "good" if bp < 400 else ("warn" if bp < 500 else "bad"), "date": str(d.date())})
    ff = D["fred:DFEDTARU"].dropna() if "fred:DFEDTARU" in D else pd.Series(dtype=float)
    if len(ff):
        up = ff.iloc[-1]
        six = ff[ff.index <= ff.index[-1] - pd.Timedelta(days=182)]
        chg = up - six.iloc[-1] if len(six) else 0
        out.append({"label": "Fed funds", "value": f"{up - 0.25:.2f}–{up:.2f}%",
                    "note": "Hiking" if chg > 0 else ("Cutting" if chg < 0 else "On hold"),
                    "tone": "bad" if chg > 0 else ("good" if chg < 0 else "neutral"), "date": str(ff.index[-1].date())})
    return out


def economy_block(P: pd.DataFrame, res: dict) -> dict:
    unrate, _ = last(latest(P, "fred:UNRATE"))
    pay = col(P, "fred:PAYEMS").dropna()
    pay3 = float(pay.diff().tail(3).mean()) if len(pay) > 4 else None
    pay_last = float(pay.diff().iloc[-1]) if len(pay) > 2 else None
    cpi = col(P, "fred:CPIAUCSL")
    cpi_yoy, cpi_d = last((cpi / cpi.shift(12) - 1) * 100)
    pce = col(P, "fred:PCEPILFE")
    pce_yoy, pce_d = last((pce / pce.shift(12) - 1) * 100)
    sahm, _ = last(col(P, "fred:SAHMREALTIME"))
    ff = col(P, "fred:DFEDTARU").dropna()
    ff_up = float(ff.iloc[-1]) if len(ff) else None
    ff6 = float(ff.iloc[-1] - ff.iloc[-7]) if len(ff) > 7 else 0.0
    ff3 = float(ff.iloc[-1] - ff.iloc[-4]) if len(ff) > 4 else 0.0
    spread6, _ = last((latest(P, "fred:BAA") - latest(P, "fred:GS10")).diff(6))
    hy, _ = last(col(P, "fred:BAMLH0A0HYM2"))
    hy = hy * 100 if hy is not None else None
    curve, _ = last(col(P, "fred:T10Y3M"))
    rec = 0.5 * (1 + math.erf((-0.5333 - 0.6330 * curve) / math.sqrt(2))) if curve is not None else None

    if sahm is not None and sahm >= 0.5:
        growth = ("Contracting", "bad")
    elif (sahm is not None and sahm >= 0.3) or (pay3 is not None and pay3 < 50):
        growth = ("Softening", "warn")
    else:
        growth = ("Holding", "good")
    if (pce_yoy or 0) >= 3.5 or (cpi_yoy or 0) >= 5:
        infl = ("Hot", "bad")
    elif (pce_yoy or 0) >= 2.5:
        infl = ("Above target", "warn")
    else:
        infl = ("Near target", "good")
    policy = ("Tightening", "bad") if ff6 > 0 else (("Easing", "good") if ff3 < 0 else ("On hold", "neutral"))
    if (hy or 0) >= 500 or (spread6 or 0) >= 0.5:
        credit = ("Stressed", "bad")
    elif (hy or 0) >= 400:
        credit = ("Watch", "warn")
    else:
        credit = ("Calm", "good")

    if credit[0] == "Stressed":
        regime, tone, text = "Credit stress", "bad", "Funding conditions are tightening fast; watch for spillover into jobs and spending."
    elif growth[0] == "Contracting":
        regime, tone, text = "Slowdown", "bad", "Job losses have reached the level that has marked recessions."
    elif infl[1] != "good" and policy[0] == "Tightening":
        regime, tone, text = "Late cycle, held back by inflation", "warn", ("Growth is holding but inflation is above target and the Fed "
                                                                               "is tightening.")
    elif infl[1] != "good":
        regime, tone, text = "Expansion with sticky inflation", "warn", "Growth is holding; inflation is keeping policy from easing."
    elif policy[0] == "Easing" and growth[0] != "Holding":
        regime, tone, text = "Slowing, policy easing", "warn", "Growth is cooling and the Fed is cutting to support it."
    else:
        regime, tone, text = "Expansion, inflation contained", "good", "Growth is holding, inflation is near target and credit is calm."

    state = ((res.get("analog") or {}).get("state")) or {}
    return {
        "regime": regime, "tone": tone, "text": text,
        "pillars": [
            {"name": "Growth", "label": growth[0], "tone": growth[1],
             "detail": f"Unemployment {unrate:.1f}%, payrolls {pay_last:+.0f}k last month" if unrate is not None and pay_last is not None else "n/a"},
            {"name": "Inflation", "label": infl[0], "tone": infl[1],
             "detail": f"CPI {cpi_yoy:.1f}% ({cpi_d:%b}), core PCE {pce_yoy:.1f}%" if cpi_yoy is not None and pce_yoy is not None else "n/a"},
            {"name": "Policy", "label": policy[0], "tone": policy[1],
             "detail": f"Fed funds {ff_up - 0.25:.2f}–{ff_up:.2f}%, {ff6 * 100:+.0f} bp in 6 months" if ff_up is not None else "n/a"},
            {"name": "Credit", "label": credit[0], "tone": credit[1],
             "detail": (f"High-yield spread {hy:.0f} bp; Baa spread {spread6:+.2f} pts in 6 months" if hy is not None and spread6 is not None
                        else "n/a")},
        ],
        "history": [{"name": k, "z": fnum(v, 2)} for k, v in state.items()],
        "recession_prob": fnum(rec, 3),
        "watch": [
            {"name": "Recession turning point", "rule": "Sahm indicator reaches 0.50", "now": f"{sahm:.2f}" if sahm is not None else "n/a"},
            {"name": "Inflation surge", "rule": "CPI 5% or more and rising", "now": f"{cpi_yoy:.1f}%" if cpi_yoy is not None else "n/a"},
            {"name": "Credit stress", "rule": "Baa spread +0.5 pts in 6 months", "now": f"{spread6:+.2f}" if spread6 is not None else "n/a"},
        ],
    }


# --------------------------------------------------------------------------- universe

def wiki_members(url: str) -> list[dict]:
    html = requests.get(url, headers=UA, timeout=30).text
    for t in pd.read_html(io.StringIO(html)):
        cols = {str(c).strip(): c for c in t.columns}
        tcol = next((cols[c] for c in ("Symbol", "Ticker") if c in cols), None)
        if tcol is None or len(t) < 50:
            continue
        ncol = next((cols[c] for c in ("Security", "Company", "Name") if c in cols), None)
        scol = cols.get("GICS Sector")
        subcol = cols.get("GICS Sub-Industry")
        out = []
        for _, r in t.iterrows():
            sym = str(r[tcol]).strip().replace(".", "-")
            if not re.fullmatch(r"[A-Z0-9\-]{1,8}", sym):
                continue
            out.append({"s": sym, "n": str(r[ncol]) if ncol is not None else sym, "k": "stock",
                        "sector": str(r[scol]) if scol is not None else None, "sub": str(r[subcol]) if subcol is not None else None})
        if out:
            return out
    return []


def load_universe(store: Path) -> list[dict]:
    cache = store / "site_universe.json"
    members: dict[str, dict] = {}
    try:
        for url in ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "https://en.wikipedia.org/wiki/Nasdaq-100"):
            for m in wiki_members(url):
                members.setdefault(m["s"], m)
                if not members[m["s"]].get("sector") and m.get("sector"):
                    members[m["s"]].update({k: m[k] for k in ("sector", "sub")})
        if len(members) > 300:
            cache.write_text(json.dumps(list(members.values())))
    except Exception as err:
        print(f"  member lists unavailable ({err.__class__.__name__}); using the last saved list")
    if len(members) < 300 and cache.exists():
        members = {m["s"]: m for m in json.loads(cache.read_text())}
    wl = ROOT / "config" / "watchlist.txt"
    if wl.exists():
        for line in wl.read_text().splitlines():
            sym = line.split("#")[0].strip().upper()
            if sym and sym not in members and sym not in BASE_ASSETS:
                members[sym] = {"s": sym, "n": sym, "k": "stock", "sector": None, "sub": None, "watch": True}
            elif sym in members:
                members[sym]["watch"] = True
    for s, (n, sec, sub) in EXTRA_STOCKS.items():
        members.setdefault(s, {"s": s, "n": n, "k": "stock", "sector": sec, "sub": sub})
    for s, (n, k) in BASE_ASSETS.items():
        members[s] = {"s": s, "n": n, "k": k, "sector": None, "sub": None}
    return list(members.values())


# --------------------------------------------------------------------------- SEC fundamentals

def sec_get(url: str, cache: Path, ttl_h: float = 72):
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < ttl_h:
        return json.loads(cache.read_text())
    r = requests.get(url, headers=SEC_UA, timeout=60)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    time.sleep(0.15)                      # stay well under the SEC's 10 requests a second
    return r.json()


def sec_fundamentals(store: Path) -> tuple[dict, str]:
    sec = store / "sec"
    try:
        tick = sec_get("https://www.sec.gov/files/company_tickers.json", sec / "company_tickers.json", 168)
    except Exception as err:
        code = getattr(getattr(err, "response", None), "status_code", "")
        hint = " - add your email as the SEC_CONTACT secret" if code in (401, 403) else ""
        return {}, f"SEC unavailable ({err.__class__.__name__} {code}{hint})".replace(" )", ")")
    cik_to_t = {int(v["cik_str"]): str(v["ticker"]).upper().replace(".", "-") for v in tick.values()}

    def frame(tag, y, q):
        try:
            d = sec_get(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/CY{y}Q{q}.json", sec / f"{tag}_{y}Q{q}.json")
            return {int(x["cik"]): float(x["val"]) for x in d.get("data", [])}
        except Exception:
            return {}

    def revenue(y, q):
        out = {}
        for tag in REV_TAGS:
            for cik, v in frame(tag, y, q).items():
                out.setdefault(cik, v)
        return out

    today = pd.Timestamp.today()
    y, q = today.year, (today.month - 1) // 3        # start with the last completed quarter
    if q == 0:
        y, q = y - 1, 4
    for _ in range(6):
        now = revenue(y, q)
        if len(now) >= 1500:
            break
        y, q = (y, q - 1) if q > 1 else (y - 1, 4)
    else:
        return {}, "SEC frames had too little coverage"
    prev = revenue(y - 1, q)
    ni = frame("NetIncomeLoss", y, q)
    out = {}
    for cik, rev in now.items():
        t = cik_to_t.get(cik)
        if not t or rev <= 0:
            continue
        g = rev / prev[cik] - 1 if prev.get(cik, 0) > 0 else None
        m = ni[cik] / rev if cik in ni else None
        out[t] = {"growth": fnum(g), "margin": fnum(m), "period": f"Q{q} {y}", "revenue": rev}
    return out, f"SEC filings, quarter Q{q} {y}"


# --------------------------------------------------------------------------- per-ticker views

def ret(p: pd.Series, n: int):
    return float(p.iloc[-1] / p.iloc[-1 - n] - 1) if len(p) > n else None


def metrics(p: pd.Series, spx: pd.Series) -> dict | None:
    p = p.dropna()
    if len(p) < 60:
        return None
    hi = float(p.tail(252).max())
    out = {"price": fnum(p.iloc[-1], 2), "date": str(p.index[-1].date()), "chg_1d": fnum(ret(p, 1)),
           "ret_1m": fnum(ret(p, 21)), "ret_3m": fnum(ret(p, 63)), "ret_12m": fnum(ret(p, 252)),
           "from_high": fnum(p.iloc[-1] / hi - 1), "sma200": fnum(p.tail(200).mean(), 2) if len(p) >= 200 else None,
           "above_200d": bool(p.iloc[-1] > p.tail(200).mean()) if len(p) >= 200 else None,
           "vol": fnum(p.pct_change().tail(126).std() * math.sqrt(252))}
    s3 = ret(spx.dropna(), 63)
    out["rel_3m"] = fnum(out["ret_3m"] - s3) if out["ret_3m"] is not None and s3 is not None else None
    both = pd.concat([p, spx], axis=1, keys=["a", "m"]).dropna()
    wk = both.resample("W-FRI").last().pct_change().dropna().tail(104)
    if len(wk) >= 30 and wk.m.var() > 0:
        out["beta"] = fnum(wk.a.cov(wk.m) / wk.m.var(), 2)
        out["r2"] = fnum(wk.a.corr(wk.m) ** 2, 2)
    return out


def tone_trend(m: dict | None) -> str:
    if not m or m.get("above_200d") is None or m.get("ret_3m") is None:
        return "neutral"
    if m["above_200d"] and m["ret_3m"] > 0:
        return "good"
    if not m["above_200d"] and m["ret_3m"] < 0:
        return "bad"
    return "warn"


def risk_evidence(asset_m: pd.Series, spx_m: pd.Series, flag: pd.Series | None) -> dict | None:
    if flag is None:
        return None
    a = asset_m.reindex(flag.index)
    df = pd.DataFrame({"f": flag, "a_dip": worst_dip_12m(a), "s_dip": worst_dip_12m(spx_m.reindex(flag.index)),
                       "a_fwd": a.shift(-12) / a - 1}).dropna()
    on = df[df.f == 1]
    eps = episodes(df.f) if len(df) else 0
    if eps < 3 or len(on) < 6:
        return None
    return {"episodes": int(eps), "months": int(len(on)), "dip": fnum(on.a_dip.mean()), "spx_dip": fnum(on.s_dip.mean()),
            "rose": fnum((on.a_fwd > 0).mean()), "rose_other": fnum((df[df.f == 0].a_fwd > 0).mean())}


def ticker_view(item: dict, prices: pd.DataFrame, spx: pd.Series, fund: dict, econ: dict, market: dict,
                flag: pd.Series | None, spx_m: pd.Series) -> dict | None:
    s = item["s"]
    if s not in prices.columns:
        return None
    m = metrics(prices[s], spx)
    if not m:
        return None
    kind = item["k"]
    etf = GICS_ETF.get(item.get("sector") or "")
    if (item.get("sub") or "").lower().startswith("semiconductor"):
        etf = "SMH"
    sector_m = metrics(prices[etf], spx) if etf and etf in prices.columns else None
    f = fund.get(s) if kind == "stock" else None
    lvl = market.get("level", "")
    layers = [
        {"name": "Economy", "text": econ["regime"], "tone": {"good": "good", "warn": "warn", "bad": "bad"}.get(econ["tone"], "neutral")},
        {"name": "Market (S&P 500)", "text": f"{lvl}" + (", no re-entry signal yet" if lvl in ("High risk", "Elevated risk") else ""),
         "tone": {"High risk": "bad", "Elevated risk": "warn", "Normal": "good", "Recovery window": "good"}.get(lvl, "neutral")},
    ]
    if kind == "stock":
        if sector_m:
            ttxt = ("Above" if sector_m.get("above_200d") else "Below") + " its 200-day average"
            rel = sector_m.get("rel_3m")
            layers.append({"name": f"Sector ({item.get('sector') or etf})",
                           "text": ttxt + (f"; {rel * 100:+.0f}% vs the S&P over 3 months" if rel is not None else ""),
                           "tone": tone_trend(sector_m)})
        else:
            layers.append({"name": "Sector", "text": "No sector data for this ticker", "tone": "neutral"})
    trend_tone = tone_trend(m)
    layers.append({"name": "Own trend", "tone": trend_tone,
                   "text": (f"{m['from_high'] * 100:+.0f}% from its 52-week high; " if m.get("from_high") is not None else "")
                   + ("above" if m.get("above_200d") else "below") + " its 200-day average"
                   + (f"; 3-month return {m['ret_3m'] * 100:+.0f}%" if m.get("ret_3m") is not None else "")})
    if kind == "stock":
        if f and f.get("growth") is not None:
            g, mg = f["growth"], f.get("margin")
            ftone = "good" if g >= 0.10 and (mg or 0) >= 0.10 else ("bad" if g < 0 or (mg is not None and mg < 0) else "warn")
            layers.append({"name": f"Fundamentals ({f['period']}, SEC)", "tone": ftone,
                           "text": f"Revenue {g * 100:+.0f}% year on year" + (f"; net margin {mg * 100:.0f}%" if mg is not None else "")})
        else:
            layers.append({"name": "Fundamentals (SEC)", "text": "Not available for this ticker", "tone": "neutral"})
    else:
        b = m.get("beta")
        layers.append({"name": "Sensitivity", "tone": "warn" if (b or 1) > 1.1 else "neutral",
                       "text": f"Moves about {b:.1f}× the S&P 500" if b is not None else "Sensitivity not available"})

    tones = [l["tone"] for l in layers]
    nb, ng = tones.count("bad"), tones.count("good")
    head = ("Headwinds stacked", "bad") if nb >= 3 else (("Tailwinds aligned", "good") if ng >= 3 else ("Mixed signals", "warn"))
    if kind != "stock" and lvl == "High risk" and (m.get("beta") or 1) > 1.1:
        head = ("High risk, amplified", "bad")
    short = {"good": ("Momentum with it", "good"), "bad": ("Momentum against it", "bad")}.get(trend_tone, ("No clear momentum", "neutral"))
    aligned = nb >= len(layers) - 1 or ng >= len(layers) - 1
    horizons = [
        {"h": "1–3 months", "text": short[0], "tone": short[1], "tag": "Research-backed"},
        {"h": "6–12 months", "text": "Layers aligned, untested" if aligned else "Unclear: layers disagree", "tone": "neutral",
         "tag": "No model edge"},
        {"h": "3–5 years", "text": "Set by the business" if kind == "stock" else "Depends on valuation", "tone": "warn",
         "tag": "Not modelled yet"},
    ]
    flips = []
    if m.get("above_200d") is False and m.get("sma200"):
        flips.append({"name": "Back above its 200-day average", "text": f"Around {m['sma200']:,.2f}; turns the trend layer positive."})
    elif m.get("above_200d"):
        flips.append({"name": "A close below its 200-day average", "text": f"Around {m['sma200']:,.2f}; turns the trend layer negative."})
    flips.append({"name": "A market re-entry trigger fires", "text": "VIX month-end ≥ 30 or credit spread +0.5 pts in 6 months."})
    if kind == "stock":
        flips.append({"name": "Next results", "text": "Revenue growth and margins confirm or break the fundamentals layer."})
    asset_m = prices[s].dropna().resample("ME").last()
    ev = risk_evidence(asset_m, spx_m, flag)
    b = m.get("beta")
    reading = (f"{'Near term' if kind == 'stock' else 'Short term'}, {nb} of {len(layers)} layers lean against it and {ng} support it"
               + (f"; it tends to move about {b:.1f}× the market" if b is not None else "") + ". "
               + ("Longer term, the business decides." if kind == "stock" else "Over years, valuation matters most."))
    return {"symbol": s, "name": item.get("n") or s, "kind": kind, "sector": item.get("sector"), "sector_etf": etf,
            "metrics": m, "sector_metrics": sector_m, "fundamentals": f, "layers": layers, "headline": head[0],
            "headline_tone": head[1], "horizons": horizons, "flips": flips, "risk_evidence": ev, "reading": reading}


# --------------------------------------------------------------------------- main

def build(store: Path, out: Path, skip_network: bool = False) -> dict:
    t0 = time.time()
    P = load_panel(store)
    D = load_daily(store, cols=("px:^GSPC", "px:^VIX", "fred:DGS10", "fred:DCOILBRENTEU", "fred:BAMLH0A0HYM2", "fred:DFEDTARU"))
    res = json.loads((store / "engine_results.json").read_text())
    econ = economy_block(P, res)
    market = {"criteria": res.get("criteria"), "call": res.get("call"), "analog": res.get("analog"), "combo": res.get("combo_now"),
              "track": {k: v for k, v in (res.get("track") or {}).items() if not k.startswith("  ")}}
    crit = res.get("criteria") or {}
    summary = {"built": datetime.now(timezone.utc).isoformat(timespec="minutes"), "data_to": res.get("data_to"),
               "strip": strip_block(D, P), "economy": econ, "market": market}

    universe = load_universe(store) if not skip_network else [
        {"s": s, "n": n, "k": k, "sector": None, "sub": None} for s, (n, k) in BASE_ASSETS.items()]
    syms = [u["s"] for u in universe]
    base = fa.load(store / "yahoo_prices")
    have = set(base.columns) if base is not None else set()
    need = [s for s in syms if s not in have]
    prices = base if base is not None else pd.DataFrame()
    if need and not skip_network:
        extra, _, secs, _ = fa.fetch_yahoo(need, store, False, name="universe_prices")
        prices = prices.join(extra[[c for c in extra.columns if c not in prices.columns]], how="outer")
        print(f"  prices for {len(need)} extra tickers in {secs:.0f}s")
    spx = prices["^GSPC"] if "^GSPC" in prices else prices.get("SPY")
    fund, fund_note = ({}, "skipped") if skip_network else sec_fundamentals(store)
    summary["fundamentals_source"] = fund_note
    health = []
    labels = {c["label"] for c in summary["strip"]}
    for need, what in (("S&P 500", "S&P 500 daily prices (Yahoo)"), ("US 10-year", "FRED daily rates"),
                       ("HY spread", "credit spreads (FRED)")):
        if need not in labels:
            health.append(what)
    if all(p["detail"] == "n/a" for p in econ["pillars"]):
        health.append("FRED economic data (jobs, inflation, policy)")
    if res.get("data_to") and pd.Timestamp(res["data_to"]) < pd.Timestamp.today() - pd.Timedelta(days=45):
        health.append(f"recent S&P 500 history (latest month used: {res['data_to']})")
    if not fund:
        health.append(f"company fundamentals ({fund_note})")
    summary["health"] = health
    for h in health:
        print(f"::warning::Missing in this run: {h}")

    S, spx_m = build_signals(P)
    on_def = [d["signal"] for d in crit.get("defensive", []) if d.get("now")]
    flag = pd.concat([S[n][0] for n in on_def if n in S], axis=1).min(axis=1) if on_def else None

    pub = out / "public"
    tdir = pub / "data" / "tickers"
    if out.exists():
        shutil.rmtree(out)
    tdir.mkdir(parents=True)
    listed, n = [], 0
    for item in universe:
        v = ticker_view(item, prices, spx, fund, econ, crit, flag, spx_m)
        if v is None:
            continue
        (tdir / f"{file_name(item['s'])}.json").write_text(json.dumps(finite(v), separators=(",", ":"), default=str, allow_nan=False))
        listed.append({"s": item["s"], "n": item.get("n"), "k": item["k"], "f": file_name(item["s"])})
        n += 1
    (pub / "data" / "summary.json").write_text(json.dumps(finite(summary), separators=(",", ":"), default=str, allow_nan=False))
    (pub / "data" / "universe.json").write_text(json.dumps(listed, separators=(",", ":")))
    site = ROOT / "site"
    for f in site.iterdir():
        if f.is_file():
            shutil.copy2(f, pub / f.name)
    if (site / "functions").exists():
        shutil.copytree(site / "functions", out / "functions")
    print(f"  site built: {n} tickers, summary, app ({time.time() - t0:.0f}s) -> {pub}")
    return {"tickers": n, "summary": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(ROOT / "data" / "store"))
    ap.add_argument("--out", default=str(ROOT / "deploy"))
    ap.add_argument("--offline", action="store_true", help="no network: base assets only, no SEC (for testing)")
    a = ap.parse_args(argv)
    build(Path(a.store), Path(a.out), skip_network=a.offline)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
