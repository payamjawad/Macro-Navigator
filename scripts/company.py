#!/usr/bin/env python3
"""
company.py - earnings and company financials for stock pages. Information, not signals: the 2026 studies found no
tradable edge in entering around earnings or in these figures for S&P 500 stocks after costs (see TESTED_REJECTED).

Earnings: SEC 8-K Item 2.02 filing times (earnings.announcements). Reaction day = same day for before-open releases,
next session for after-close ones. Move = the stock's close-to-close return that day minus the S&P 500's.
Next date: last year's same-quarter reaction day + 52 weeks (in the 2026 study the expected date landed within 5
trading days of the real one in 94% of announcements); falls back to the last date + 13 weeks.
Financials: SEC company facts, first-reported values (ideafactory.parse_companyfacts), refreshed on a rolling basis.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import earnings as ea
import ideafactory as ifa

HOLD_THROUGH = ("Tested (2026, S&P 500 members 2001-now): holding through earnings added risk but no extra return "
                "since 2013, and buying after the news showed no edge from 1 minute to 60 days after costs.")

TESTED_REJECTED = [
    {"study": "Entering after earnings", "tests": 244, "period": "2001-2026 (minutes from 2016)",
     "result": "No edge after costs from 1 minute to 60 days. The first run's apparent 20-day edge (+0.65% a trade) was "
               "survivorship bias: +0.12% once companies counted only while in the S&P 500."},
    {"study": "Holding into earnings (announcement premium)", "tests": 23, "period": "2004-2026",
     "result": "Worked until about 2010 (+0.36% a trade 2001-2012), nothing since 2013 (+0.01%)."},
    {"study": "11 published single-stock signals", "tests": 17, "period": "2002-2026",
     "result": "All fail since 2013 (momentum, residual momentum, 52-week high, industry momentum, three reversal types, "
               "low MAX, low volatility, seasonality). MAX and seasonality faded after publication; low volatility "
               "lowers risk but not return."},
    {"study": "About 600 home-made price, volume, event and fundamental ideas", "tests": 600, "period": "1996-2026",
     "result": "None survived the holdout (1 qualified in discovery, then +0.3% a year). Broad themes carry over weakly "
               "(rank correlation +0.22 price, +0.41 fundamentals) but too weakly to trade after costs."},
]


# --------------------------------------------------------------------------- earnings

def earnings_block(store: Path, tickers: list, closes: pd.DataFrame, spx: pd.Series, budget_s: float = 600,
                   today: pd.Timestamp | None = None) -> dict:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    try:
        ann = ea.announcements(store, tickers, budget_s=budget_s)
    except Exception as err:
        print(f"  earnings dates unavailable ({err.__class__.__name__})")
        return {}
    if ann.empty:
        return {}
    days = closes.index
    ev = ea.classify(ann, days)
    s = spx.reindex(days).ffill().values
    out = {}
    for tk, g in ev.groupby("ticker"):
        if tk not in closes:
            continue
        c = closes[tk].values
        rows = []
        for rp, ts, timing in zip(g.rpos.values, g.ts.values, g.timing.values):
            if rp < 1 or rp >= len(days) or np.isnan(c[rp]) or np.isnan(c[rp - 1]):
                continue
            raw = c[rp] / c[rp - 1] - 1
            rows.append({"date": str(pd.Timestamp(days[rp]).date()), "released": str(pd.Timestamp(ts))[:16],
                         "timing": timing, "move": float(raw - (s[rp] / s[rp - 1] - 1)), "raw": float(raw)})
        if not rows:
            continue
        react = pd.to_datetime([r["date"] for r in rows])
        last8 = rows[-8:]
        nxt, how = next_date(react, today)
        out[tk] = {"count": len(rows), "since": rows[0]["date"][:4], "recent": last8[::-1],
                   "typical_move": float(np.mean([abs(r["move"]) for r in last8])),
                   "up_share": float(np.mean([r["move"] > 0 for r in last8])),
                   "biggest": max(last8, key=lambda r: abs(r["move"])),
                   "next": {"date": str(nxt.date()) if nxt is not None else None, "days": int((nxt - today).days) if nxt is not None else None,
                            "method": how},
                   "note": HOLD_THROUGH}
    return out


def next_date(react: pd.DatetimeIndex, today: pd.Timestamp):
    """Estimated next reaction day from last year's same-quarter date + 52 weeks."""
    if not len(react):
        return None, None
    last = react.max()
    # the upcoming announcement's counterpart a year earlier sits about 13 weeks after the last one, minus 52 weeks
    target = last + pd.Timedelta(days=91) - pd.Timedelta(days=364)
    near = react[(react >= target - pd.Timedelta(days=25)) & (react <= target + pd.Timedelta(days=25))]
    if len(near):
        guess = near[np.argmin(np.abs((near - target).days))] + pd.Timedelta(days=364)
        how = "last year's date + 52 weeks (within 5 trading days in 94% of past cases)"
    else:
        guess = last + pd.Timedelta(days=91)
        how = "last date + 13 weeks (rough: no matching date last year)"
    while guess < today - pd.Timedelta(days=3):               # an estimate already in the past means it is due
        guess = guess + pd.Timedelta(days=91)
    if guess.weekday() >= 5:
        guess = guess + pd.Timedelta(days=7 - guess.weekday())
    return guess, how


# --------------------------------------------------------------------------- financials

def load_facts(store: Path, tickers: list, budget_s: float = 600, refresh_per_run: int = 120) -> dict:
    """First-reported facts per ticker. New companies first, then the oldest caches (over 7 days), within the budget."""
    t0 = time.time()
    ciks = ea.ticker_ciks(store)
    folder = store / "sec" / "facts"
    folder.mkdir(parents=True, exist_ok=True)
    todo_new, todo_old, out = [], [], {}
    for tk in tickers:
        cik = ciks.get(tk)
        if cik is None:
            continue
        p = folder / f"{cik}.pkl"
        if p.exists():
            try:
                out[tk] = pd.read_pickle(p)
            except Exception:
                todo_new.append((tk, cik, p))
                continue
            age = (time.time() - p.stat().st_mtime) / 86400
            if age > 7:
                todo_old.append((age, tk, cik, p))
        else:
            todo_new.append((tk, cik, p))
    todo = todo_new + [(tk, cik, p) for _, tk, cik, p in sorted(todo_old, reverse=True)[:refresh_per_run]]
    n = 0
    for tk, cik, p in todo:
        if time.time() - t0 > budget_s:
            break
        try:
            r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=ea.SEC_UA, timeout=60)
            time.sleep(0.12)
            if r.status_code == 404:
                df = pd.DataFrame(columns=["key", "end", "filed", "val"])
            else:
                r.raise_for_status()
                df = ifa.parse_companyfacts(r.json())
        except Exception:
            continue
        df.to_pickle(p)
        out[tk] = df
        n += 1
    if n:
        print(f"  company financials: {n} downloaded, {len(out)} companies available")
    return out


def _latest(df: pd.DataFrame, key: str, today: pd.Timestamp):
    f = df[(df.key == key) & (df.filed <= today)]
    if f.empty:
        return None, None, None
    r = f.sort_values("end").iloc[-1]
    return float(r.val), pd.Timestamp(r.end), pd.Timestamp(r.filed)


def _year_ago(df: pd.DataFrame, key: str, end: pd.Timestamp, today: pd.Timestamp):
    f = df[(df.key == key) & (df.filed <= today)]
    if f.empty or end is None:
        return None
    d = (f.end - (end - pd.Timedelta(days=365))).abs()
    i = d.idxmin()
    return float(f.loc[i, "val"]) if d.loc[i] <= pd.Timedelta(days=20) else None


def summarize(df: pd.DataFrame, today: pd.Timestamp | None = None) -> dict | None:
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    if df is None or df.empty:
        return None
    def growth(a, b):
        return None if a is None or b is None or b == 0 else (a - b) / abs(b)
    rev, q_end, q_filed = _latest(df, "revenue", today)
    if q_end is not None and (today - q_end).days > 550:
        rev, q_end, q_filed = None, None, None
    def q(key):
        v, e, _ = _latest(df, key, today)
        return v if (e is not None and q_end is not None and e == q_end) else None
    gp, oi, ni, rnd = q("gross_profit"), q("op_income"), q("net_income"), q("rnd")
    rev_y = _year_ago(df, "revenue", q_end, today)
    gp_y, oi_y, ni_y = (_year_ago(df, k, q_end, today) for k in ("gross_profit", "op_income", "net_income"))
    ratio = lambda a, b: None if a is None or not b else a / b  # noqa: E731
    out = {"quarter_end": str(q_end.date()) if q_end is not None else None, "filed": str(q_filed.date()) if q_filed is not None else None,
           "revenue": rev, "revenue_growth": growth(rev, rev_y),
           "gross_margin": ratio(gp, rev), "gross_margin_chg": (ratio(gp, rev) - ratio(gp_y, rev_y)) if ratio(gp, rev) is not None and ratio(gp_y, rev_y) is not None else None,
           "op_margin": ratio(oi, rev), "op_margin_chg": (ratio(oi, rev) - ratio(oi_y, rev_y)) if ratio(oi, rev) is not None and ratio(oi_y, rev_y) is not None else None,
           "net_income": ni, "net_income_growth": growth(ni, ni_y) if (ni_y is not None and ni_y > 0) else None,
           "net_income_year_ago": ni_y, "net_margin": ratio(ni, rev), "rnd_share": ratio(rnd, rev)}
    ra, a_end, _ = _latest(df, "revenue_a", today)
    def a(key):
        v, e, _ = _latest(df, key, today)
        return v if (e is not None and a_end is not None and e == a_end) else None
    cfo, capex, div, bb, nia = a("cfo_a"), a("capex_a"), a("dividends_a"), a("buybacks_a"), a("net_income_a")
    fcf = (cfo - capex) if cfo is not None and capex is not None else None
    out.update({"year_end": str(a_end.date()) if a_end is not None else None, "revenue_year": ra, "net_income_year": nia,
                "op_cash_flow_year": cfo, "capex_year": capex, "free_cash_flow_year": fcf, "fcf_margin": ratio(fcf, ra),
                "dividends_year": div, "buybacks_year": bb})
    assets, b_end, _ = _latest(df, "assets", today)
    def b(key):
        v, e, _ = _latest(df, key, today)
        return v if (e is not None and b_end is not None and abs((e - b_end).days) <= 5) else None
    liab, cash, eq = b("liabilities"), b("cash"), b("equity")
    liab_est = False
    if liab is None and assets is not None and eq is not None:
        liab, liab_est = assets - eq, True                     # many companies do not report a total: assets minus equity
    sh, s_end, _ = _latest(df, "shares", today)
    sh_y = _year_ago(df, "shares", s_end, today) if s_end is not None else None
    shc = (sh / sh_y - 1) if sh and sh_y else None
    if shc is not None and not (-0.3 < shc < 0.4):
        shc = None                                              # share splits, not issuance or buybacks
    out.update({"balance_date": str(b_end.date()) if b_end is not None else None, "assets": assets, "cash": cash,
                "liabilities": liab, "liabilities_estimated": liab_est, "equity": eq, "liabilities_share": ratio(liab, assets),
                "cash_share": ratio(cash, assets),
                "share_change": shc})
    if all(out.get(k) is None for k in ("revenue", "revenue_year", "assets")):
        return None
    return out


def financials_block(store: Path, tickers: list, budget_s: float = 600, today=None) -> dict:
    facts = load_facts(store, tickers, budget_s=budget_s)
    out = {}
    for tk, df in facts.items():
        try:
            s = summarize(df, today)
        except Exception:
            s = None
        out[tk] = s if s else {"unavailable": "No US-GAAP figures in SEC company facts (often a company filing under "
                                               "international rules, such as a foreign issuer)."}
    return out
