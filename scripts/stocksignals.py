#!/usr/bin/env python3
"""
stocksignals.py - single-stock signal lab. Eleven published signals, tested on companies while they were in the S&P 500.

Each month-end t, among members at t, the top 10% by a signal (equal weight) is held for month t+1 (or, for slow signals,
three overlapping monthly cohorts, Jegadeesh-Titman style). Cost 0.10% for each name replaced. Benchmarks: the average
member (equal weight), so the result is stock picking and not the S&P's large-company tilt; the S&P for context; and a
risk-adjusted alpha (regression on the average member), so quiet stocks are not penalised for moving less.
Verdict: Solid = alpha positive in both periods (2002-2012, 2013-now), t >= 3 in the recent one, and the top 10% beating
the bottom 10% recently; Not proven = recent alpha positive with t >= 2; Fails otherwise.

Signals (sign chosen from the papers before testing; higher score = buy):
  momentum      12-month return skipping the last month                      Jegadeesh & Titman (1993)
  resmom        residual (beyond the market) 12-1 return / its volatility    Blitz, Huij & Martens (2011)
  high52        price / 52-week high                                         George & Hwang (2004)
  frog          momentum gained in small steady steps                        Da, Gurun & Warachka (2014)
  indmom        6-month return of the stock's industry (SIC 2-digit)         Moskowitz & Grinblatt (1999)
  reversal      minus last month's return                                    Jegadeesh (1990)
  rev_nonews    reversal, only stocks without an earnings announcement      Da, Liu & Schaumburg (2014)
  rev_industry  minus last month's return relative to its industry          Da, Liu & Schaumburg (2014); Hameed & Mian (2015)
  low_max       minus the largest daily return of the last month             Bali, Cakici & Whitelaw (2011)
  low_vol       minus 12-month daily volatility                              Blitz & van Vliet (2007); Ang et al. (2006)
  seasonal      average return in the coming calendar month, past 1-10 years Heston & Sadka (2008)
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import earnings as ea

COST = 0.0010
SPLIT = ("2002-01-01", "2012-12-31", "2013-01-01")
SIGNALS = {
    "momentum": ("Momentum (12 months, skipping the last)", "Jegadeesh & Titman 1993", (1, 3)),
    "resmom": ("Residual momentum", "Blitz, Huij & Martens 2011", (1, 3)),
    "high52": ("Near the 52-week high", "George & Hwang 2004", (1, 3)),
    "frog": ("Momentum in small steady steps", "Da, Gurun & Warachka 2014", (1, 3)),
    "indmom": ("Industry momentum (6 months)", "Moskowitz & Grinblatt 1999", (1, 3)),
    "reversal": ("Short-term reversal (last month's losers)", "Jegadeesh 1990", (1,)),
    "rev_nonews": ("Reversal without earnings news", "Da, Liu & Schaumburg 2014", (1,)),
    "rev_industry": ("Reversal within the industry", "Da, Liu & Schaumburg 2014; Hameed & Mian 2015", (1,)),
    "low_max": ("Low MAX (no recent lottery-like spike)", "Bali, Cakici & Whitelaw 2011", (1,)),
    "low_vol": ("Low volatility (12 months)", "Blitz & van Vliet 2007", (1, 3)),
    "seasonal": ("Same-calendar-month seasonality", "Heston & Sadka 2008", (1,)),
}


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().values)


def membership_matrix(mem: pd.DataFrame | None, months: pd.DatetimeIndex, tickers: list) -> pd.DataFrame:
    if mem is None:
        return pd.DataFrame(True, index=months, columns=tickers)
    M = pd.DataFrame(False, index=months, columns=tickers)
    cols = set(tickers)
    for r in mem.itertuples():
        if r.ticker in cols:
            M.loc[(months >= r.start) & (months <= r.end), r.ticker] = True
    return M


def sic_codes(store: Path, tickers: list) -> dict:
    out = {}
    try:
        ciks = ea.ticker_ciks(store)
    except Exception:
        return out
    for t in tickers:
        cik = ciks.get(t)
        p = store / "sec" / "subs" / f"{cik}.json"
        if cik is None or not p.exists():
            continue
        try:
            sic = str(json.loads(p.read_text()).get("sic") or "")
        except Exception:
            continue
        if len(sic) >= 2 and sic[:2].isdigit():
            out[t] = sic[:2]
    return out


def _roll_cov(x: pd.DataFrame, y: pd.Series, w: int, mp: int) -> pd.DataFrame:
    mx, my = x.rolling(w, min_periods=mp).mean(), y.rolling(w, min_periods=mp).mean()
    mxy = x.mul(y, axis=0).rolling(w, min_periods=mp).mean()
    return mxy - mx.mul(my, axis=0)


def build_signals(C: pd.DataFrame, M: pd.DataFrame, event_months: pd.DataFrame | None = None, sic: dict | None = None) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Scores (month-end x ticker, members only) for every signal; the next month's return; the S&P's monthly return."""
    stocks = [c for c in C.columns if c != "SPY"]
    R = C[stocks].pct_change(fill_method=None)
    me = month_ends(C.index)
    Pm = C.reindex(me)
    Rm_all = Pm.pct_change(fill_method=None)
    spy = Rm_all["SPY"]
    Rm = Rm_all[stocks]
    mask = M.reindex(index=me, columns=stocks).fillna(False).astype(bool)
    S = {}
    P = Pm[stocks]
    S["momentum"] = P.shift(1) / P.shift(12) - 1
    # residual momentum: beta from 36 months, residual return over months t-11..t-1 scaled by its volatility
    beta = _roll_cov(Rm, spy, 36, 24).div(spy.rolling(36, min_periods=24).var(ddof=0), axis=0)
    x, s = Rm.shift(1), spy.shift(1)
    res_sum = x.rolling(11, min_periods=9).sum() - beta.mul(s.rolling(11, min_periods=9).sum(), axis=0)
    var_e = (x.rolling(11, min_periods=9).var(ddof=0) + beta.pow(2).mul(s.rolling(11, min_periods=9).var(ddof=0), axis=0)
             - 2 * beta * _roll_cov(x, s, 11, 9))
    S["resmom"] = res_sum / np.sqrt(var_e.where(var_e > 0))
    S["high52"] = (C[stocks] / C[stocks].rolling(252, min_periods=200).max()).reindex(me)
    pos = (R > 0).astype(float).where(R.notna()).rolling(231, min_periods=180).mean().shift(21).reindex(me)
    neg = (R < 0).astype(float).where(R.notna()).rolling(231, min_periods=180).mean().shift(21).reindex(me)
    S["_id"] = np.sign(S["momentum"]) * (neg - pos)
    S["reversal"] = -Rm
    S["rev_nonews"] = (-Rm).where(~event_months.reindex(index=me, columns=stocks).fillna(False).astype(bool)) if event_months is not None else -Rm
    S["low_max"] = -R.rolling(21, min_periods=15).max().reindex(me)
    S["low_vol"] = -R.rolling(252, min_periods=200).std().reindex(me)
    seas = sum(Rm.shift(12 * k - 1).fillna(0) for k in range(1, 11))
    cnt = sum(Rm.shift(12 * k - 1).notna().astype(int) for k in range(1, 11))
    S["seasonal"] = (seas / cnt.replace(0, np.nan)).where(cnt >= 5)
    if sic:
        groups = pd.Series({t: sic.get(t) for t in stocks})
        has = groups.notna()
        r_m = Rm.where(mask)
        six = (P / P.shift(6) - 1).where(mask)
        ind_r = r_m.loc[:, has].T.groupby(groups[has]).transform("mean").T
        ind_n = r_m.loc[:, has].notna().T.groupby(groups[has]).transform("sum").T
        ind6 = six.loc[:, has].T.groupby(groups[has]).transform("mean").T
        ind6_n = six.loc[:, has].notna().T.groupby(groups[has]).transform("sum").T
        S["rev_industry"] = (-(r_m.loc[:, has] - ind_r)).where(ind_n >= 3).reindex(columns=stocks)
        S["indmom"] = ind6.where(ind6_n >= 3).reindex(columns=stocks)
    out = {k: v.where(mask) for k, v in S.items()}
    out["frog"] = out["momentum"].rank(axis=1, pct=True) - out.pop("_id").rank(axis=1, pct=True)
    F1 = Rm.shift(-1)
    return out, F1.where(mask), spy, mask


def portfolios(score: pd.DataFrame, F1: pd.DataFrame, mask: pd.DataFrame, hold: int = 1, q: float = 0.1) -> pd.DataFrame:
    """Monthly returns (indexed by formation month) of the top and bottom 10%, the average member, turnover and cost."""
    rk = score.rank(axis=1, pct=True)
    n_ok = score.notna().sum(axis=1)
    ok = n_ok >= 30
    top = (rk >= 1 - q).mul(ok, axis=0).astype(bool)
    bot = (rk <= q).mul(ok, axis=0).astype(bool)
    Fz = F1.fillna(0.0)                                          # a member with no next price counts as flat (rare)
    def cohort_ret(sel_shifted):
        n = sel_shifted.sum(axis=1)
        return (Fz * sel_shifted).sum(axis=1) / n.where(n > 0)
    r_top = sum(cohort_ret(top.shift(k, fill_value=False)) for k in range(hold)) / hold
    r_bot = sum(cohort_ret(bot.shift(k, fill_value=False)) for k in range(hold)) / hold
    prev = top.shift(hold, fill_value=False)
    n_top = top.sum(axis=1)
    turnover = 1 - (top & prev).sum(axis=1) / n_top.where(n_top > 0)
    cost = turnover.fillna(1.0) * COST / hold
    avg = (Fz.where(mask)).sum(axis=1) / mask.sum(axis=1).where(mask.sum(axis=1) > 0)
    out = pd.DataFrame({"top": r_top - cost, "bottom": r_bot - cost, "avg": avg, "turnover": turnover, "n_top": n_top})
    first = top.any(axis=1)
    started = first.cumsum() >= hold                              # wait until every cohort exists
    return out[started & out["top"].notna() & out["avg"].notna()]


def ols_alpha(y: pd.Series, x: pd.Series) -> dict | None:
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(d) < 36:
        return None
    X = np.column_stack([np.ones(len(d)), d.x.values])
    b, *_ = np.linalg.lstsq(X, d.y.values, rcond=None)
    e = d.y.values - X @ b
    s2 = (e @ e) / (len(d) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"alpha": float(b[0]), "t": float(b[0] / math.sqrt(cov[0, 0])), "beta": float(b[1]), "months": int(len(d))}


def mean_t(x: pd.Series) -> dict | None:
    x = x.dropna()
    if len(x) < 24 or x.std() == 0:
        return None
    return {"mean": float(x.mean()), "t": float(x.mean() / (x.std() / math.sqrt(len(x)))), "months": int(len(x))}


def evaluate(pf: pd.DataFrame, spy: pd.Series, broken: pd.Series) -> dict:
    when = pd.Series(pf.index + pd.offsets.MonthEnd(1), index=pf.index)       # the month the return is earned
    early = (when >= SPLIT[0]) & (when <= SPLIT[1])
    late = when >= SPLIT[2]
    spy_next = spy.shift(-1).reindex(pf.index)
    res = {}
    for name, sel in (("early", early), ("late", late)):
        p = pf[sel]
        res[name] = {"alpha": ols_alpha(p.top, p.avg), "alpha_bottom": ols_alpha(p.bottom, p.avg),
                     "vs_average": mean_t(p.top - p.avg), "vs_sp500": mean_t(p.top - spy_next[sel]),
                     "top_minus_bottom": mean_t(p.top - p.bottom), "turnover": float(p.turnover.mean()) if len(p) else None}
    st = broken.reindex(pf.index)
    res["by_trend"] = {"trend_intact": mean_t((pf.top - pf.avg)[late & (st == False)]),  # noqa: E712
                       "trend_broken": mean_t((pf.top - pf.avg)[late & (st == True)])}  # noqa: E712
    yr = (pf.top - pf.avg).groupby(when.dt.year).sum()
    res["by_year"] = {int(k): float(v) for k, v in yr.items()}
    a_e, a_l = (res["early"]["alpha"] or {}), (res["late"]["alpha"] or {})
    tb = res["late"]["top_minus_bottom"] or {}
    if not a_l:
        v = "Too little data"
    elif a_l["alpha"] > 0 and a_l["t"] >= 3 and a_e.get("alpha", -1) > 0 and tb.get("mean", -1) > 0:
        v = "Solid"
    elif a_l["alpha"] > 0 and a_l["t"] >= 2:
        v = "Not proven"
    else:
        v = "Fails"
    res["verdict"] = v
    return res


def run(store: Path, tickers: list, membership: pd.DataFrame | None = None) -> dict:
    t0 = time.time()
    stocks = [t for t in tickers if not str(t).startswith("^") and t not in ea.ETFS]
    px = ea.daily_prices(store, sorted(set(stocks) | {"SPY"}))
    C = px["close"]
    C = C[[c for c in C.columns if c in set(stocks) | {"SPY"}]].sort_index()
    C = C[C.index >= "1995-01-01"]
    me = month_ends(C.index)
    M = membership_matrix(membership, me, [c for c in C.columns if c != "SPY"])
    # months with an earnings announcement (reaction day inside the month), for the no-news reversal
    ev_m = None
    try:
        ann = ea.announcements(store, stocks, budget_s=0)
        if len(ann):
            ev = ea.classify(ann, C.index)
            per = pd.PeriodIndex(ev.reaction, freq="M")
            mp = pd.Series(me, index=me.to_period("M"))
            ev = ev.assign(me=mp.reindex(per).values).dropna(subset=["me"])
            ev_m = pd.crosstab(ev.me, ev.ticker).gt(0).reindex(index=me, columns=M.columns, fill_value=False)
    except Exception as err:
        print(f"  (no earnings dates: {err})")
    sic = sic_codes(store, list(M.columns))
    S, F1, spy, mask = build_signals(C, M, ev_m, sic)
    spy_m = C["SPY"].reindex(me)
    broken = spy_m < spy_m.rolling(10).mean()
    results, series = [], {}
    for key, (label, paper, holds) in SIGNALS.items():
        if key not in S:
            results.append({"key": key, "label": label, "paper": paper, "hold": holds[0], "verdict": "No data"})
            continue
        for h in holds:
            pf = portfolios(S[key], F1, mask, hold=h)
            if len(pf) < 60:
                results.append({"key": key, "label": label, "paper": paper, "hold": h, "verdict": "Too little data"})
                continue
            r = evaluate(pf, spy, broken)
            series[(key, h)] = (pf.top - pf.avg)[pf.index >= SPLIT[2]]
            results.append({"key": key, "label": label, "paper": paper, "hold": h, **r})
    base = series.get(("momentum", 1))
    for r in results:
        s_ = series.get((r["key"], r["hold"]))
        if base is not None and s_ is not None:
            j = pd.concat([s_, base], axis=1).dropna()
            r["overlap_with_momentum"] = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 24 else None
    solid = [r for r in results if r["verdict"] == "Solid"]
    summary = (f"{len(solid)} of {len(results)} signal tests passed every test after costs." if solid else
               f"None of {len(results)} signal tests passed every test after costs.")
    if solid:
        b = max(solid, key=lambda r: r["late"]["alpha"]["alpha"])
        summary += f" Strongest: {b['label']} ({b['hold']}-month hold), {b['late']['alpha']['alpha'] * 1200:+.1f}% a year risk-adjusted (t = {b['late']['alpha']['t']:.1f})."
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "seconds": round(time.time() - t0),
            "stocks_with_prices": int(C.shape[1] - 1), "members_per_month": float(mask.sum(axis=1)[mask.sum(axis=1) > 0].mean()),
            "industry_codes": len(sic), "summary": summary, "results": results,
            "method": ("Each month-end, the top 10% of companies in the S&P 500 at that time by the signal (equal weight), held 1 month "
                       "(or 3 overlapping monthly cohorts), 0.10% cost per name replaced. Alpha = risk-adjusted return against the "
                       "average member (regression), per month. Periods 2002-2012 and 2013-now (by the month the return is earned). "
                       "Solid = alpha positive in both periods, t >= 3 recently, and the top 10% beating the bottom 10% recently.")}
