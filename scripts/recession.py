#!/usr/bin/env python3
"""
recession.py - recession watch and market situation for the app (pipeline version of the 2026 dip-check study).

Gauge (rules fixed in advance, equal weights): 9 early warnings and 4 confirmers; Low = at most one warning and no confirmer,
Watch = two warnings, High = three or more warnings or any confirmer. Built from figures as first published where the
store has ALFRED first releases (fetch_all saves them weekly), otherwise from revised data one month after the period.

Findings it reports, from the 2026 studies: the gauge read Watch or High before every recession since 1945 and a recession
began within 12 months in 55% of Watch months versus 21% of Low months; but buying 10% dips by gauge level did not beat
buying every dip. So the market situation shows how all similar past dips turned out, with the gauge split as context.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import fetch_all as fa

HOLDS = {"1 month": 21, "3 months": 63, "6 months": 126, "12 months": 252}
TH = {"curve_memory": 18, "rate_shock": 1.5, "permits": -0.20, "claims_warn": 1.20, "credit_pct": 0.80, "trucks": -0.20,
      "hours": -0.5, "sentiment": -15, "claims_conf": 1.30, "ip_conf": -0.01}


# --------------------------------------------------------------------------- inputs

def _series(store: Path, sid: str) -> pd.Series:
    try:
        d = fa.load(store / "fred" / sid)
    except Exception:
        return pd.Series(dtype=float)
    if d is None:
        return pd.Series(dtype=float)
    s = d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    return s


def _first(store: Path, sid: str) -> pd.DataFrame:
    try:
        d = fa.load(store / "fred_first_release" / sid)
    except Exception:
        return pd.DataFrame()
    if d is None or not len(d) or "value" not in d:
        return pd.DataFrame()
    d = d.copy()
    d.index = pd.to_datetime(d.index)
    d["published"] = pd.to_datetime(d["first_release"])
    return d[["value", "published"]].dropna().sort_index()


def monthly(s: pd.Series, how: str = "mean") -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    g = s.groupby(s.index.to_period("M"))
    return g.mean() if how == "mean" else g.last()


def asof_months(sig: pd.Series, pub: pd.Series, months: pd.PeriodIndex) -> pd.Series:
    df = pd.DataFrame({"sig": sig, "pub": pub}).dropna(subset=["pub"]).sort_values("pub")
    if df.empty:
        return pd.Series(np.nan, index=months)
    ends = months.to_timestamp(how="end").normalize().values.astype("datetime64[ns]")
    pos = np.searchsorted(df["pub"].values.astype("datetime64[ns]"), ends, side="right") - 1
    latest = pd.Series(df.index.values).cummax().values
    sig_by = df["sig"].groupby(level=0).first()
    return pd.Series([np.nan if p < 0 else sig_by.get(latest[p], np.nan) for p in pos], index=months, dtype=float)


def first_print_signal(fp: pd.DataFrame, rule, revised: pd.Series, months, lag_days=35) -> tuple[pd.Series, float]:
    if len(fp):
        v, pub = fp["value"].copy(), fp["published"].copy()
        stale = (pub - v.index) > pd.Timedelta(days=150)                 # observations from before the archive began
        pub = pd.Series(np.where(stale.values, (v.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)).values, pub.values), index=v.index)
        share = float(1 - stale.mean())
    elif len(revised):
        v = revised
        pub = pd.Series(v.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days), index=v.index)
        share = 0.0
    else:
        return pd.Series(np.nan, index=months), 0.0
    s = rule(v.sort_index())
    return asof_months(s.astype(float).where(s.notna()), pub.reindex(s.index), months), share


def signals(store: Path, P: pd.DataFrame, D: pd.DataFrame, months: pd.PeriodIndex, th: dict = TH):
    on = lambda c, a: c.astype(float).where(a)  # noqa: E731
    m = lambda sid: monthly(_series(store, sid))  # noqa: E731
    gs10 = m("GS10").combine_first(monthly(P["shiller:GS10"].dropna()) if "shiller:GS10" in P else pd.Series(dtype=float)).reindex(months)
    tb3, gs1 = m("TB3MS").reindex(months), m("GS1").reindex(months)
    dgs2 = monthly(D["fred:DGS2"].dropna()).reindex(months) if "fred:DGS2" in D else pd.Series(np.nan, index=months)
    baa, sent = m("BAA").reindex(months), m("UMCSENT").reindex(months)
    spread = gs10 - tb3
    inv = (spread < 0).astype(float).where(spread.notna()).rolling(int(th["curve_memory"]), min_periods=1).max()
    ntfs = (2 * dgs2 - gs1) - tb3
    credit = baa - gs10
    rank = credit.rolling(120, min_periods=60).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    W, share = {}, {}
    W["Yield curve inverted in the last 18 months"] = on(inv == 1, inv.notna())
    W["Near-term forward spread below zero"] = on(ntfs < 0, ntfs.notna())
    W["Short rate up 1.5+ points in a year"] = on(tb3.diff(12) >= th["rate_shock"], tb3.diff(12).notna())
    W["Credit spread in the top 20% of 10 years"] = on(rank >= th["credit_pct"], rank.notna())
    W["Consumer sentiment 15+ below its 12-month high"] = on(sent - sent.rolling(12).max() <= th["sentiment"], sent.rolling(12).max().notna())
    for name, sid, rule in (("Building permits down 20%+ on a year ago", "PERMIT", lambda v: v.pct_change(12, fill_method=None) <= th["permits"]),
                            ("Heavy truck sales 20%+ below their 12-month high", "HTRUCKSSAAR", lambda v: v / v.rolling(12, min_periods=6).max() - 1 <= th["trucks"]),
                            ("Manufacturing hours down 0.5+ in 6 months", "AWHMAN", lambda v: v.diff(6) <= th["hours"])):
        W[name], share[sid] = first_print_signal(_first(store, sid), rule, _series(store, sid), months)
    C = {}
    cl = D["fred:ICSA"].dropna() if "fred:ICSA" in D else _series(store, "ICSA")
    if len(cl):
        a4 = cl.rolling(4).mean()
        ratio = a4 / a4.rolling(52, min_periods=26).min()
        pub = pd.Series(cl.index + pd.Timedelta(days=5), index=cl.index)
        W["Jobless claims 20%+ above their 52-week low"] = asof_months((ratio >= th["claims_warn"]).astype(float).where(ratio.notna()), pub, months)
        C["Jobless claims 30%+ above their 52-week low"] = asof_months((ratio >= th["claims_conf"]).astype(float).where(ratio.notna()), pub, months)
    sahm = m("SAHMREALTIME").reindex(months)
    un = m("UNRATE").reindex(months).shift(1)
    un3 = un.rolling(3).mean()
    sahm = sahm.combine_first(un3 - un3.rolling(12).min())
    C["Sahm rule 0.5+ (real time)"] = on(sahm >= 0.5, sahm.notna())
    C["Payrolls down over 3 months (first report)"], share["PAYEMS"] = first_print_signal(_first(store, "PAYEMS"), lambda v: v.diff(3) < 0, _series(store, "PAYEMS"), months)
    C["Industrial production down 1%+ over 6 months (first report)"], share["INDPRO"] = first_print_signal(
        _first(store, "INDPRO"), lambda v: v.pct_change(6, fill_method=None) <= th["ip_conf"], _series(store, "INDPRO"), months)
    return W, C, share, {"spread_3m": spread, "gs10": gs10, "tb3": tb3, "dgs2": dgs2}


def gauge(W: dict, C: dict) -> pd.DataFrame:
    w, c = pd.DataFrame(W), pd.DataFrame(C)
    nw, nc = w.sum(axis=1, min_count=1), c.sum(axis=1, min_count=1)
    lvl = pd.Series("Low", index=w.index, dtype=object)
    lvl[nw >= 2] = "Watch"
    lvl[(nw >= 3) | (nc >= 1)] = "High"
    lvl[nw.isna() & nc.isna()] = None
    return pd.DataFrame({"warnings": nw, "warnings_avail": w.notna().sum(axis=1), "confirmers": nc, "confirmers_avail": c.notna().sum(axis=1), "level": lvl})


# --------------------------------------------------------------------------- recession watch

def recession_watch(store: Path, P: pd.DataFrame, D: pd.DataFrame) -> tuple[dict | None, pd.DataFrame | None]:
    rec_raw = monthly(_series(store, "USREC"), "last")
    if rec_raw.empty and "fred:USREC" in P:
        rec_raw = monthly(P["fred:USREC"].dropna(), "last")
    if rec_raw.empty:
        return None, None
    months = pd.period_range("1900-01", rec_raw.index.max(), freq="M")
    rec = rec_raw.round().reindex(months).fillna(0).astype(int)
    W, C, share, curve = signals(store, P, D, months)
    G = gauge(W, C)
    G = G[G.level.notna()]
    if G.empty:
        return None, None
    starts = pd.PeriodIndex(rec[(rec == 1) & (rec.shift(1, fill_value=0) == 0)].index, freq="M")
    def within(mo, k):
        return bool(((starts > mo) & (starts <= mo + k)).any())
    known = G.index[(rec.reindex(G.index) == 0) & (G.index <= G.index[-1] - 12)]
    odds = {}
    for lvl in ("Low", "Watch", "High"):
        mm = [x for x in known if G.level.get(x) == lvl]
        if mm:
            odds[lvl] = {"months": len(mm), "p12": float(np.mean([within(x, 12) for x in mm]))}
    record = []
    for s in starts:
        if s < pd.Period("1945-01", "M"):
            continue
        w = G.loc[s - 12: s - 1]
        hit = w.index[w.level.isin(["Watch", "High"])]
        record.append({"recession": str(s), "warned": bool(len(hit)), "lead": int((s - hit[0]).n) if len(hit) else None})
    last = G.index[-1]
    sig_now = [{"name": n, "kind": "warning", "on": None if pd.isna(W[n].get(last)) else bool(W[n].get(last) == 1)} for n in W]
    sig_now += [{"name": n, "kind": "confirmer", "on": None if pd.isna(C[n].get(last)) else bool(C[n].get(last) == 1)} for n in C]
    # yield curve phase and odds by spread level (10-year minus 3-month bill)
    sp = curve["spread_3m"].dropna()
    phase = None
    if len(sp):
        now = float(sp.iloc[-1])
        inv_months = sp[sp < 0]
        last_inv = inv_months.index[-1] if len(inv_months) else None
        months_since = int((sp.index[-1] - last_inv).n) if last_inv is not None else None
        if now < 0:
            state = "Inverted"
        elif last_inv is not None and months_since <= 18:
            state = "Re-steepening after an inversion"
        else:
            state = "Normal"
        exp = sp[(rec.reindex(sp.index) == 0) & (sp.index <= sp.index[-1] - 24)]
        buckets = []
        for lo, hi, lab in ((-99, 0, "below 0"), (0, 0.5, "0 to 0.5"), (0.5, 1.5, "0.5 to 1.5"), (1.5, 99, "above 1.5")):
            sel = exp[(exp >= lo) & (exp < hi)]
            if len(sel):
                buckets.append({"bucket": lab, "months": int(len(sel)), "p24": float(np.mean([within(x, 24) for x in sel.index])),
                                "now": lo <= now < hi})
        s2 = (curve["gs10"] - curve["dgs2"]).dropna()
        phase = {"spread_10y_3m": now, "spread_10y_2y": float(s2.iloc[-1]) if len(s2) else None, "month": str(sp.index[-1]), "state": state,
                 "last_inverted": str(last_inv) if last_inv is not None else None, "months_since_inversion": months_since, "odds": buckets}
    watch = {"month": str(last), "level": G.level.iloc[-1],
             "warnings_on": int(G.warnings.iloc[-1]) if pd.notna(G.warnings.iloc[-1]) else 0, "warnings_avail": int(G.warnings_avail.iloc[-1]),
             "confirmers_on": int(G.confirmers.iloc[-1]) if pd.notna(G.confirmers.iloc[-1]) else 0, "confirmers_avail": int(G.confirmers_avail.iloc[-1]),
             "signals": sig_now, "odds": odds, "base_p12": float(np.mean([within(x, 12) for x in known])) if len(known) else None,
             "record": record, "warned": sum(r["warned"] for r in record), "recessions": len(record), "curve": phase,
             "first_report_share": {k: round(v, 2) for k, v in share.items()},
             "note": ("A forecast of the economy, not a market-timing tool: in the 2026 dip study, buying dips by this gauge did not beat buying "
                      "every dip, because markets usually bottom during recessions.")}
    return watch, G


# --------------------------------------------------------------------------- market situation

def dip_events(px: pd.Series, level: float, G: pd.DataFrame | None, rearm: float = 0.05):
    """First close at or below `level` (a drawdown from the 252-day high), re-armed after recovering by `rearm`."""
    p = px.dropna()
    dd = p / p.rolling(252, min_periods=60).max() - 1
    out, armed, vals = [], True, p.values
    for i, (d, v) in enumerate(dd.items()):
        if np.isnan(v):
            continue
        if armed and v <= level:
            row = {"i": i, "date": d}
            mo = pd.Period(d, "M") - 1
            row["level"] = (G.level.get(mo) if G is not None and mo in G.index else None)
            for k, h in HOLDS.items():
                row[k] = float(vals[i + h] / vals[i] - 1) if i + h < len(p) else None
            row["worst6"] = float(vals[i: i + 127].min() / vals[i] - 1) if i + 126 < len(p) else None
            out.append(row)
            armed = False
        if not armed and v > level + rearm:
            armed = True
    return out


def _stats(x):
    x = [v for v in x if v is not None]
    if not x:
        return None
    a = np.array(x)
    return {"n": int(len(a)), "up": float((a > 0).mean()), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def situation(D: pd.DataFrame, G: pd.DataFrame | None, level_now: str | None) -> dict:
    out = {}
    for name, col, since in (("S&P 500", "px:^GSPC", "1928-01-01"), ("Nasdaq-100", "px:^NDX", "1986-01-01"), ("Nasdaq Composite", "px:^IXIC", "1971-02-01")):
        if col not in D or D[col].notna().sum() < 600:
            continue
        p = D[col].dropna()
        p = p[p.index >= since]
        hi = p.iloc[-252:].max()
        dd_now = float(p.iloc[-1] / hi - 1)
        item = {"index": name, "price": float(p.iloc[-1]), "date": str(p.index[-1].date()), "from_high": dd_now, "since": since[:4]}
        if dd_now > -0.03:
            item["state"] = "Near its high"
            base = {k: _stats(list((p.shift(-h) / p - 1).dropna().values)) for k, h in HOLDS.items()}
            item["any_day"] = base
        else:
            lvl = max(-0.40, round(dd_now * 100) / 100)                       # today's fall, to the nearest percent
            ev = dip_events(p, lvl, G)
            item["state"] = f"{abs(lvl) * 100:.0f}% below its high"
            item["level_used"] = lvl
            item["episodes"] = len(ev)
            item["outcomes"] = {k: _stats([e[k] for e in ev]) for k in HOLDS}
            item["worst6"] = _stats([e["worst6"] for e in ev])
            for g, sel in (("low", ["Low"]), ("watch_high", ["Watch", "High"])):
                sub = [e for e in ev if e["level"] in sel]
                item[g] = {k: _stats([e[k] for e in sub]) for k in ("3 months", "6 months", "12 months")} | {"n": len(sub)}
            item["recent"] = [{"date": str(e["date"].date()), "level": e["level"], **{k: e[k] for k in HOLDS}} for e in ev[-6:]][::-1]
            item["event_dates"] = [str(e["date"].date()) for e in ev]
        out[name] = item
    return {"indices": out, "gauge_now": level_now,
            "note": "Past moments when the index first fell this far below its 52-week high, and what happened next. Price returns; recession "
                    "readings did not change these odds much in the 2026 tests."}


def stock_after_dips(prices: pd.DataFrame, sym: str, index_col: pd.Series, event_dates: list) -> dict | None:
    """How a stock did after the index's past falls like today's, against the index over the same windows."""
    if sym not in prices or not event_dates:
        return None
    s = prices[sym].dropna()
    ix = index_col.dropna()
    rows = []
    for d in event_dates:
        d = pd.Timestamp(d)
        if d < s.index[0] or d > s.index[-1]:
            continue
        i, j = s.index.searchsorted(d), ix.index.searchsorted(d)
        if i >= len(s) or j >= len(ix):
            continue
        r = {}
        for k, h in (("3 months", 63), ("6 months", 126), ("12 months", 252)):
            r[k] = float(s.iloc[i + h] / s.iloc[i] - 1) if i + h < len(s) else None
            r[k + " index"] = float(ix.iloc[j + h] / ix.iloc[j] - 1) if j + h < len(ix) else None
        r["worst6"] = float(s.iloc[i: i + 127].min() / s.iloc[i] - 1) if i + 126 < len(s) else None
        rows.append(r)
    done = [r for r in rows if r.get("6 months") is not None]
    if len(done) < 3:
        return {"episodes": len(rows), "too_few": True}
    out = {"episodes": len(done)}
    for k in ("3 months", "6 months", "12 months"):
        out[k] = _stats([r[k] for r in rows])
        out[k + " index"] = _stats([r[k + " index"] for r in rows])
        both = [(r[k], r[k + " index"]) for r in rows if r[k] is not None and r[k + " index"] is not None]
        out[k + " beat"] = float(np.mean([a > b for a, b in both])) if both else None
    out["worst6"] = _stats([r["worst6"] for r in rows])
    return out
