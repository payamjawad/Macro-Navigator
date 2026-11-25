#!/usr/bin/env python3
"""
composite.py - forward test of a pre-registered composite (paper only; no claims until there is enough data).

config/composite.json, written once on the Mac from the idea factory's discovery results, lists the features (and
directions) of the top 10% of ideas in discovery, restricted to ones the pipeline can compute (daily closes, earnings
dates, SEC figures). From the registration month on, each month-end the composite score (average of the features'
cross-sectional percentile ranks, signed) picks the top 10% of current S&P 500 members, equal weight, held one month,
0.10% per name replaced, and is scored against the average member. Everything is recomputed each run from data as of
each month-end, so nothing needs storing and nothing can be edited afterwards.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import ideafactory as ifa
import stocksignals as ss

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "config" / "composite.json"
COMPUTABLE = {"Returns", "Risk", "Market link", "Trend", "Consistency", "Seasonality", "Earnings", "Growth", "Profitability",
              "Quality", "Investment", "Capital", "Balance sheet", "Valuation"}
MIN_MONTHS_FOR_A_VIEW = 36


def completed_month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    me = ss.month_ends(idx)
    last = idx[-1]
    full = last == (last + pd.offsets.BMonthEnd(0))                     # the last data day is a month's last business day
    return me if full else me[me.to_period("M") < last.to_period("M")]


def forward_test(store: Path, closes: pd.DataFrame, members: list, facts: dict | None, ev: pd.DataFrame | None = None,
                 reg_path: Path = REG) -> dict:
    if not reg_path.exists():
        return {"registered": False, "note": "Not registered yet: run the one-off export on the Mac and upload config/composite.json."}
    reg = json.loads(reg_path.read_text())
    feats = [f for f in reg.get("features", []) if f.get("family") in COMPUTABLE]
    if not feats:
        return {"registered": False, "note": "The registration lists no computable features."}
    start = pd.Timestamp(reg["registered"])
    C = closes.copy()
    if "SPY" not in C and "^GSPC" in C:
        C["SPY"] = C["^GSPC"]
    names = [m for m in members if m in C.columns and m != "SPY"]
    C = C[names + ["SPY"]].sort_index()
    C = C[C.index >= C.index[-1] - pd.Timedelta(days=int(365.25 * 8))]  # 8 years is enough for every feature
    me = ss.month_ends(C.index)
    done = completed_month_ends(C.index)
    mask = pd.DataFrame(True, index=me, columns=names)
    groups = pd.Series(ss.sic_codes(store, names)) if names else None
    if groups is not None and groups.empty:
        groups = None
    want = {f["feature"]: f for f in feats}
    got = {}
    ones = pd.DataFrame(1.0, index=C.index, columns=C.columns)
    for key, fam, desc, F in ifa.price_features(C, C, ones, me, ev):
        for vkey, vdesc, G in ifa.variants_generic(key, desc, F, mask, groups, ifa.REL_LIST, ifa.CHG_LIST):
            if vkey in want:
                got[vkey] = G
    if facts:
        sub = {t: df for t, df in facts.items() if t in names and len(df)}
        if sub:
            for key, fam, desc, F in ifa.fundamental_features(sub, C, me):
                for vkey, vdesc, G in ifa.variants_generic(key, desc, F, mask, groups, ifa.F_REL, set()):
                    if vkey in want:
                        got[vkey] = G
    if not got:
        return {"registered": True, "since": str(start.date()), "features": len(feats), "computed": 0,
                "note": "None of the registered features could be computed from this run's data."}
    ranks = [want[k]["sign"] * G.rank(axis=1, pct=True) for k, G in got.items()]
    stack = np.stack([r.reindex(index=me, columns=names).values for r in ranks])
    n_ok = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        avg = np.nanmean(stack, axis=0)
    score = pd.DataFrame(avg, index=me, columns=names).where(n_ok >= max(1, len(ranks) // 2))
    Rm = C[names].reindex(me).pct_change(fill_method=None)
    F1 = Rm.shift(-1)
    months, prev = [], set()
    first = done[done >= start]
    for d in first:
        s = score.loc[d].dropna()
        if len(s) < 30:
            continue
        top = set(s[s.rank(pct=True) >= 0.9].index)
        if d == done[-1] or pd.isna(F1.loc[d]).all():
            prev = top
            continue
        r = F1.loc[d]
        turnover = 1 - len(top & prev) / len(top) if prev else 1.0
        months.append({"formed": str(d.date()), "picks": float(r[list(top)].fillna(0).mean()) - turnover * ss.COST,
                       "average": float(r[names].fillna(0).mean()), "turnover": float(turnover)})
        prev = top
    now_d = done[-1] if len(done) else None
    picks_now = []
    if now_d is not None and now_d in score.index:
        s = score.loc[now_d].dropna()
        if len(s) >= 30:
            picks_now = sorted(s[s.rank(pct=True) >= 0.9].index.tolist())
    ex = np.array([m["picks"] - m["average"] for m in months])
    out = {"registered": True, "since": str(start.date()), "features": len(feats), "computed": len(got),
           "rule": reg.get("rule"), "months": len(months), "history": months[-60:],
           "picks_formed": str(now_d.date()) if now_d is not None else None, "picks_now": picks_now,
           "min_months": MIN_MONTHS_FOR_A_VIEW}
    if len(ex):
        out.update({"mean_excess": float(ex.mean()), "cumulative_excess": float(np.prod(1 + ex) - 1), "beat_share": float((ex > 0).mean()),
                    "t": float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))) if len(ex) > 2 and ex.std(ddof=1) > 0 else None})
    out["status"] = ("Collecting data: no conclusion before " + str(MIN_MONTHS_FOR_A_VIEW) + " months."
                     if len(months) < MIN_MONTHS_FOR_A_VIEW else "Enough months to read, with the usual bar of t >= 2.")
    return out
