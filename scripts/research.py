#!/usr/bin/env python3
"""
research.py - the research lab. Tests candidate direction signals with the protocol that guards against curve fitting:

  1. walk-forward: every forecast uses only outcomes already known at that date;
  2. split periods: judged separately on an early period and on a later one (the later one never informed the design);
  3. placebo: the same test on the signal scrambled in time (circular shifts keep its persistence) shows the luck level.

Verdict: Solid      = positive skill in both periods and beyond the placebo's 95th percentile on the later period
         Not proven = positive on the later period only, or not beyond the placebo
         Fails      = no skill on the later period
Runs at most weekly (results cached in the store); build_site.py calls it. Signals come from the data the pipeline
already downloads. References: Neely, Rapach, Tu & Zhou (2014); Bollerslev, Tauchen & Zhou (2009);
Goyal, Welch & Zafirov (2024); Faber (2007); Moskowitz, Ooi & Pedersen (2012).
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_check import col, load_panel  # noqa: E402
from engine import latest, load_daily  # noqa: E402

PLACEBO_RUNS = 60
MIN_CELL = 24


def walk(x: np.ndarray, y: np.ndarray, h: int, minn: int = MIN_CELL):
    """Walk-forward conditional frequency of a gain, given the signal's state; falls back to the base rate."""
    n = len(x)
    counts: dict = {}
    tot = [0, 0.0]
    p = np.full(n, np.nan)
    base = np.full(n, np.nan)
    for i in range(n):
        j = i - h
        if j >= 0 and not np.isnan(y[j]):
            tot[0] += 1
            tot[1] += y[j]
            if not np.isnan(x[j]):
                c = counts.setdefault(x[j], [0, 0.0])
                c[0] += 1
                c[1] += y[j]
        if tot[0] < 120:
            continue
        base[i] = tot[1] / tot[0]
        c = counts.get(x[i]) if not np.isnan(x[i]) else None
        p[i] = c[1] / c[0] if c and c[0] >= minn else base[i]
    return p, base


def skill(p, base, y, mask):
    m = mask & ~np.isnan(p) & ~np.isnan(y) & ~np.isnan(base)
    if m.sum() < 60:
        return None
    b0 = np.mean((base[m] - y[m]) ** 2)
    return float(1 - np.mean((p[m] - y[m]) ** 2) / b0) if b0 > 0 else None


def when_on(x, y, r, mask):
    m = mask & ~np.isnan(x) & ~np.isnan(y)
    on, off = m & (x == 1), m & (x == 0)
    if on.sum() < 6 or off.sum() < 6:
        return None
    return {"rose_on": float(y[on].mean()), "rose_off": float(y[off].mean()),
            "avg_on": float(np.nanmean(r[on])), "avg_off": float(np.nanmean(r[off])), "months_on": int(on.sum())}


def evaluate(x: pd.Series, fwd: pd.Series, h: int, early, late, rng, binary=True):
    xv = x.values.astype(float)
    y = (fwd.values > 0).astype(float)
    y[np.isnan(fwd.values)] = np.nan
    idx = x.index
    m_early = (idx >= early[0]) & (idx <= early[1])
    m_late = (idx >= late[0]) & (idx <= late[1])
    p, b = walk(xv, y, h)
    s_early, s_late = skill(p, b, y, m_early), skill(p, b, y, m_late)
    null = []
    for _ in range(PLACEBO_RUNS):
        k = int(rng.integers(120, max(121, len(xv) - 120)))
        pn, bn = walk(np.roll(xv, k), y, h)
        s = skill(pn, bn, y, m_late)
        if s is not None:
            null.append(s)
    p95 = float(np.percentile(null, 95)) if null else None
    if s_late is None:
        verdict = "Too little data"
    elif s_late <= 0:
        verdict = "Fails"
    elif s_early is not None and s_early > 0 and p95 is not None and s_late > p95:
        verdict = "Solid"
    else:
        verdict = "Not proven"
    out = {"early_skill": s_early, "late_skill": s_late, "placebo_p95": p95, "verdict": verdict,
           "early": f"{pd.Timestamp(early[0]).year}–{pd.Timestamp(early[1]).year}",
           "late": f"{pd.Timestamp(late[0]).year}–{min(pd.Timestamp(late[1]), idx[-1]).year}"}
    if binary:
        out["late_when_on"] = when_on(xv, y, fwd.values, m_late)
    return out


def candidates(P: pd.DataFrame, store: Path):
    spx = col(P, "px:^GSPC", "shiller:P").dropna()
    P = P.loc[spx.index[0]:spx.index[-1]]
    spx = spx.reindex(P.index)
    trend = (spx < spx.rolling(10).mean()).astype(float).where(spx.rolling(10).mean().notna())
    tb = latest(P, "fred:TB3MS")
    gs10 = latest(P, "fred:GS10", "shiller:GS10")
    baa, aaa = latest(P, "fred:BAA"), latest(P, "fred:AAA")
    cpi = col(P, "fred:CPIAUCSL")
    infl = (cpi / cpi.shift(12) - 1).shift(1)
    q = baa - aaa
    q_rank = q.expanding(240).apply(lambda s: (s[:-1] < s[-1]).mean() if len(s) > 1 else np.nan, raw=True)
    C = {
        "Trend: S&P below its 10-month average": (trend, "Faber 2007; Moskowitz, Ooi & Pedersen 2012", True),
        "T-bill rate up 1 point or more in 12 months": (((tb - tb.shift(12)) >= 1.0).astype(float).where(tb.shift(12).notna()), "Goyal, Welch & Zafirov 2024", True),
        "Yield curve inverted (10-year below 3-month bill)": (((gs10 - tb) < 0).astype(float).where((gs10 - tb).notna()), "Estrella & Mishkin 1998", True),
        "Credit spread wide (top 20% of its history)": ((q_rank >= 0.8).astype(float).where(q_rank.notna()), "Gilchrist & Zakrajsek 2012", True),
        "Credit spread up 0.5 points in 6 months": ((((baa - gs10).diff(6)) >= 0.5).astype(float).where((baa - gs10).diff(6).notna()), "Gilchrist & Zakrajsek 2012", True),
        "Inflation 4% or more": ((infl >= 0.04).astype(float).where(infl.notna()), "Neville et al. 2021", True),
        "Trend + credit spread (combined)": ((trend * 2 + (q_rank >= 0.8)).where(trend.notna() & q_rank.notna()), "Neely, Rapach, Tu & Zhou 2014", False),
        "Trend + inflation (combined)": ((trend * 2 + (infl >= 0.04)).where(trend.notna() & infl.notna()), "Neely et al. 2014; Neville et al. 2021", False),
    }
    # volatility risk premium: implied minus realised variance (needs daily S&P and VIX, 1990-)
    D = load_daily(store, cols=("px:^GSPC", "px:^VIX"))
    vrp_x = None
    if "px:^GSPC" in D and "px:^VIX" in D and D["px:^GSPC"].notna().sum() > 2000:
        r = np.log(D["px:^GSPC"]).diff()
        rv = (r ** 2).resample("ME").sum()
        iv = (D["px:^VIX"].resample("ME").last() / 100) ** 2 / 12
        vrp = (iv - rv).reindex(P.index)
        vrank = vrp.expanding(60).apply(lambda s: (s[:-1] < s[-1]).mean() if len(s) > 1 else np.nan, raw=True)
        vrp_x = (vrank >= 2 / 3).astype(float).where(vrank.notna())
    return spx, C, vrp_x


ETFS = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC", "SMH", "GLD", "TLT"}


def pooled_transfer(store: Path, P: pd.DataFrame, rng) -> list:
    """Does the S&P 500's trend state predict single stocks? Pooled across every stock in the daily list.
    Note: today's members only (survivorship), which lifts both states alike; the test compares the states."""
    import fetch_all as fa
    up = fa.load(store / "universe_prices")
    if up is None or up.empty:
        return []
    stocks = [c for c in up.columns if not str(c).startswith("^") and c not in ETFS]
    if len(stocks) < 30:
        return []
    spx = col(P, "px:^GSPC", "shiller:P")
    ma = spx.rolling(10).mean()
    state = (spx < ma).astype(float).where(ma.notna())
    M = up[stocks].resample("ME").last().reindex(P.index)
    idx = P.index
    out = []
    for h in (3, 12):
        F = (M.shift(-h) / M - 1).values
        Y = np.where(np.isnan(F), np.nan, (F > 0).astype(float))
        nups, nvalid = np.nansum(Y, axis=1), np.sum(~np.isnan(Y), axis=1)

        def score(st):
            counts = {0.0: [0.0, 0.0], 1.0: [0.0, 0.0]}
            tot = [0.0, 0.0]
            se, se0 = {"e": 0.0, "l": 0.0}, {"e": 0.0, "l": 0.0}
            for i in range(len(idx)):
                j = i - h
                if j >= 0 and nvalid[j] > 0:
                    tot[0] += nvalid[j]
                    tot[1] += nups[j]
                    if not np.isnan(st[j]):
                        counts[st[j]][0] += nvalid[j]
                        counts[st[j]][1] += nups[j]
                if tot[0] < 3000 or nvalid[i] == 0 or np.isnan(st[i]):
                    continue
                base = tot[1] / tot[0]
                c = counts[st[i]]
                p = c[1] / c[0] if c[0] >= 1500 else base
                y = Y[i][~np.isnan(Y[i])]
                period = "e" if idx[i] <= pd.Timestamp("2009-12-31") else "l"
                se[period] += np.sum((p - y) ** 2)
                se0[period] += np.sum((base - y) ** 2)
            return {k: (1 - se[k] / se0[k]) if se0[k] > 0 else None for k in se}
        real = score(state.values)
        null = [score(np.roll(state.values, int(rng.integers(60, len(idx) - 60))))["l"] for _ in range(PLACEBO_RUNS // 2)]
        null = [x for x in null if x is not None]
        p95 = float(np.percentile(null, 95)) if null else None
        e, l = real["e"], real["l"]
        verdict = ("Too little data" if l is None else "Fails" if l <= 0 else
                   "Solid" if (e is not None and e > 0 and p95 is not None and l > p95) else "Not proven")
        out.append({"signal": "S&P trend rule applied to single stocks (pooled)", "horizon": h,
                    "reference": f"{len(stocks)} stocks; Faber 2007 applied via the market", "early_skill": e, "late_skill": l,
                    "placebo_p95": p95, "verdict": verdict, "early": "1995–2009", "late": f"2010–{idx[-1].year}"})
    return out


def run(store: Path) -> dict:
    t0 = time.time()
    P = load_panel(store)
    spx, C, vrp_x = candidates(P, store)
    rng = np.random.default_rng(11)
    last = spx.index[-1]
    results = []
    for name, (x, ref, binary) in C.items():
        for h in (3, 12):
            fwd = spx.shift(-h) / spx - 1
            results.append({"signal": name, "horizon": h, "reference": ref,
                            **evaluate(x, fwd, h, ("1950-01-31", "1989-12-31"), ("1990-01-31", last), rng, binary)})
    if vrp_x is not None:
        for h in (3, 12):
            fwd = spx.shift(-h) / spx - 1
            results.append({"signal": "Volatility risk premium high (VIX above realised volatility)", "horizon": h,
                            "reference": "Bollerslev, Tauchen & Zhou 2009",
                            **evaluate(vrp_x, fwd, h, ("1991-01-31", "2007-12-31"), ("2008-01-31", last), rng, True)})
    try:
        results += pooled_transfer(store, P, rng)
    except Exception as err:
        print(f"  pooled single-stock test skipped ({err.__class__.__name__})")
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "seconds": round(time.time() - t0),
           "protocol": ("Walk-forward; judged separately on an early and a later period; placebo = the same test on the signal "
                        "scrambled in time. Solid = positive in both periods and beyond the placebo's 95th percentile."),
           "results": results}
    return out


def run_if_stale(store: Path, max_age_days: float = 6.5, force: bool = False) -> dict | None:
    path = store / "research_results.json"
    if path.exists() and not force and (time.time() - path.stat().st_mtime) / 86400 < max_age_days:
        return json.loads(path.read_text())
    try:
        res = run(store)
    except Exception as err:                       # the lab must never break the daily build
        print(f"  research lab skipped ({err.__class__.__name__}: {err})")
        return json.loads(path.read_text()) if path.exists() else None
    from engine import finite
    path.write_text(json.dumps(finite(res), indent=1, allow_nan=False))
    print(f"  research lab: {len(res['results'])} tests in {res['seconds']}s")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(HERE.parent / "data" / "store"))
    a = ap.parse_args()
    r = run_if_stale(Path(a.store), force=True)
    for x in (r or {}).get("results", []):
        print(f"{x['verdict']:<14} {x['horizon']:>2}m  early {x['early_skill'] if x['early_skill'] is None else round(x['early_skill'], 3)!s:>7}  "
              f"late {x['late_skill'] if x['late_skill'] is None else round(x['late_skill'], 3)!s:>7}  {x['signal']}")
