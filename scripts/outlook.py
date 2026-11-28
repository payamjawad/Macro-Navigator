#!/usr/bin/env python3
"""
outlook.py - the executive panel: what to expect, the option history supports now, and what to watch.

Expectations are historical distributions for situations like today's, never forecasts:
  3 months   - forward returns in months with the same S&P 500 trend-rule state (the tested short-term lever)
  12 months  - the base rate over all months (no model has a proven 12-month edge)
  5 years    - S&P 500 total return a year after months with a similar valuation (CAPE 30+, the tested long-run signal);
               for the Nasdaq-100 the base rate since 1986 (no long valuation history)
Options (same trend state and a similar distance from the 12-month high): invest now, stage in over 3 months (a third
each month-end), or wait for the trend rule (stay in cash until a month-end close above the 10-month average).
Rules for the best-supported option are fixed in advance (see best_option). Single stocks: the market's 12-month median
scaled by the stock's sensitivity (beta) with its own historical range, flagged as rough because single-stock forecasts
failed every 2026 test; plus the position size that carries the same risk as 10% in an S&P 500 fund.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def q(x) -> dict | None:
    x = pd.Series(x).dropna()
    if len(x) < 12:
        return None
    return {"n": int(len(x)), "median": float(x.median()), "p10": float(x.quantile(0.10)), "p90": float(x.quantile(0.90)),
            "up": float((x > 0).mean()), "mean": float(x.mean())}


def month_end(daily: pd.Series, complete_only: bool = True) -> pd.Series:
    d = daily.dropna()
    m = d.resample("ME").last()
    if complete_only and len(d):
        last = d.index[-1]
        if last != last + pd.offsets.BMonthEnd(0):
            m = m.iloc[:-1]                                              # the current month is not finished
    return m


def dd_bucket(d: float) -> str:
    return "near its high" if d > -0.05 else ("5-10% below its high" if d > -0.10 else "10%+ below its high")


def options_backtest(px_m: pd.Series, in_trend: pd.Series, sel: pd.Series) -> dict:
    """12-month outcome of three ways to put money in, from each selected month-end."""
    v = px_m.values
    it = in_trend.reindex(px_m.index).values
    idx = np.where(sel.reindex(px_m.index).fillna(False).values)[0]
    lump, stage, wait = [], [], []
    for i in idx:
        if i + 12 >= len(v):
            continue
        lump.append(v[i + 12] / v[i] - 1)
        stage.append(np.mean([v[i + 12] / v[i + k] for k in range(3)]) - 1)
        if it[i] == 1:
            wait.append(v[i + 12] / v[i] - 1)
        else:
            s = next((j for j in range(i + 1, i + 13) if it[j] == 1), None)
            wait.append(v[i + 12] / v[s] - 1 if s is not None else 0.0)    # cash (interest ignored) if the trend never turned
    lump, stage, wait = map(np.array, (lump, stage, wait))
    out = {"months": int(len(lump))}
    for name, arr in (("invest_now", lump), ("stage_3_months", stage), ("wait_for_trend", wait)):
        if len(arr):
            out[name] = {"median": float(np.median(arr)), "p10": float(np.percentile(arr, 10)), "worst": float(arr.min()),
                         "beat_invest_now": float(np.mean(arr > lump)) if name != "invest_now" else None}
    return out


def best_option(trend_now: dict | None, trend_live: dict | None, dd_now: float, watch_level: str | None) -> dict:
    """Pre-declared rules, from the tested evidence: the trend rule for timing, dips for patience, recession watch for preparation."""
    if trend_now and not trend_now.get("in_stocks"):
        o = {"option": "Wait for the trend rule, or stage in", "why": "The S&P 500 closed its last month below its 10-month average. "
             "Out of stocks, the rule has roughly halved the worst falls in both halves of history."}
    elif trend_live and not trend_live.get("in_stocks"):
        o = {"option": "Stage in over 3 months", "why": "The trend rule is still in stocks, but today's price is below the average, so it "
             "could switch at the month-end close."}
    elif dd_now <= -0.10:
        o = {"option": "Invest normally, in stages if a further fall would worry you", "why": "The trend rule is in stocks. Dips this deep "
             "usually recovered within months, though often after a further fall."}
    else:
        o = {"option": "Invest normally", "why": "The trend rule is in stocks and no tested warning is on; staging has usually cost a "
             "little return in rising markets."}
    if watch_level in ("Watch", "High"):
        o["why"] += " Recession watch is raised: keep a cash buffer and avoid borrowing to invest."
    return o


def market_outlook(name: str, px_d: pd.Series, spx_d: pd.Series, P: pd.DataFrame | None, trend_now: dict | None,
                   trend_live: dict | None, watch_level: str | None, since: str) -> dict | None:
    px_m = month_end(px_d)
    px_m = px_m[px_m.index >= since]
    spx_m = month_end(spx_d)
    if len(px_m) < 120 or len(spx_m) < 130:
        return None
    in_trend = (spx_m >= spx_m.rolling(10).mean()).astype(float).where(spx_m.rolling(10).mean().notna())
    state = bool(trend_now["in_stocks"]) if trend_now else bool(in_trend.dropna().iloc[-1] == 1)
    same = in_trend.reindex(px_m.index) == (1.0 if state else 0.0)
    f3, f12 = px_m.shift(-3) / px_m - 1, px_m.shift(-12) / px_m - 1
    live = px_d.dropna()
    dd_now = float(live.iloc[-1] / live.iloc[-252:].max() - 1) if len(live) > 252 else 0.0
    out = {"index": name, "since": since[:4], "trend_state": "in stocks" if state else "in T-bills", "from_high": dd_now,
           "3 months": q(f3[same]) | {"basis": "Tested: months with the same trend-rule state"} if q(f3[same]) else None,
           "12 months": q(f12) | {"basis": "Base rate: all months (no proven 12-month edge)"} if q(f12) else None,
           "12 months same trend": q(f12[same])}
    # 5 years a year: valuation for the S&P 500 (Shiller total return), base rate for the Nasdaq-100
    five = None
    if name == "S&P 500" and P is not None and {"shiller:P", "shiller:D", "shiller:CAPE"} <= set(P.columns):
        sp = P["shiller:P"].dropna()
        dy = (P["shiller:D"] / 12 / P["shiller:P"]).reindex(sp.index).ffill().fillna(0)
        tr = (1 + sp.pct_change().fillna(0) + dy.shift(1).fillna(0)).cumprod()
        cape = P["shiller:CAPE"].reindex(sp.index)
        lv = cape.last_valid_index()
        if lv is not None:
            cape = cape.copy()
            after = cape.index > lv
            cape[after] = cape[lv] * (sp[after] / sp[lv])
        cape_now = float(cape.dropna().iloc[-1]) if cape.notna().any() else None
        r5 = (tr.shift(-60) / tr) ** (1 / 5) - 1
        thr = 30 if (cape_now or 0) >= 30 else (cape_now or 0) * 0.85
        lo, hi = (30, 1e9) if (cape_now or 0) >= 30 else (thr, (cape_now or 0) * 1.15)
        sel = (cape >= lo) & (cape <= hi) & r5.notna()
        st = q(r5[sel])
        if st:
            years = sorted({d.year for d in r5[sel].index})
            spans, start = [], None
            for y in years:
                if start is None:
                    start = prev = y
                elif y != prev + 1:
                    spans.append((start, prev)); start = y
                prev = y
            spans.append((start, prev))
            five = st | {"basis": f"Tested: 5 years after months with CAPE {'30 or more' if cape_now >= 30 else 'near today'}",
                         "cape_now": cape_now, "periods": ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans),
                         "all_months": q(r5)}
    if five is None:
        r5 = (px_m.shift(-60) / px_m) ** (1 / 5) - 1
        st = q(r5)
        if st:
            five = st | {"basis": f"Base rate since {since[:4]} (no long valuation history for this index)"}
    out["5 years a year"] = five
    # options in situations like today's: same trend state and a similar distance from the 12-month high
    ddm = px_m / px_m.rolling(12, min_periods=6).max() - 1
    b_now = dd_bucket(dd_now)
    sel = same & (ddm.map(lambda d: dd_bucket(d) if pd.notna(d) else None) == b_now)
    note = f"same trend state, {b_now}"
    if sel.sum() < 30:
        sel, note = same, "same trend state"
    out["options"] = options_backtest(px_m, in_trend, sel) | {"situation": note}
    out["best"] = best_option(trend_now, trend_live, dd_now, watch_level)
    return out


def stock_outlook(p_d: pd.Series, spx_d: pd.Series, beta: float | None, market_12m_median: float | None, rf: float) -> dict | None:
    p = p_d.dropna()
    s = spx_d.dropna()
    if len(p) < 300:
        return None
    p20 = p[p.index >= p.index[-1] - pd.Timedelta(days=int(365.25 * 20))]
    pm = p20.resample("ME").last()
    own12, own3 = q(pm.shift(-12) / pm - 1), q(pm.shift(-3) / pm - 1)
    r = p.pct_change().iloc[-252:]
    rs = s.pct_change().reindex(r.index)
    vol, vol_s = float(r.std() * np.sqrt(252)), float(rs.std() * np.sqrt(252)) if rs.notna().sum() > 100 else None
    central = (rf + beta * (market_12m_median - rf)) if beta is not None and market_12m_median is not None else None
    return {"12 months": {"central": central, "p10": (own12 or {}).get("p10"), "p90": (own12 or {}).get("p90"), "up": (own12 or {}).get("up"),
                          "years": round((p20.index[-1] - p20.index[0]).days / 365.25),
                          "basis": "Rough: the market's typical year scaled by its sensitivity, with its own historical range. Single-stock "
                                   "forecasts failed every 2026 test."},
            "volatility": vol, "spx_volatility": vol_s,
            "same_risk_as_10pct_sp500": (0.10 * vol_s / vol) if vol and vol_s else None,
            "review_fall_3m": (own3 or {}).get("p10")}
