#!/usr/bin/env python3
"""
engine.py - Phase 2: a tested, directional read on the S&P 500 and US industries.

    python engine.py            # after fetch_all.py; writes data/store/engine_report.html and engine_results.json

Five parts, all built only from data that was available at each point in time:
  1. Horizon test: each signal at 3, 6, 12, 18 and 24 months (the yield curve's lag is long and variable).
  2. Signal roles from evidence: Defensive (worse returns after it fires, in both halves of history),
     Recovery (better returns after it fires), or Unproven.
  3. Historical analogs (Mulliner, Harvey, Xia, Fang & Van Hemert 2026): the past months most similar to
     today on seven standardised state variables, and what the market and each industry did next.
  4. Combination forecast (Rapach, Strauss & Zhou 2010): the average of single-variable logistic
     forecasts of "S&P 500 higher in 12 months", re-estimated every year on past data only.
  5. The call: models are weighted by their out-of-sample skill since 1975 (Brier score versus the
     historical base rate). A model with no measured skill gets zero weight.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from html import escape as e
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_check import build_signals, col, default_store, episodes, forward, load_panel  # noqa: E402

HORIZONS = [3, 6, 12, 18, 24]
OOS_START = "1975-01-31"
INDUSTRY_ETF = {"NoDur": "Staples (XLP)", "Durbl": "Durables (XLY)", "Manuf": "Industrials (XLI)", "Enrgy": "Energy (XLE)",
                "Chems": "Materials (XLB)", "BusEq": "Technology (XLK)", "Telcm": "Telecom (XLC)", "Utils": "Utilities (XLU)",
                "Shops": "Retail (XLY)", "Hlth": "Health care (XLV)", "Money": "Financials (XLF)", "Other": "Other"}


# --------------------------------------------------------------------------- data

def load_daily(store: Path, cols=("px:^GSPC", "px:^VIX")) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
        p = store / "panel_daily.parquet"
        if p.exists():
            have = set(pq.read_schema(p).names)
            return pd.read_parquet(p, columns=[c for c in cols if c in have])
    except Exception:
        pass
    c = store / "panel_daily.csv.gz"
    if c.exists():
        try:
            return pd.read_csv(c, index_col=0, parse_dates=True, usecols=lambda n: n == "date" or n in cols)
        except Exception:
            pass
    return pd.DataFrame()


def latest(P: pd.DataFrame, *names: str) -> pd.Series:
    """Latest published value: monthly macro data arrive weeks late, so carry it forward up to 2 months."""
    return col(P, *names).ffill(limit=2)


def zscore_expanding(x: pd.Series, minp: int = 120) -> pd.Series:
    m, s = x.expanding(minp).mean(), x.expanding(minp).std()
    return ((x - m) / s).clip(-3, 3)


def state_variables(P: pd.DataFrame, spx: pd.Series, daily: pd.Series | None) -> pd.DataFrame:
    """Seven state variables in the spirit of Mulliner et al. (2026), standardised with past data only."""
    gs10, tb3 = latest(P, "fred:GS10", "shiller:GS10"), latest(P, "fred:TB3MS")
    wti, cu = latest(P, "fred:WTISPLC"), latest(P, "fred:PCOPPUSDM", "px:HG=F")
    if daily is not None and len(daily) > 500:
        r = np.log(daily).diff()
        vol = (r.rolling(63, min_periods=40).std() * math.sqrt(252)).resample("ME").last().reindex(P.index)
    else:
        vol = spx.pct_change().rolling(12, min_periods=9).std() * math.sqrt(12)
    raw = pd.DataFrame({
        "Equity trend (12m)": np.log(spx / spx.shift(12)),
        "Yield curve (10Y-3M)": gs10 - tb3,
        "Oil (12m change)": np.log(wti / wti.shift(12)),
        "Copper (12m change)": np.log(cu / cu.shift(12)),
        "T-bill (12m change)": tb3.diff(12),
        "Volatility": vol,
        "Stock-bond correlation (3y)": spx.pct_change().rolling(36, min_periods=24).corr(-gs10.diff()),
    }, index=P.index)
    return raw.apply(zscore_expanding), raw


# --------------------------------------------------------------------------- 1-2. horizons and roles

def compare(flag: pd.Series, fwd: pd.Series) -> dict:
    df = pd.DataFrame({"f": flag, "r": fwd}).dropna()
    if df.empty or df.f.sum() == 0 or (df.f == 0).sum() == 0:
        return {}
    half = df.index[len(df) // 2]
    halves = []
    for part in (df[df.index < half], df[df.index >= half]):
        a, b = part[part.f == 1].r, part[part.f == 0].r
        if len(a) and len(b):
            halves.append((a.mean() - b.mean(), (a > 0).mean() - (b > 0).mean()))
        else:
            halves.append((np.nan, np.nan))
    on, off = df[df.f == 1].r, df[df.f == 0].r
    return {"on_mean": on.mean(), "off_mean": off.mean(), "on_up": (on > 0).mean(), "off_up": (off > 0).mean(),
            "eps": episodes(df.f), "halves": halves}


def role_of(c: dict) -> str:
    if not c or c["eps"] < 4:
        return "Unproven (too few episodes)"
    h = c["halves"]
    if any(np.isnan(v) for pair in h for v in pair):
        return "Unproven (one half lacks data)"
    if all(m < 0 and u < 0 for m, u in h):
        return "Defensive"
    if all(m > 0 and u > 0 for m, u in h):
        return "Recovery"
    return "Unproven (inconsistent)"


# --------------------------------------------------------------------------- 3. analogs

def distances(Z: np.ndarray, i: int, last_j: int) -> np.ndarray:
    if last_j < 0:
        return np.array([])
    zi = Z[i]
    if np.isfinite(zi).sum() < 4:
        return np.full(last_j + 1, np.nan)
    diff = Z[: last_j + 1] - zi
    ok = np.isfinite(diff)
    cnt = ok.sum(axis=1)
    sq = np.where(ok, diff ** 2, 0.0).sum(axis=1)
    d = np.sqrt(sq / np.maximum(cnt, 1))
    d[cnt < 4] = np.nan
    return d


def analog_at(Z, i, fwd12, h=12, q=0.2):
    d = distances(Z, i, i - h)
    valid = np.isfinite(d) & np.isfinite(fwd12[: len(d)])
    if valid.sum() < 120:
        return None
    dv = d[valid]
    lo, hi = np.quantile(dv, q), np.quantile(dv, 1 - q)
    idx = np.where(valid)[0]
    sim, anti = idx[dv <= lo], idx[dv >= hi]
    return {"sim": sim, "anti": anti, "d": d}


def analog_episodes(sim_idx, d, dates, fwd12, k=6):
    """Group similar months into runs; rank runs by their closest month."""
    if len(sim_idx) == 0:
        return []
    runs, cur = [], [sim_idx[0]]
    for j in sim_idx[1:]:
        if j - cur[-1] <= 2:
            cur.append(j)
        else:
            runs.append(cur)
            cur = [j]
    runs.append(cur)
    out = []
    for r in runs:
        best = min(r, key=lambda j: d[j])
        out.append({"start": dates[r[0]].strftime("%b %Y"), "end": dates[r[-1]].strftime("%b %Y"),
                    "closest": dates[best].strftime("%b %Y"), "dist": float(d[best]),
                    "next12": float(np.nanmean(fwd12[r]))})
    return sorted(out, key=lambda x: x["dist"])[:k]


# --------------------------------------------------------------------------- 4. combination forecast

def logit_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2, iters: int = 25) -> np.ndarray:
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(2)
        g = X.T @ (y - p) - ridge * b
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def cape_updated(P: pd.DataFrame, spx: pd.Series) -> pd.Series:
    """Shiller CAPE, extended to the latest month by price (earnings change slowly; standard practice)."""
    cape = col(P, "shiller:CAPE").copy()
    lv = cape.last_valid_index()
    if lv is not None and lv < P.index[-1] and pd.notna(spx.get(lv)):
        after = cape.index > lv
        cape[after] = cape[lv] * (spx[after] / spx[lv])
    return cape


def features(P: pd.DataFrame, spx: pd.Series, raw_state: pd.DataFrame) -> pd.DataFrame:
    gs10, tb3 = latest(P, "fred:GS10", "shiller:GS10"), latest(P, "fred:TB3MS")
    cpi = latest(P, "fred:CPIAUCSL", "shiller:CPI")
    unrate = latest(P, "fred:UNRATE")
    u3 = unrate.rolling(3).mean()
    return pd.DataFrame({
        "Trend (price vs 10-month average)": spx / spx.rolling(10).mean() - 1,
        "Valuation (log CAPE)": np.log(cape_updated(P, spx)).shift(3),
        "Inflation (CPI YoY)": ((cpi / cpi.shift(12) - 1) * 100).shift(1),
        "Credit (Baa-10Y, 6m change)": (latest(P, "fred:BAA") - gs10).diff(6),
        "Yield curve (10Y-3M)": gs10 - tb3,
        "Volatility": raw_state["Volatility"],
        "Sahm indicator": (u3 - u3.rolling(13).min()).shift(1),
        "Excess bond premium": latest(P, "ebp:ebp").shift(1),
    }, index=P.index)


def combination_forecasts(F: pd.DataFrame, up12: pd.Series, refit_every: int = 12) -> pd.DataFrame:
    """Out-of-sample probability from each single-variable model, refit yearly on data known at the time."""
    dates = F.index
    out = pd.DataFrame(index=dates, columns=F.columns, dtype=float)
    y_all = up12.to_numpy()
    start = int(np.searchsorted(dates, pd.Timestamp(OOS_START)))
    for name in F.columns:
        x_all = F[name].to_numpy(dtype=float)
        b, mu, sd = None, 0.0, 1.0
        for i in range(start, len(dates)):
            if b is None or (i - start) % refit_every == 0:
                train = np.arange(0, max(i - 12 + 1, 0))          # targets known by month i
                m = np.isfinite(x_all[train]) & np.isfinite(y_all[train])
                if m.sum() >= 120:
                    xt, yt = x_all[train][m], y_all[train][m]
                    mu, sd = xt.mean(), xt.std() or 1.0
                    b = logit_fit((xt - mu) / sd, yt)
            if b is not None and np.isfinite(x_all[i]):
                z = (x_all[i] - mu) / sd
                out.iat[i, out.columns.get_loc(name)] = 1 / (1 + math.exp(-(b[0] + b[1] * z)))
    return out


def score(prob: pd.Series, up12: pd.Series, fwd12: pd.Series) -> dict:
    df = pd.DataFrame({"p": prob, "y": up12, "r": fwd12}).dropna()
    df = df[df.index >= pd.Timestamp(OOS_START)]
    if len(df) < 60:
        return {}
    clim = up12.expanding(120).mean().shift(12).reindex(df.index)      # base rate known at the time
    ok = clim.notna()
    df, clim = df[ok], clim[ok]
    brier, brier_c = ((df.p - df.y) ** 2).mean(), ((clim - df.y) ** 2).mean()
    terc = df.p.rank(pct=True)
    return {"months": int(len(df)), "from": df.index[0].strftime("%Y-%m"),
            "hit": float(((df.p >= 0.5) == (df.y == 1)).mean()), "always_up": float(df.y.mean()),
            "skill": float(1 - brier / brier_c), "top": float(df.r[terc >= 2 / 3].mean()),
            "bottom": float(df.r[terc <= 1 / 3].mean())}


# --------------------------------------------------------------------------- main

def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def run(store: Path) -> dict:
    P = load_panel(store)
    S, spx = build_signals(P)
    last = spx.last_valid_index()                 # today = latest month with a market price
    P = P.loc[:last]
    S, spx = build_signals(P)
    if spx.notna().sum() < 240:
        raise SystemExit("Not enough S&P 500 history. Run fetch_all.py first.")
    dd = load_daily(store)
    daily = dd["px:^GSPC"].dropna() if "px:^GSPC" in dd else None
    fwd = {h: forward(spx, h) for h in HORIZONS}
    fwd12 = fwd[12]
    up12 = (fwd12 > 0).astype(float).where(fwd12.notna())
    dates = P.index
    T = len(dates)

    # 1-2. horizons and roles
    horizon_rows, roles = [], {}
    for name, (flag, direction, basis) in S.items():
        row = {"signal": name, "expected": direction}
        for h in HORIZONS:
            c = compare(flag, fwd[h])
            row[h] = c
            if h == 12:
                roles[name] = role_of(c)
        best = [h for h in HORIZONS if role_of(row[h]) in ("Defensive", "Recovery")]
        row["role"] = roles[name]
        row["holds_at"] = best
        now = flag.dropna()
        row["now"] = bool(len(now) and now.iloc[-1] == 1)
        horizon_rows.append(row)

    from evidence_check import worst_dip_12m
    dip = worst_dip_12m(spx)
    for r in horizon_rows:
        r["roles_h"] = {h: role_of(r[h]) for h in HORIZONS}
        d = pd.DataFrame({"f": S[r["signal"]][0], "d": dip}).dropna()
        r["dip_on"] = float(d[d.f == 1].d.mean()) if (d.f == 1).any() else None
        r["dip_off"] = float(d[d.f == 0].d.mean()) if (d.f == 0).any() else None

    # today's combination of validated signals
    on_valid = [r["signal"] for r in horizon_rows if r["now"] and r["role"] in ("Defensive", "Recovery")]
    combo = None
    if len(on_valid) >= 2:
        both = pd.concat([S[n][0] for n in on_valid], axis=1).min(axis=1)
        c = compare(both, fwd12)
        if c:
            combo = {"signals": on_valid, **{k: c[k] for k in ("on_mean", "off_mean", "on_up", "off_up", "eps")}}
            flag = both.dropna().astype(bool)
            starts = flag & ~flag.shift(1, fill_value=False)
            combo["episodes_list"] = [{"start": d.strftime("%b %Y"), "next12": float(fwd12.get(d, np.nan))}
                                      for d in flag.index[starts]]

    # 3. analogs
    Z, raw = state_variables(P, spx, daily)
    Zn = Z.to_numpy(dtype=float)
    f12 = fwd12.to_numpy(dtype=float)
    ind_cols = [c for c in P.columns if c.startswith("ind:")]
    mkt = col(P, "ff:Mkt-RF") + col(P, "ff:RF")
    logm = np.log1p(mkt / 100)
    fwd_mkt = logm.rolling(12).sum().shift(-12)
    ind_ex = {}
    for c in ind_cols:
        li = np.log1p(P[c].astype(float) / 100)
        ind_ex[c[4:]] = (np.exp(li.rolling(12).sum().shift(-12)) - np.exp(fwd_mkt)).to_numpy()

    analog_prob = pd.Series(np.nan, index=dates)
    ls_spread = []
    start = int(np.searchsorted(dates, pd.Timestamp(OOS_START)))
    for i in range(start, T):
        a = analog_at(Zn, i, f12)
        if a is None:
            continue
        analog_prob.iat[i] = float(np.mean(f12[a["sim"]] > 0))
        if ind_ex and i + 12 < T:                      # out-of-sample industry ranking
            means = {k: np.nanmean(v[a["sim"]]) for k, v in ind_ex.items() if np.isfinite(v[a["sim"]]).sum() >= 24}
            realized = {k: ind_ex[k][i] for k in means if np.isfinite(ind_ex[k][i])}
            ranked = [k for k in sorted(means, key=means.get) if k in realized]
            if len(ranked) >= 8:
                ls_spread.append(np.mean([realized[k] for k in ranked[-3:]]) - np.mean([realized[k] for k in ranked[:3]]))
    cur = analog_at(Zn, T - 1, f12)
    analog_now = None
    if cur is not None:
        sim, anti = cur["sim"], cur["anti"]
        sr = f12[sim]
        analog_now = {"n": int(len(sim)), "up": float(np.mean(sr > 0)), "mean": float(np.nanmean(sr)),
                      "p10": float(np.nanpercentile(sr, 10)), "p90": float(np.nanpercentile(sr, 90)),
                      "anti_up": float(np.mean(f12[anti] > 0)), "anti_mean": float(np.nanmean(f12[anti])),
                      "episodes": analog_episodes(sim, cur["d"], dates, f12),
                      "state": {k: (None if not np.isfinite(v) else float(v)) for k, v in Z.iloc[-1].items()},
                      "vars_used": int(np.isfinite(Z.iloc[-1].to_numpy(dtype=float)).sum())}
        tilts = []
        for k, v in ind_ex.items():
            vals = v[sim]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 24:
                tilts.append({"industry": INDUSTRY_ETF.get(k, k), "excess": float(vals.mean()),
                              "beat": float((vals > 0).mean()), "n": int(len(vals))})
        analog_now["tilts"] = sorted(tilts, key=lambda t: -t["excess"])
    ls = {"months": len(ls_spread), "mean": float(np.mean(ls_spread)) if ls_spread else None,
          "positive": float(np.mean(np.array(ls_spread) > 0)) if ls_spread else None}

    # 4. combination forecast
    F = features(P, spx, raw)
    single = combination_forecasts(F, up12)
    combo_prob = single.mean(axis=1, skipna=True).where(single.notna().sum(axis=1) >= 3)
    # current single-variable probabilities, refit on all data known now
    now_single = {}
    for name in F.columns:
        x = F[name].to_numpy(dtype=float)
        train = np.arange(0, T - 12)
        m = np.isfinite(x[train]) & np.isfinite(up12.to_numpy()[train])
        if m.sum() >= 120 and np.isfinite(x[-1]):
            xt, yt = x[train][m], up12.to_numpy()[train][m]
            mu, sd = xt.mean(), xt.std() or 1.0
            b = logit_fit((xt - mu) / sd, yt)
            now_single[name] = 1 / (1 + math.exp(-(b[0] + b[1] * (x[-1] - mu) / sd)))
    combo_now = float(np.mean(list(now_single.values()))) if len(now_single) >= 3 else None

    # 5. validation and the call
    track = {"Historical analogs": score(analog_prob, up12, fwd12), "Combination forecast": score(combo_prob, up12, fwd12)}
    for name in F.columns:
        track[f"  single: {name}"] = score(single[name], up12, fwd12)
    base_now = float(up12.dropna().mean())
    models = {"Historical analogs": analog_now["up"] if analog_now else None, "Combination forecast": combo_now}
    def usable(t):          # positive skill AND higher forecasts followed by higher returns
        return bool(t) and t.get("skill", 0) > 0 and t.get("top", 0) > t.get("bottom", 0)
    for k, t in track.items():
        if t:
            t["usable"] = usable(t)
    weights = {k: track[k]["skill"] if track.get(k) and track[k].get("usable") else 0.0 for k in models}
    used = {k: w for k, w in weights.items() if w > 0 and models[k] is not None}
    if used:
        prob = sum(models[k] * w for k, w in used.items()) / sum(used.values())
        basis = "skill-weighted model forecast"
    else:
        prob, basis = base_now, ("historical base rate: no model has a proven 12-month edge (positive skill and "
                                 "correct ordering out of sample)")
    best_skill = max([track[k].get("skill", 0) for k in models if track.get(k)] or [0])
    agree = [m for m in models.values() if m is not None]
    same_side = len(agree) == 2 and ((agree[0] - base_now) * (agree[1] - base_now) > 0)
    confidence = "Low" if not used or best_skill < 0.02 else ("Moderate" if best_skill < 0.06 or not same_side else "Higher")
    gap = prob - base_now
    direction = ("Clearly better odds than usual" if gap >= 0.10 else "Somewhat better odds than usual" if gap >= 0.04
                 else "Clearly worse odds than usual" if gap <= -0.10 else "Somewhat worse odds than usual" if gap <= -0.04
                 else "About normal odds")

    # criteria: roles come from the evidence above, so they update themselves as history grows
    def ev(r):
        c12 = r[12] or {}
        return {"signal": r["signal"], "on_up": c12.get("on_up"), "off_up": c12.get("off_up"),
                "on_mean": c12.get("on_mean"), "off_mean": c12.get("off_mean"), "eps": c12.get("eps"),
                "dip_on": r["dip_on"], "dip_off": r["dip_off"],
                "horizons": [h for h in HORIZONS if r["roles_h"][h] in ("Defensive", "Recovery")]}
    def completed(r):        # episodes whose 12-month outcome is known: the current, unfinished one never counts
        return (r[12] or {}).get("eps", 0) >= 4
    recovery = [r for r in horizon_rows if completed(r) and (r["roles_h"][12] == "Recovery" or r["roles_h"][6] == "Recovery")]
    defensive = [r for r in horizon_rows if completed(r) and r["roles_h"][12] == "Defensive"]
    short = [r for r in horizon_rows if completed(r) and r["roles_h"][3] == "Defensive" and r["roles_h"][6] == "Defensive"
             and r["roles_h"][12] != "Defensive"]
    rec_on, def_on, short_on = [r for r in recovery if r["now"]], [r for r in defensive if r["now"]], [r for r in short if r["now"]]
    if rec_on:
        level, colour = "Recovery window", "good"
        summary = ("A proven re-entry signal is on. After it fired, the following 6 to 24 months beat the average in both "
                   "halves of history.")
    elif len(def_on) >= 2:
        level, colour = "High risk", "bad"
        summary = (f"{len(def_on)} defensive signals are on together. Odds of gains are below normal and dips have run deeper; "
                   "no re-entry signal has fired yet.")
    elif def_on or short_on:
        level, colour = "Elevated risk", "mixed"
        summary = "A defensive signal is on: lower odds of gains and deeper dips than usual. No re-entry signal has fired yet."
    else:
        level, colour = "Normal", "good"
        summary = "No proven defensive or re-entry signal is on. History says expect roughly normal odds."
    vix_d = dd["px:^VIX"].dropna() if "px:^VIX" in dd else pd.Series(dtype=float)
    vix_me = col(P, "px:^VIX", "fred:VIXCLS").dropna()
    spread6 = (latest(P, "fred:BAA") - latest(P, "fred:GS10", "shiller:GS10")).diff(6).dropna()
    triggers = [
        {"name": "Volatility capitulation", "rule": "VIX closes a month at 30 or above",
         "now": f"VIX {vix_d.iloc[-1]:.1f} on {vix_d.index[-1]:%d %b}" if len(vix_d) else
                (f"VIX {vix_me.iloc[-1]:.1f} at last month-end" if len(vix_me) else "n/a"),
         "gap": None if not len(vix_me) else float(30 - (vix_d.iloc[-1] if len(vix_d) else vix_me.iloc[-1]))},
        {"name": "Credit stress washout", "rule": "Baa minus 10-year spread up 0.5 points or more in 6 months",
         "now": f"{spread6.iloc[-1]:+.2f} points over 6 months" if len(spread6) else "n/a",
         "gap": None if not len(spread6) else float(0.5 - spread6.iloc[-1])},
    ]
    criteria = {"level": level, "colour": colour, "summary": summary, "triggers": triggers,
                "recovery": [ev(r) | {"now": r["now"]} for r in recovery],
                "defensive": [ev(r) | {"now": r["now"]} for r in defensive],
                "short": [ev(r) | {"now": r["now"]} for r in short],
                "watch": [r["signal"] for r in horizon_rows if r not in recovery + defensive + short]}

    return {"generated": datetime.now().isoformat(timespec="minutes"), "data_to": dates[-1].strftime("%Y-%m-%d"),
            "criteria": criteria,
            "call": {"prob_up_12m": prob, "base_rate": base_now, "direction": direction, "confidence": confidence,
                     "basis": basis, "models": models, "weights": weights},
            "track": track, "analog": analog_now, "industry_ls": ls, "combo_now": combo, "single_now": now_single,
            "signals": [{"signal": r["signal"], "expected": r["expected"], "role": r["role"], "now": r["now"],
                         "holds_at": r["holds_at"],
                         "h": {str(h): ({k: r[h][k] for k in ("on_mean", "off_mean", "on_up", "off_up", "eps")} if r[h] else None)
                               for h in HORIZONS}} for r in horizon_rows]}


# --------------------------------------------------------------------------- report

def criteria_html(res: dict) -> str:
    cr = res["criteria"]
    out = [f"<div class='stance {cr['colour']}'><div class='lvl'>{e(cr['level'])}</div><p>{e(cr['summary'])}</p></div>"]
    def gap_text(t):
        if t["gap"] is None:
            return ""
        return "fired" if t["gap"] <= 0 else f"{t['gap']:.2f} to go"
    rows = "".join(
        f"<tr><td><b>{e(t['name'])}</b><div class='dim'>{e(t['rule'])}</div></td><td>{e(t['now'])}</td>"
        f"<td class='n'>{gap_text(t)}</td></tr>" for t in cr["triggers"])
    out.append("<h2>Re-entry triggers to watch</h2><div class='scroll'><table><thead><tr><th>Trigger</th><th>Now</th>"
               "<th>Distance</th></tr></thead><tbody>" + rows + "</tbody></table></div>"
               "<p class='note'>These marked better-than-average 6 to 24 month returns in both halves of history: by the time they "
               "fire, most of the selling has usually happened.</p>")

    def table(items, title, note):
        if not items:
            return ""
        def odds(x):
            return "" if x["on_up"] is None else f"{x['on_up']:.0%} vs {x['off_up']:.0%}"
        r = "".join(
            f"<tr><td>{e(x['signal'])}</td><td>{'<b>ON</b>' if x['now'] else 'off'}</td>"
            f"<td class='n'>{odds(x)}</td>"
            f"<td class='n'>{pct(x['dip_on'], 0)} vs {pct(x['dip_off'], 0)}</td><td class='n'>{x['eps'] or ''}</td>"
            f"<td>{e(', '.join(f'{h}m' for h in x['horizons']))}</td></tr>" for x in items)
        return (f"<h2>{e(title)}</h2><div class='scroll'><table><thead><tr><th>Signal</th><th>Now</th><th>Rose within 12m (on vs off)</th>"
                f"<th>Worst dip within 12m (on vs off)</th><th>Episodes</th><th>Proven at</th></tr></thead><tbody>{r}</tbody></table></div>"
                f"<p class='note'>{e(note)}</p>")
    out.append(table(cr["defensive"], "Defensive criteria", "Lower odds and deeper dips after these fired, in both halves of history."))
    out.append(table(cr["short"], "Short-term warnings", "Proven over 3 to 6 months only."))
    out.append(table(cr["recovery"], "Recovery criteria", "Better returns after these fired: reasons to add risk, not cut it."))
    if cr["watch"]:
        out.append(f"<p class='note'><b>Watch only, not proven:</b> {e('; '.join(cr['watch']))}.</p>")
    return "".join(out)


def finite(o):
    """Strict JSON: NaN and infinity become null (browsers reject NaN)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return finite(o.item())
        except Exception:
            return o
    return o


def write_report(res: dict, store: Path) -> Path:
    c = res["call"]
    a = res["analog"]
    parts = [criteria_html(res), "<h2>12-month odds</h2>", f"<div class='call'><div class='big'>{c['prob_up_12m']:.0%}</div><div><b>Chance the S&amp;P 500 is higher in 12 months</b>"
             f"<br>Historical base rate {c['base_rate']:.0%} · {e(c['direction'])} · confidence {e(c['confidence'])}"
             f"<br><span class='dim'>From the {e(c['basis'])}.</span></div></div>"]
    if a:
        parts.append(f"<p>In the {a['n']} past months most like today, the next 12 months averaged {pct(a['mean'])} "
                     f"(10th to 90th percentile {pct(a['p10'])} to {pct(a['p90'])}) and were positive {a['up']:.0%} of the time "
                     f"(matched on {a.get('vars_used', '?')} of 7 measures). "
                     f"In the months least like today: {pct(a['anti_mean'])}, positive {a['anti_up']:.0%}.</p>")
    if res.get("combo_now"):
        cb = res["combo_now"]
        done = [x for x in cb["episodes_list"] if math.isfinite(x["next12"])]
        recent = [x for x in done if int(x["start"][-4:]) >= 1980]
        eps = ", ".join(f"{x['start']} ({pct(x['next12'], 0)})" for x in done)
        if recent:
            eps += f". Since 1980: {sum(x['next12'] > 0 for x in recent)} of {len(recent)} followed by gains"
        parts.append(f"<h2>Today's combination</h2><p>{e(' + '.join(cb['signals']))} are on together. When that happened before "
                     f"({cb['eps']} episodes), the next 12 months averaged {pct(cb['on_mean'])} versus {pct(cb['off_mean'])} otherwise, "
                     f"positive {cb['on_up']:.0%} versus {cb['off_up']:.0%}.</p><p class='dim'>Starts and what followed: {e(eps)}.</p>")

    rows = []
    for k, t in res["track"].items():
        if not t:
            continue
        w = res["call"]["weights"].get(k)
        rows.append(f"<tr class='{'sub' if k.startswith('  ') else ''}'><td>{e(k.strip())}</td><td class='n'>{t['months']}</td>"
                    f"<td class='n'>{t['hit']:.0%}</td><td class='n'>{t['always_up']:.0%}</td><td class='n'>{t['skill'] * 100:+.1f}</td>"
                    f"<td class='n'>{pct(t['top'])}</td><td class='n'>{pct(t['bottom'])}</td>"
                    f"<td class='n'>{'' if w is None else f'{w:.3f}'}</td>"
                    f"<td>{'yes' if t.get('usable') else 'no'}</td></tr>")
    parts.append("<h2>Track record, out of sample since 1975</h2><div class='scroll'><table><thead><tr><th>Model</th><th>Months</th>"
                 "<th>Hit rate</th><th>Always-up hit rate</th><th>Skill vs base rate (%)</th><th>12m return, most bullish third</th>"
                 "<th>12m return, most bearish third</th><th>Weight</th><th>Usable</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
                 "<p class='note'>Every forecast uses only data available at the time (models refit yearly). Skill compares the Brier score "
                 "with the base rate known at the time; above zero means the model added information. <b>Usable</b> also requires the "
                 "most bullish third of forecasts to have been followed by higher returns than the most bearish third.</p>")

    if a:
        ep = "".join(f"<tr><td>{e(x['start'])} to {e(x['end'])}</td><td>{e(x['closest'])}</td><td class='n'>{x['dist']:.2f}</td>"
                     f"<td class='n'>{pct(x['next12'])}</td></tr>" for x in a["episodes"])
        st = "".join(f"<tr><td>{e(k)}</td><td class='n'>{'n/a' if v is None else f'{v:+.2f}'}</td></tr>" for k, v in a["state"].items())
        parts.append("<h2>Closest past periods</h2><div class='two'><div class='scroll'><table><thead><tr><th>Period</th><th>Closest month</th>"
                     "<th>Distance</th><th>Next 12m (avg)</th></tr></thead><tbody>" + ep + "</tbody></table></div>"
                     "<div class='scroll'><table><thead><tr><th>Today's state</th><th>z-score</th></tr></thead><tbody>" + st +
                     "</tbody></table></div></div>")
        if a.get("tilts"):
            ls = res["industry_ls"]
            tr = "".join(f"<tr><td>{e(t['industry'])}</td><td class='n'>{pct(t['excess'])}</td><td class='n'>{t['beat']:.0%}</td>"
                         f"<td class='n'>{t['n']}</td></tr>" for t in a["tilts"])
            ev = (f"Out of sample since 1975, the 3 industries the analogs ranked highest beat the 3 ranked lowest by {pct(ls['mean'])} "
                  f"a year on average, in {ls['positive']:.0%} of {ls['months']} months.") if ls.get("mean") is not None else ""
            parts.append("<h2>Industries in similar past periods</h2><div class='scroll'><table><thead><tr><th>Industry (closest ETF)</th>"
                         "<th>Next 12m vs market</th><th>Beat the market</th><th>Months</th></tr></thead><tbody>" + tr +
                         f"</tbody></table></div><p class='note'>{e(ev)} Ken French value-weighted industry portfolios since 1926.</p>")

    role_cls = {"Defensive": "bad", "Recovery": "good"}
    srows = []
    for s in res["signals"]:
        h12 = s["h"]["12"]
        cells = "".join(
            f"<td class='n'>{'' if not s['h'][str(h)] else pct(s['h'][str(h)]['on_up'] - s['h'][str(h)]['off_up'], 0)}</td>" for h in HORIZONS)
        srows.append(f"<tr><td>{e(s['signal'])}</td><td><span class='v {role_cls.get(s['role'], 'muted')}'>{e(s['role'])}</span></td>"
                     f"<td>{'<b>ON</b>' if s['now'] else 'off'}</td>{cells}"
                     f"<td>{e(', '.join(f'{h}m' for h in s['holds_at']) or 'none')}</td></tr>")
    parts.append("<h2>What each signal really does</h2><div class='scroll'><table><thead><tr><th>Signal</th><th>Role at 12 months</th><th>Now</th>"
                 + "".join(f"<th>{h}m</th>" for h in HORIZONS) + "<th>Consistent at</th></tr></thead><tbody>" + "".join(srows) +
                 "</tbody></table></div><p class='note'>Horizon columns: change in the chance of a positive return when the signal is on "
                 "versus off. <b>Defensive</b>: lower average return and lower chance of gains after it fires, in both halves of history. "
                 "<b>Recovery</b>: higher on both counts, so it marks a good time to add risk, not cut it.</p>")
    single = res.get("single_now") or {}
    if single:
        sr = "".join(f"<tr><td>{e(k)}</td><td class='n'>{v:.0%}</td></tr>" for k, v in sorted(single.items(), key=lambda kv: kv[1]))
        parts.append("<h2>Single-variable forecasts today</h2><div class='scroll'><table class='small'><thead><tr><th>Variable</th>"
                     "<th>Chance higher in 12m</th></tr></thead><tbody>" + sr + "</tbody></table></div>")

    css = """
    :root{--bg:#F6F7F4;--ink:#15263B;--muted:#5D6B7A;--line:#D9DED8;--card:#FFF;--good:#2E7D5B;--goodbg:#E3F1EA;--bad:#A63A32;--badbg:#F5DEDB;--mutedbg:#ECEFEA}
    @media (prefers-color-scheme: dark){:root{--bg:#10161F;--ink:#E7ECF2;--muted:#9AA7B5;--line:#2A3442;--card:#161E29;--good:#7FCFA8;--goodbg:#173528;--bad:#F0A39B;--badbg:#3D1D1A;--mutedbg:#1F2833}}
    body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;font-variant-numeric:tabular-nums}
    main{max-width:1180px;margin:0 auto;padding:40px 24px 64px} h1{font-size:2.2rem;margin:0 0 4px;letter-spacing:-.02em} h2{font-size:1.2rem;margin:36px 0 12px}
    .stance{border-left:8px solid var(--c);background:var(--card);border-radius:0 12px 12px 0;padding:18px 24px;margin:20px 0}
    .stance .lvl{font-size:2.6rem;font-weight:700;letter-spacing:-.02em;color:var(--c)} .stance p{margin:4px 0 0;max-width:70ch}
    .stance.good{--c:var(--good)} .stance.bad{--c:var(--bad)} .stance.mixed{--c:#B7791F}
    .call{display:flex;gap:24px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin:20px 0}
    .big{font-size:3.4rem;font-weight:700;letter-spacing:-.03em} .dim{color:var(--muted)} .note{color:var(--muted);font-size:.9rem;max-width:80ch}
    .scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)} .two{display:grid;grid-template-columns:2fr 1fr;gap:16px}
    @media (max-width:800px){.two{grid-template-columns:1fr}}
    table{border-collapse:collapse;width:100%;font-size:.9rem} th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    th{background:var(--card);font-weight:600;color:var(--muted);font-size:.82rem} td.n{text-align:right;white-space:nowrap} tr.sub td{color:var(--muted);font-size:.85rem}
    .v{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:600;font-size:.82rem;white-space:nowrap}
    .v.good{color:var(--good);background:var(--goodbg)} .v.bad{color:var(--bad);background:var(--badbg)} .v.muted{color:var(--muted);background:var(--mutedbg)}
    footer{color:var(--muted);font-size:.85rem;margin-top:40px}"""
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Macro Navigator</title><style>{css}</style></head><body><main><h1>Macro Navigator</h1>"
            f"<p class='dim'>Data to {e(res['data_to'])} · built {e(res['generated'].replace('T', ' '))}</p>{''.join(parts)}"
            "<p class='note'>Price returns without dividends. Twelve-month windows overlap, so effective sample sizes are smaller than "
            "the month counts. <a href='evidence_report.html'>Full signal evidence</a>. Research framework, not investment advice.</p></main></body></html>")
    path = store / "engine_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(default_store()))
    args = ap.parse_args(argv)
    store = Path(args.store)
    res = run(store)
    (store / "engine_results.json").write_text(json.dumps(finite(res), default=str, indent=1, allow_nan=False))
    path = write_report(res, store)
    c = res["call"]
    print(f"\nMacro Navigator (data to {res['data_to']})")
    print(f"  Stance: {res['criteria']['level']} - {res['criteria']['summary']}")
    print(f"  Chance S&P 500 higher in 12 months: {c['prob_up_12m']:.0%}  (base rate {c['base_rate']:.0%}) · {c['direction']} · confidence {c['confidence']}")
    for k in ("Historical analogs", "Combination forecast"):
        t = res["track"].get(k) or {}
        if t:
            m = c["models"].get(k)
            print(f"  {k:<22} now {'n/a' if m is None else f'{m:.0%}'}   out-of-sample skill {t['skill'] * 100:+.1f}%   hit rate {t['hit']:.0%} vs always-up {t['always_up']:.0%}")
    roles = {}
    for s in res["signals"]:
        roles.setdefault(s["role"].split(" (")[0], []).append(s["signal"].split(" (")[0])
    for r in ("Defensive", "Recovery"):
        if roles.get(r):
            print(f"  {r}: {', '.join(roles[r])}")
    print(f"\nSaved {path}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)
