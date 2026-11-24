"""Per-asset outlook on three horizons. Scores run from -100 (most negative) to +100.

Short term (1-3 months): trend and momentum, adjusted by the market regime.
   Basis: Moskowitz, Ooi & Pedersen (2012); Faber (2007); Moskowitz & Grinblatt (1999).
Medium term (6-12 months): regime fit, 12-1 month relative momentum, recession and dollar overlays.
   Basis: Jegadeesh & Titman (1993); Asness, Moskowitz & Pedersen (2013); Estrella & Mishkin (1998);
   Moench & Stein (2026).
Long term (3-5+ years): cyclically adjusted earnings yield, only where a CAPE is available.
   Basis: Campbell & Shiller (1998); the excess CAPE yield.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

GROUPS = {
    "us_indices": "US index",
    "us_sectors": "US sector",
    "us_industries": "US industry",
    "global_markets": "Global",
}


def label(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "No view"
    if score >= 40:
        return "Positive"
    if score >= 15:
        return "Leaning positive"
    if score > -15:
        return "Neutral"
    if score > -40:
        return "Leaning negative"
    return "Negative"


def _ret(s: pd.Series, n: int, skip: int = 0) -> float:
    s = s.dropna()
    if len(s) <= n:
        return float("nan")
    end = s.iloc[-1 - skip] if skip else s.iloc[-1]
    return float(end / s.iloc[-1 - n] - 1)


def short_term(p: pd.Series, bench: pd.Series | None, regime: int, amber: bool, vix_ratio: float) -> tuple[float, list[str]]:
    p = p.dropna()
    if len(p) < 210:
        return float("nan"), ["not enough history"]
    last, sma50, sma200 = p.iloc[-1], p.tail(50).mean(), p.tail(200).mean()
    notes = []
    score = 25 if last > sma200 else -25
    notes.append("above 200-day average" if last > sma200 else "below 200-day average")
    score += 15 if sma50 > sma200 else -15
    score += 10 if last > sma50 else -10
    r63 = _ret(p, 63)
    vol = p.pct_change().tail(126).std() * math.sqrt(252)
    if vol and not math.isnan(r63):
        z = r63 / (vol * math.sqrt(63 / 252))
        score += float(np.clip(z, -2, 2)) * 12.5
        notes.append(f"3-month return {r63:+.1%}")
    if bench is not None and bench is not p:
        rb = _ret(bench, 63)
        if not math.isnan(rb) and not math.isnan(r63):
            rel = r63 - rb
            score += float(np.clip(rel / 0.05, -1, 1)) * 15
            notes.append(f"{rel:+.1%} vs benchmark over 3 months")
    overlay = {1: 5, 2: 0, 3: -20, 4: 10}[regime] - (10 if amber else 0)
    if vix_ratio == vix_ratio and vix_ratio > 1.0:
        overlay -= 10
        notes.append("volatility curve inverted")
    if regime == 4 and score < 0:
        score *= 0.5   # momentum is least reliable right after bear markets (Daniel & Moskowitz 2016)
        notes.append("negative trend discounted in a recovery regime")
    return float(np.clip(score + overlay, -100, 100)), notes


def medium_term(t: str, p: pd.Series, bench: pd.Series | None, kind: str, is_em: bool, regime: int,
                tilts: dict, rec_prob: float, hy_change_63: float, dollar_yoy: float) -> tuple[float, list[str]]:
    p = p.dropna()
    if len(p) < 260:
        return float("nan"), ["not enough history"]
    notes = []
    tilt = int(tilts.get(f"R{regime}", {}).get(t, 0))
    score = 35 * tilt
    if tilt:
        notes.append(("favoured" if tilt > 0 else "disfavoured") + f" in R{regime}")
    m12 = _ret(p, 252, skip=21)
    if not math.isnan(m12):
        score += 10 if m12 > 0 else -10
        notes.append(f"12-1 month return {m12:+.1%}")
        if bench is not None and bench is not p:
            b12 = _ret(bench, 252, skip=21)
            if not math.isnan(b12):
                score += float(np.clip((m12 - b12) / 0.15, -1, 1)) * 25
    if rec_prob == rec_prob:
        if rec_prob >= 0.30:
            score += -15 if kind == "cyclical" else (10 if kind == "defensive" else 0)
            notes.append(f"recession probability {rec_prob:.0%}")
        elif rec_prob <= 0.15 and kind == "cyclical":
            score += 5
    if hy_change_63 == hy_change_63 and hy_change_63 >= 100 and kind == "cyclical":
        score -= 10
        notes.append("credit spreads widening")
    if is_em and dollar_yoy == dollar_yoy:
        if dollar_yoy >= 8:
            score -= 10
            notes.append("strong dollar headwind")
        elif dollar_yoy <= -5:
            score += 5
    return float(np.clip(score, -100, 100)), notes


def long_term(cape: float | None, real_10y: float) -> tuple[float, str]:
    if not cape or cape <= 0:
        return float("nan"), "CAPE unavailable"
    earnings_yield = 100 / cape
    if real_10y != real_10y:
        return float("nan"), f"CAPE {cape:.1f}; real yield unavailable"
    ecy = earnings_yield - real_10y
    score = 60 if ecy >= 4 else 20 if ecy >= 2 else -20 if ecy >= 0 else -60
    return float(score), f"CAPE {cape:.1f}, earnings yield {earnings_yield:.1f}%, excess over real 10Y {ecy:+.1f} pp"


def score_assets(prices: pd.DataFrame, universe: dict, P: pd.DataFrame, regime: int, amber: bool,
                 cape: float | None, country_cape: dict | None = None) -> pd.DataFrame:
    last = P.iloc[-1]
    tilts = universe.get("regime_tilts", {})
    bench_us = prices.get(universe.get("benchmark", "SPY"))
    bench_gl = prices.get(universe.get("global_benchmark", "ACWI"))
    lt_us = long_term(cape, float(last.get("real_10y", np.nan)))
    country_cape = country_cape or {}
    rows = []
    for group, group_name in GROUPS.items():
        for row in universe.get(group, []):
            t = row["ticker"]
            if t not in prices.columns or prices[t].dropna().empty:
                rows.append({"group": group_name, "ticker": t, "name": row["name"], "status": "no data"})
                continue
            p = prices[t]
            bench = bench_gl if group == "global_markets" else bench_us
            st, st_notes = short_term(p, bench, regime, amber, float(last.get("vix_ratio", np.nan)))
            mt, mt_notes = medium_term(t, p, bench, row.get("kind", "neutral"), bool(row.get("em")), regime, tilts,
                                       float(last.get("recession_prob_12m", np.nan)), float(last.get("hy_change_63", np.nan)),
                                       float(last.get("dollar_yoy", np.nan)))
            if group == "us_indices" and row.get("long_term"):
                lt, lt_note = lt_us
            elif group == "global_markets" and t in country_cape:
                lt, lt_note = long_term(float(country_cape[t]), float(last.get("real_10y", np.nan)))
            else:
                lt, lt_note = float("nan"), "not modelled (needs valuation data)"
            s = p.dropna()
            rows.append({
                "group": group_name, "ticker": t, "name": row["name"], "status": "ok",
                "price": float(s.iloc[-1]), "ret_1m": _ret(s, 21), "ret_3m": _ret(s, 63), "ret_12m": _ret(s, 252),
                "short_score": st, "short_view": label(st), "short_why": "; ".join(st_notes),
                "medium_score": mt, "medium_view": label(mt), "medium_why": "; ".join(mt_notes),
                "long_score": lt, "long_view": label(lt), "long_why": lt_note,
            })
    return pd.DataFrame(rows)
