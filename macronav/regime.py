"""Flags, Amber overlay, risk score and the 4-regime state machine (booklet Sections 3 and 5)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIME_NAMES = {
    1: "R1 Goldilocks",
    2: "R2 Hawkish / overheating",
    3: "R3 Breakdown / stagflation",
    4: "R4 Pivot / early recovery",
}
REGIME_SHORT = {1: "Goldilocks", 2: "Hawkish", 3: "Breakdown", 4: "Recovery"}


def _gt(s, x):
    return (s >= x).fillna(False)


def _lt(s, x):
    return (s <= x).fillna(False)


def compute_flags(P: pd.DataFrame, th: dict, pivot_events: dict | None = None) -> pd.DataFrame:
    i, p, g, c, v, a = (th[k] for k in ("inflation", "policy", "growth", "credit", "volatility", "amber"))
    F = pd.DataFrame(index=P.index)

    F["INFL_HOT"] = (_gt(P.core_pce_yoy, i["core_pce_hot"]) | _gt(P.core_cpi_3m_ann, i["core_cpi_3m_hot"])
                     | (_gt(P.brent_yoy, i["brent_yoy_hot"]) & _gt(P.core_cpi_3m_ann, i["core_cpi_3m_with_oil"])))
    F["INFL_COOL"] = (_lt(P.core_pce_yoy, i["core_pce_cool"]) & _lt(P.core_cpi_3m_ann, i["core_cpi_3m_cool"])
                      & (P.brent_yoy < i["brent_yoy_cool"]).fillna(False))
    tightening = (P.ff_change_6m > 0).fillna(False) | _gt(P.two_y_minus_ff, p["two_y_minus_ff_hike"])
    F["HAWKISH"] = tightening & (_gt(P.real_ff, p["real_ff_restrictive"]) | _gt(P.ff_change_12m, p["ff_12m_hawkish"]))

    event_pivot = pd.Series(False, index=P.index)
    for d in (pivot_events or {}):
        d = pd.Timestamp(d)
        event_pivot |= (P.index >= d) & (P.index <= d + pd.Timedelta(days=60))
    F["PIVOT"] = (P.ff_change_3m < 0).fillna(False) | _lt(P.two_y_minus_ff, p["two_y_minus_ff_pivot"]) | event_pivot

    weak = [_gt(P.sahm, g["sahm_warn"]), _gt(P.claims_vs_low, g["claims_warn"]),
            _lt(P.philly, g["philly_weak"]), _lt(P.permits_yoy, g["permits_yoy_weak"])]
    F["GROWTH_WEAK"] = sum(w.astype(int) for w in weak) >= 2
    F["RECESSION"] = _gt(P.sahm, g["sahm_recession"]) | (_gt(P.claims_vs_low, g["claims_recession"]) & _lt(P.philly, g["philly_recession"]))

    F["CREDIT_STRESS"] = (_gt(P.hy_bp, c["hy_stress_bp"]) | _gt(P.hy_bp - P.hy_min_126, c["hy_rise_stress_bp"])
                          | (P.nfci > c["nfci_stress"]).fillna(False))
    F["CREDIT_CRISIS"] = (_gt(P.hy_bp, c["hy_crisis_bp"]) | _gt(P.vix, v["vix_crisis"])
                          | (P.vix_ratio_min_3d > v["vix_ratio_crisis"]).fillna(False) | _gt(P.cp_ff_bp, c["cp_ff_crisis_bp"]))
    F["TREND_DOWN"] = P.trend_down.fillna(False).astype(bool)
    washout = _gt(P.vix_max_60, v["vix_washout"])
    release = (P.vix_avg_5 <= v["vix_release"] * P.vix_max_60).fillna(False)
    F["CAPITULATION_DONE"] = washout & release
    F["CREDIT_HEALING"] = _lt(P.hy_bp, P.hy_max_126 - c["hy_heal_from_max_bp"]) & (P.hy_bp < P.hy_10d_ago).fillna(False)

    amber_signs = pd.concat({
        "HY up 100 bp from its 6-month low": _gt(P.hy_bp - P.hy_min_126, a["hy_rise_bp"]),
        "VIX at or above 25 for 5 sessions": _gt(P.vix_days_above_25, a["vix_days"]),
        "S&P 500 down 10% or more from its high": _lt(P.drawdown_pct, -a["drawdown_pct"]),
        "Bull steepener after a curve inversion": (P.curve_pattern == "bull steepener") & P.inverted_within_6m,
        "Regional banks lagging by 10% or more": _lt(P.kre_rel_20d, a["kre_rel_pct"]),
        "Financial conditions tightening (NFCI)": (P.nfci > a["nfci"]).fillna(False),
    }, axis=1)
    F["AMBER_COUNT"] = amber_signs.sum(axis=1)
    F.attrs["amber_signs"] = amber_signs
    return F


# ---------------------------------------------------------------- risk score (Section 5)

def _three(cond10, cond5):
    return np.where(cond10, 10, np.where(cond5, 5, 0))


def risk_score(P: pd.DataFrame) -> pd.DataFrame:
    S = pd.DataFrame(index=P.index)
    S["Inflation"] = _three(_gt(P.core_pce_yoy, 3.0) | _gt(P.core_cpi_3m_ann, 4.0),
                            _gt(P.core_pce_yoy, 2.5) | _gt(P.core_cpi_3m_ann, 3.0))
    S["Energy"] = _three(_gt(P.brent_yoy, 50), _gt(P.brent_yoy, 20))
    S["Policy direction"] = _three(_gt(P.two_y_minus_ff, 0.5) | (P.ff_change_3m > 0).fillna(False),
                                   (P.two_y_minus_ff > -0.5).fillna(False) & (P.ff_change_3m >= 0).fillna(False))
    S["Real rates and yields"] = _three(_gt(P.real_ff, 1.5) | _gt(P.dgs10, 5.0) | _gt(P.dgs10_change_6m, 1.0),
                                        _gt(P.real_ff, 0.5) | _gt(P.dgs10, 4.5))
    S["Curve"] = _three((P.curve_pattern == "bull steepener") & P.inverted_within_6m,
                        _lt(P.t10y3m, 0.5))
    S["Labour and growth"] = _three(_gt(P.sahm, 0.5) | _gt(P.claims_vs_low, 0.20),
                                    _gt(P.sahm, 0.2) | _gt(P.claims_vs_low, 0.10) | _lt(P.philly, -15))
    S["Credit"] = _three(_gt(P.hy_bp, 500) | _gt(P.hy_bp - P.hy_min_126, 150),
                         _gt(P.hy_bp, 350) | _gt(P.hy_bp - P.hy_min_126, 75))
    S["Funding"] = _three((P.nfci > 0).fillna(False) | _gt(P.sofr_iorb_bp, 25) | _gt(P.cp_ff_bp, 100),
                          (P.nfci > -0.3).fillna(False) | (P.sofr_iorb_bp > 5).fillna(False) | _lt(P.kre_rel_20d, -10))
    S["Volatility"] = _three(_gt(P.vix, 30) | (P.vix_ratio > 1.0).fillna(False) | _gt(P.move, 150),
                             _gt(P.vix, 18) | _gt(P.vix_ratio, 0.90))
    near_sma = ((P.spx / P.spx_sma10m - 1).abs() <= 0.02).fillna(False)
    S["Trend and breadth"] = _three(P.trend_down.fillna(False) | _lt(P.drawdown_pct, -10) | _lt(P.sector_breadth_pct, 40),
                                    _lt(P.drawdown_pct, -5) | near_sma | _lt(P.sector_breadth_pct, 60))
    S["Total"] = S.sum(axis=1)
    return S


def score_band(score: float) -> str:
    if score <= 25:
        return "Low"
    if score <= 45:
        return "Moderate"
    if score <= 65:
        return "Elevated"
    if score <= 80:
        return "High"
    return "Extreme"


# ---------------------------------------------------------------- state machine

@dataclass
class RegimeResult:
    weekly: pd.DataFrame                    # date, regime, amber, score, reason
    current: int
    amber: bool
    reason: str
    episode: dict = field(default_factory=dict)


def _month_end_counts(flag: pd.Series) -> pd.Series:
    """Consecutive month-ends a flag was true, indexed by each month's last business day."""
    me = flag.groupby(flag.index.to_period("M")).tail(1)
    b = me.fillna(False).astype(bool)
    return b.astype(int).groupby((~b).cumsum()).cumsum()


def _count_asof(counts: pd.Series, date) -> int:
    past = counts[counts.index <= date]
    return int(past.iloc[-1]) if len(past) else 0


def run_state_machine(P: pd.DataFrame, F: pd.DataFrame, th: dict) -> RegimeResult:
    per, c, v = th["persistence"], th["credit"], th["volatility"]
    need = th["amber"]["need"]
    hawk = F.INFL_HOT | F.HAWKISH
    r3_cond = (F.CREDIT_CRISIS | F.RECESSION | (F.CREDIT_STRESS & (F.GROWTH_WEAK | F.TREND_DOWN))
               | (F.INFL_HOT & F.GROWTH_WEAK & F.TREND_DOWN))
    hawk_me = _month_end_counts(hawk)
    cool_me = _month_end_counts(~hawk)
    infl_cool_me = _month_end_counts(F.INFL_COOL & ~F.HAWKISH)

    core_ok = P[["hy_bp", "vix", "spx"]].notna().all(axis=1) & P.core_pce_yoy.notna() & P.hy_min_126.notna()
    if not core_ok.any():
        raise ValueError("Not enough overlapping data to run the regime model (need HY OAS, VIX, S&P, core PCE).")
    start = core_ok[core_ok].index[0]
    fridays = P.loc[start:].index[P.loc[start:].index.day_of_week == 4]
    if len(fridays) == 0 or fridays[-1] != P.index[-1]:
        fridays = fridays.append(pd.DatetimeIndex([P.index[-1]]))  # always evaluate the latest day

    # initial state without persistence
    d0 = fridays[0]
    state = 3 if r3_cond.loc[d0] else (2 if hawk.loc[d0] else 1)
    r3_weeks = r3_exit_weeks = amber_clear = 0
    amber = False
    ep = {"peak_hy": float(P.hy_bp.loc[d0]), "low_spx": float(P.spx.loc[d0]), "start": d0} if state == 3 else {}
    r4_start = None
    rows = []
    prev = d0
    for d in fridays:
        week = P.loc[prev:d]
        crisis_week = bool(F.CREDIT_CRISIS.loc[prev:d].any())
        reason = ""
        if state == 3:
            ep["peak_hy"] = max(ep.get("peak_hy", -np.inf), float(week.hy_bp.max()))
            ep["low_spx"] = min(ep.get("low_spx", np.inf), float(week.spx.min()))
            if F.PIVOT.loc[d] and F.CAPITULATION_DONE.loc[d] and F.CREDIT_HEALING.loc[d]:
                state, r4_start, reason = 4, d, "Central bank pivot after a volatility washout, with credit spreads healing"
            else:
                exit_ok = ((P.hy_bp.loc[d] < c["hy_exit_bp"]) or (P.hy_bp.loc[d] <= P.hy_max_126.loc[d] - c["hy_exit_from_max_bp"])) \
                    and not F.TREND_DOWN.loc[d] and (P.vix_avg_20.loc[d] < v["vix_exit"])
                r3_exit_weeks = r3_exit_weeks + 1 if exit_ok else 0
                if r3_exit_weeks >= per["r3_exit_weeks"]:
                    state = 2 if hawk.loc[d] else 1
                    reason = "Credit, trend and volatility normalised for 4 weeks without a pivot"
        elif state == 4:
            failed = (P.hy_bp.loc[d] > ep.get("peak_hy", np.inf) + 50) or (P.spx.loc[d] < 0.95 * ep.get("low_spx", -np.inf))
            months = (d - r4_start).days / 30.4 if r4_start is not None else 0
            done = (P.months_above_sma10.loc[d] >= 2) and (P.hy_bp.loc[d] < c["hy_exit_bp"]) and (P.vix_avg_20.loc[d] < v["vix_calm"])
            if failed:
                state, reason = 3, "Failed recovery: new credit-spread high or index below 95% of the breakdown low"
                ep = {"peak_hy": float(P.hy_bp.loc[d]), "low_spx": float(P.spx.loc[d]), "start": d}
                r3_exit_weeks = 0
            elif done or months >= per["r4_max_months"]:
                state = 2 if hawk.loc[d] else 1
                reason = "Recovery confirmed: trend restored, spreads and volatility normal"
        else:
            r3_weeks = r3_weeks + 1 if r3_cond.loc[d] else 0
            if crisis_week or r3_weeks >= per["r3_weeks"]:
                state, r3_exit_weeks = 3, 0
                reason = "Crisis trigger (VIX, HY spread or funding)" if crisis_week else "Credit stress with weak growth or a broken trend"
                ep = {"peak_hy": float(week.hy_bp.max()), "low_spx": float(week.spx.min()), "start": d}
            elif state == 1 and _count_asof(hawk_me, d) >= per["r2_month_ends"]:
                state, reason = 2, "Inflation hot or policy hawkish at consecutive month-ends"
            elif state == 2:
                need_me = 1 if _count_asof(infl_cool_me, d) >= 1 else per["r1_month_ends"]
                if _count_asof(cool_me, d) >= need_me:
                    state, reason = 1, "Inflation and policy no longer hot at consecutive month-ends"

        # Amber overlay applies only in R1/R2
        cnt = int(F.AMBER_COUNT.loc[d])
        if state in (1, 2):
            if cnt >= need:
                amber, amber_clear = True, 0
            elif amber:
                amber_clear += 1
                if amber_clear >= per["amber_clear_weeks"]:
                    amber, amber_clear = False, 0
        else:
            amber, amber_clear = False, 0
        rows.append({"date": d, "regime": state, "amber": amber, "amber_count": cnt, "reason": reason})
        prev = d

    weekly = pd.DataFrame(rows).set_index("date")
    last_reason = next((r for r in reversed(weekly.reason.tolist()) if r), "Initial classification")
    since = weekly.index[0]
    changes = weekly.index[weekly.regime != weekly.regime.shift()]
    if len(changes):
        since = changes[-1]
    current = int(weekly.regime.iloc[-1])
    episode = {"regime_since": str(since.date())}
    if current in (3, 4) and ep:   # breakdown memory matters only inside R3/R4
        episode.update({"breakdown_peak_hy_bp": ep.get("peak_hy"), "breakdown_low_spx": ep.get("low_spx"),
                        "breakdown_start": str(pd.Timestamp(ep["start"]).date()) if ep.get("start") is not None else None})
    return RegimeResult(weekly=weekly, current=current, amber=bool(weekly.amber.iloc[-1]),
                        reason=last_reason, episode=episode)
