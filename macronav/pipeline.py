"""One call builds the full snapshot: python -m macronav.pipeline (or scripts/build_snapshot.py)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import data as D
from .assets import score_assets
from .indicators import build_panel
from .narrative import build_narrative
from .regime import REGIME_NAMES, compute_flags, risk_score, run_state_machine, score_band
from .research import remember, research_watch

SNAPSHOT = D.DATA / "snapshot.json"
REPORT = D.DATA / "report.md"
HISTORY = D.DATA / "history.csv"

READINGS = [  # (key in panel, label, unit)
    ("core_pce_yoy", "Core PCE inflation, YoY", "%"),
    ("core_cpi_3m_ann", "Core CPI, 3-month annualised", "%"),
    ("brent", "Brent crude", "$"),
    ("brent_yoy", "Brent, YoY", "%"),
    ("ff_upper", "Fed funds upper bound", "%"),
    ("dgs2", "2-year Treasury", "%"),
    ("dgs10", "10-year Treasury", "%"),
    ("real_10y", "Real 10-year (TIPS)", "%"),
    ("two_y_minus_ff", "2Y minus fed funds", "pp"),
    ("real_ff", "Real policy rate", "pp"),
    ("t10y3m", "10Y minus 3M curve", "pp"),
    ("t10y2y", "10Y minus 2Y curve", "pp"),
    ("recession_prob_12m", "Recession probability, 12 months", "prob"),
    ("unrate", "Unemployment rate", "%"),
    ("sahm", "Sahm indicator", ""),
    ("claims_vs_low", "Claims vs 52-week low", "share"),
    ("philly", "Philadelphia Fed activity", ""),
    ("permits_yoy", "Building permits, YoY", "%"),
    ("hy_bp", "High-yield spread", "bp"),
    ("hy_min_126", "High-yield spread, 6-month low", "bp"),
    ("nfci", "Chicago Fed NFCI", ""),
    ("sofr_iorb_bp", "SOFR minus IORB", "bp"),
    ("vix", "VIX", ""),
    ("vix_ratio", "VIX / VIX3M", ""),
    ("move", "MOVE (bond volatility)", ""),
    ("dollar_yoy", "Broad dollar, YoY", "%"),
    ("spx", "S&P 500", "pts"),
    ("drawdown_pct", "S&P 500 from 52-week high", "%"),
    ("sector_breadth_pct", "Sectors above 200-day average", "%"),
    ("stock_bond_corr_60d", "Stock-bond correlation, 60 days", ""),
]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return o


def build_snapshot(fred_key: str | None = None, demo: bool = False, with_research: bool = True) -> dict:
    th = D.load_yaml("thresholds.yaml")
    universe = D.load_yaml("universe.yaml")
    events = (D.load_yaml("events.yaml") or {}).get("pivot_events", {})
    raw = D.load_raw(fred_key, demo=demo)
    D.save_archives(raw)
    P = build_panel(raw, universe)
    F = compute_flags(P, th, events)
    S = risk_score(P)
    R = run_state_machine(P, F, th)
    country_cape = (th.get("manual_overrides") or {}).get("country_cape") or {}
    assets = score_assets(raw.prices, universe, P, R.current, R.amber, raw.cape, country_cape)
    story = build_narrative(P, F, S, R.current, R.amber, assets, th, raw.cape)
    research = research_watch(offline=demo or not with_research)

    last = P.iloc[-1]
    readings = [{"key": k, "label": lab, "unit": unit, "value": last.get(k)} for k, lab, unit in READINGS]
    weekly = R.weekly.join(S["Total"].rename("score"), how="left").join(P["spx"], how="left")
    flags_now = {k: bool(F[k].iloc[-1]) for k in F.columns if k != "AMBER_COUNT"}
    indices = []
    for row in universe.get("global_indices", []):
        s = raw.prices.get(row["ticker"])
        if s is not None and not s.dropna().empty:
            s = s.dropna()
            indices.append({"ticker": row["ticker"], "name": row["name"], "level": float(s.iloc[-1]),
                            "ret_1m": float(s.iloc[-1] / s.iloc[-22] - 1) if len(s) > 22 else None,
                            "ret_12m": float(s.iloc[-1] / s.iloc[-253] - 1) if len(s) > 253 else None})
    problems = {k: v for k, v in raw.status.items() if not str(v).startswith(("ok", "demo"))}
    snap = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_as_of": str(P.index[-1].date()),
        "demo": raw.demo,
        "regime": {"code": R.current, "name": REGIME_NAMES[R.current], "amber": R.amber, "reason": R.reason, **R.episode},
        "score": {"total": int(S["Total"].iloc[-1]), "band": score_band(S["Total"].iloc[-1]),
                  "components": {k: int(v) for k, v in S.iloc[-1].items() if k != "Total"}},
        "flags": flags_now,
        "narrative": story,
        "readings": readings,
        "assets": assets.to_dict(orient="records"),
        "global_indices": indices,
        "history": [{"date": str(d.date()), **{k: r[k] for k in ("regime", "amber", "score", "spx")}} for d, r in weekly.iterrows()],
        "research": research,
        "data_status": {"problems": problems, "series_checked": len(raw.status), "cape": raw.cape},
    }
    return _clean(snap)


def write_outputs(snap: dict) -> None:
    D.DATA.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snap, indent=1))
    REPORT.write_text(render_markdown(snap))
    row = pd.DataFrame([{"date": snap["data_as_of"], "regime": snap["regime"]["code"], "amber": snap["regime"]["amber"],
                         "score": snap["score"]["total"], "generated_at": snap["generated_at"]}])
    if HISTORY.exists():
        hist = pd.read_csv(HISTORY)
        hist = pd.concat([hist[hist.date != snap["data_as_of"]], row]).sort_values("date")
    else:
        hist = row
    hist.to_csv(HISTORY, index=False)


def render_markdown(snap: dict) -> str:
    n = snap["narrative"]
    out = [f"# Macro Navigator report", "", f"Data as of {snap['data_as_of']} · generated {snap['generated_at'][:16]} UTC"
           + (" · DEMO DATA" if snap["demo"] else ""), "", f"## {n['headline']}", "", n["stance"], ""]
    out += ["## Situation", ""] + [f"- {l}" for l in n["situation"]] + [""]
    out += ["## Outlook by horizon", ""] + [f"- **{k}:** {v}" for k, v in n["horizons"].items()] + [f"- **Global:** {n['global']}", ""]
    if n["amber_signs"]:
        out += ["## Amber signs firing", ""] + [f"- {s}" for s in n["amber_signs"]] + [""]
    out += ["## What would change the call", ""] + [f"- {t}" for t in n["triggers"]] + [""]
    out += ["## Sectors and indices", "", "| Ticker | Name | 1-3 months | 6-12 months | 3-5+ years | 3m return |", "|---|---|---|---|---|---|"]
    for a in snap["assets"]:
        if a.get("status") != "ok":
            continue
        r3 = a.get("ret_3m")
        out.append(f"| {a['ticker']} | {a['name']} | {a['short_view']} | {a['medium_view']} | {a['long_view']} | "
                   f"{'' if r3 is None else format(r3, '+.1%')} |")
    items = (snap.get("research") or {}).get("items", [])[:10]
    if items:
        out += ["", "## Research watch", ""] + [f"- {'NEW · ' if i['new'] else ''}[{i['title']}]({i['link']}) — {i['source']}, module: {i['module']}" for i in items]
    if snap["data_status"]["problems"]:
        out += ["", "## Data problems", ""] + [f"- {k}: {v}" for k, v in snap["data_status"]["problems"].items()]
    out += ["", "_Research framework, not investment advice._", ""]
    return "\n".join(out)


def main(demo: bool = False) -> dict:
    snap = build_snapshot(D.fred_key_from_env(), demo=demo)
    write_outputs(snap)
    if not demo:
        remember((snap.get("research") or {}).get("items", []))
    return snap


if __name__ == "__main__":
    import sys
    s = main(demo="--demo" in sys.argv)
    print(s["narrative"]["headline"])
