#!/usr/bin/env python3
"""
evidence_check.py - how did each signal actually perform? Runs in seconds on the fetched data.

    python fetch_all.py          # once, to download the data
    python evidence_check.py     # prints the scorecard; writes data/store/evidence_report.md and .csv

For every signal it compares what the S&P 500 did after months when the signal was ON with
months when it was OFF: average forward return, how often the return was positive, and the
worst dip within the next 12 months. Thresholds come from the booklet and the papers it cites;
nothing here is fitted to the data. Macro data are lagged to when they were actually published.

Read it with care: 12-month windows overlap, so the number of separate episodes matters more
than the number of months. A signal "holds up" only if its effect points the expected way in
both halves of its history and rests on at least 4 separate episodes.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_panel(store: Path) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401
        candidates = [store / "panel_monthly.parquet", store / "panel_monthly.csv.gz"]
    except ImportError:                       # lighter environment: compressed CSV only
        candidates = [store / "panel_monthly.csv.gz"]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0, parse_dates=True)
    sys.exit(f"No panel found in {store}. Run fetch_all.py first.")


def col(P: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in P.columns and P[n].notna().sum() > 24:
            return P[n].astype(float)
    return pd.Series(np.nan, index=P.index)


def phi(x: pd.Series) -> pd.Series:
    return 0.5 * (1 + (x / math.sqrt(2)).apply(lambda v: math.erf(v) if pd.notna(v) else np.nan))


def build_signals(P: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Each signal: (boolean series with NaN where data are missing, expected direction, basis)."""
    spx = col(P, "px:^GSPC")
    if spx.notna().sum() < 120:
        spx = col(P, "shiller:P")             # Shiller monthly average price as a fallback
    gs10, tb3 = col(P, "fred:GS10", "shiller:GS10"), col(P, "fred:TB3MS")
    spread = gs10 - tb3
    cpi = col(P, "fred:CPIAUCSL", "shiller:CPI")
    cpi_yoy = (cpi / cpi.shift(12) - 1) * 100
    unrate = col(P, "fred:UNRATE")
    u3 = unrate.rolling(3).mean()
    sahm = u3 - u3.rolling(13).min()
    baa_spread = col(P, "fred:BAA") - gs10
    ff = col(P, "fred:FEDFUNDS")
    wti = col(P, "fred:WTISPLC")
    vix = col(P, "px:^VIX", "fred:VIXCLS")
    cape = col(P, "shiller:CAPE")
    ebp = col(P, "ebp:ebp")
    gscpi = col(P, "gscpi:GSCPI")
    sma10 = spx.rolling(10).mean()

    def on(cond: pd.Series, *inputs: pd.Series, lag: int = 0) -> pd.Series:
        valid = pd.concat(inputs, axis=1).notna().all(axis=1)
        out = cond.astype(float).where(valid)
        return out.shift(lag)            # lag = months until the data were published

    rec_prob = phi(-0.5333 - 0.6330 * spread)
    S = {
        "Yield curve inverted (10Y below 3M)": (on(spread < 0, spread), "bearish", "Estrella & Mishkin 1998"),
        "Recession probability 30%+ (NY Fed probit)": (on(rec_prob >= 0.30, spread), "bearish", "NY Fed model; Moench & Stein 2026"),
        "Sahm rule 0.50+": (on(sahm >= 0.50, sahm, lag=1), "bearish", "Sahm 2019"),
        "Credit spread up 0.5pp+ in 6 months (Baa-10Y)": (on(baa_spread.diff(6) >= 0.5, baa_spread.diff(6)), "bearish", "Gilchrist & Zakrajsek 2012"),
        "Excess bond premium above 0.5": (on(ebp > 0.5, ebp, lag=1), "bearish", "Gilchrist & Zakrajsek 2012; Favara et al. 2016"),
        "Trend broken (S&P below 10-month average)": (on(spx < sma10, spx, sma10), "bearish", "Faber 2007; Moskowitz et al. 2012"),
        "Inflation surge (CPI 5%+ and rising)": (on((cpi_yoy >= 5) & (cpi_yoy > cpi_yoy.shift(6)), cpi_yoy, lag=1), "bearish", "Neville et al. 2021"),
        "Fed tightening (funds rate up 1pp+ in 12 months)": (on(ff.diff(12) >= 1.0, ff.diff(12)), "bearish", "Bernanke & Kuttner 2005"),
        "Oil shock (WTI up 50%+ in 12 months)": (on(wti / wti.shift(12) - 1 >= 0.5, wti.shift(12)), "bearish", "Hamilton 1983; Kilian 2009"),
        "Volatility stress (VIX 30+ at month-end)": (on(vix >= 30, vix), "bullish", "Capitulation evidence; Section 1D.7"),
        "Supply-chain pressure (GSCPI 1.0+)": (on(gscpi >= 1.0, gscpi, lag=1), "bearish", "Benigno et al. 2022"),
        "Expensive market (CAPE 30+)": (on(cape >= 30, cape, lag=3), "bearish", "Campbell & Shiller 1998"),
        "Breakdown proxy (trend broken and credit widening)": (
            on((spx < sma10) & (baa_spread.diff(6) >= 0.5), spx, sma10, baa_spread.diff(6)), "bearish", "Booklet Regime 3 logic"),
        "Stagflation proxy (inflation surge and trend broken)": (
            on((cpi_yoy >= 5) & (cpi_yoy > cpi_yoy.shift(6)) & (spx < sma10), cpi_yoy, spx, sma10), "bearish", "Booklet Regime 3; Neville et al. 2021"),
    }
    return S, spx


def forward(spx: pd.Series, h: int) -> pd.Series:
    return spx.shift(-h) / spx - 1


def worst_dip_12m(spx: pd.Series) -> pd.Series:
    vals = spx.to_numpy()
    out = np.full(len(vals), np.nan)
    for i in range(len(vals) - 12):
        window = vals[i + 1:i + 13]
        if np.isfinite(vals[i]) and np.isfinite(window).all():
            out[i] = window.min() / vals[i] - 1
    return pd.Series(out, index=spx.index)


def episodes(flag: pd.Series) -> int:
    f = flag.dropna().astype(bool)
    return int(((f) & (~f.shift(1, fill_value=False))).sum())


def score(flag: pd.Series, fwd: dict, dip: pd.Series, direction: str, long_h: int) -> dict:
    df = pd.DataFrame({"flag": flag, **{f"r{h}": r for h, r in fwd.items()}, "dip": dip}).dropna(subset=["flag", "r12"])
    if df.empty:
        return {}
    if df.flag.sum() == 0:
        return {"from": df.index[0].strftime("%Y-%m"), "months_on": 0, "episodes": 0, "verdict": "Never on"}
    on_, off = df[df.flag == 1], df[df.flag == 0]
    sign = -1 if direction == "bearish" else 1
    half = df.index[len(df) // 2]
    halves = [df[df.index < half], df[df.index >= half]]
    agree = []
    for part in halves:
        a, b = part[part.flag == 1].r12, part[part.flag == 0].r12
        agree.append(len(a) > 0 and len(b) > 0 and sign * (a.mean() - b.mean()) > 0)
    eps = episodes(df.flag)
    if eps < 4:
        verdict = "Too few episodes"
    elif all(agree):
        verdict = "Holds up"
    elif any(agree):
        verdict = "Mixed"
    else:
        verdict = "Not supported"
    row = {
        "from": df.index[0].strftime("%Y-%m"), "months_on": int(on_.shape[0]), "share_on": on_.shape[0] / len(df),
        "episodes": eps,
        "r3_on": on_.r3.mean(), "r3_off": off.r3.mean(),
        "r12_on": on_.r12.mean(), "r12_off": off.r12.mean(),
        "pos12_on": (on_.r12 > 0).mean(), "pos12_off": (off.r12 > 0).mean(),
        "dip_on": on_.dip.mean(), "dip_off": off.dip.mean(),
        "verdict": verdict,
    }
    if long_h:
        lo = df.dropna(subset=[f"r{long_h}"])
        a, b = lo[lo.flag == 1][f"r{long_h}"], lo[lo.flag == 0][f"r{long_h}"]
        row["r_long_on"], row["r_long_off"] = a.mean(), b.mean()
    return row


def pct(x, digits=1):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{digits}f}%"


VERDICT_CLASS = {"Holds up": "good", "Mixed": "mixed", "Not supported": "bad", "Too few episodes": "muted", "Never on": "muted"}


def write_html(store: Path, out: pd.DataFrame | None, period: str, base_line: str) -> Path:
    """Self-contained report the one-click launcher opens in the browser."""
    from datetime import datetime
    from html import escape as e

    cat_path = store / "catalog.csv"
    cat = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame()
    parts = []
    if out is not None and len(out):
        live = out[(out.now == "ON") & out.verdict.notna()]
        chips = "".join(f'<li><span class="v {VERDICT_CLASS.get(r.verdict, "muted")}">{e(r.verdict)}</span> {e(r.signal)}</li>'
                        for _, r in live.iterrows()) or "<li>No signal is on this month.</li>"
        parts.append(f"<h2>Signals on now ({len(live)} of {len(out)})</h2><ul class='now'>{chips}</ul>")
        rows = []
        for _, r in out.iterrows():
            v = r.get("verdict")
            if pd.isna(v):
                rows.append(f"<tr><td>{e(r.signal)}</td><td>{e(r.expected)}</td><td colspan='9' class='dim'>no data</td>"
                            f"<td></td></tr>")
                continue
            if v == "Never on":
                rows.append(f"<tr><td>{e(r.signal)}</td><td>{e(r.expected)}</td><td class='d'>{e(str(r['from']))}</td><td class='n'>0</td>"
                            f"<td colspan='7' class='dim'>never switched on in this history</td>"
                            f"<td><span class='v muted'>Never on</span></td></tr>")
                continue
            now = "<b>ON</b>" if r.now == "ON" else e(str(r.now))
            rows.append(
                f"<tr><td>{e(r.signal)}<div class='basis'>{e(str(r.basis))}</div></td><td>{e(r.expected)}</td>"
                f"<td class='d'>{e(str(r['from']))}</td><td class='n'>{int(r.episodes)}</td>"
                f"<td class='n'>{pct(r.r12_on)}</td><td class='n'>{pct(r.r12_off)}</td>"
                f"<td class='n'>{r.pos12_on:.0%}</td><td class='n'>{r.pos12_off:.0%}</td>"
                f"<td class='n'>{pct(r.dip_on, 0)}</td><td class='n'>{pct(r.dip_off, 0)}</td><td>{now}</td>"
                f"<td><span class='v {VERDICT_CLASS.get(v, 'muted')}'>{e(v)}</span></td></tr>")
        parts.append(
            "<h2>Signal scorecard</h2><div class='scroll'><table><thead><tr><th>Signal</th><th>Expected</th><th>Since</th>"
            "<th>Episodes</th><th>12m after ON</th><th>12m after OFF</th><th>Rose after ON</th><th>Rose after OFF</th>"
            "<th>Worst dip ON</th><th>Worst dip OFF</th><th>Now</th><th>Verdict</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>"
            "<p class='note'>Returns are S&amp;P 500 price changes over the next 12 months, without dividends. Twelve-month "
            "windows overlap, so the number of separate episodes matters more than the number of months. "
            "<b>Holds up</b> means the effect pointed the expected way in both halves of history across at least 4 episodes. "
            "Macro data are lagged to their publication dates. This scores single signals, not the full engine.</p>")
    else:
        parts.append("<h2>No scorecard yet</h2><p>Not enough S&amp;P 500 history was downloaded. Check the data coverage "
                     "below and the terminal window, then run again.</p>")
    if len(cat):
        lines = []
        for src in ("fred", "yahoo", "files"):
            sub = cat[cat.source == src]
            if len(sub):
                ok = sub.status.astype(str).str.startswith(("ok", "cached")).sum()
                first = pd.to_datetime(sub["first"], errors="coerce").min()
                lines.append(f"<tr><td>{src}</td><td class='n'>{ok} of {len(sub)}</td>"
                             f"<td>{'' if pd.isna(first) else first.strftime('%Y')}</td></tr>")
        bad = cat[~cat.status.astype(str).str.startswith(("ok", "cached"))]
        fails = "".join(f"<li>{e(str(r.source))}: {e(str(r['name']))}: {e(str(r.status)[:110])}</li>" for _, r in bad.head(60).iterrows())
        parts.append("<h2>Data coverage</h2><table class='small'><thead><tr><th>Source</th><th>Loaded</th><th>Earliest year</th>"
                     "</tr></thead><tbody>" + "".join(lines) + "</tbody></table>"
                     + (f"<details><summary>{len(bad)} item(s) need attention</summary><ul class='fails'>{fails}</ul></details>" if len(bad) else
                        "<p>Every series loaded.</p>"))
    css = """
    :root{--bg:#F6F7F4;--ink:#15263B;--muted:#5D6B7A;--line:#D9DED8;--card:#FFFFFF;
      --good:#2E7D5B;--goodbg:#E3F1EA;--mixed:#8A5A12;--mixedbg:#F8EBD3;--bad:#A63A32;--badbg:#F5DEDB;--mutedbg:#ECEFEA}
    @media (prefers-color-scheme: dark){:root{--bg:#10161F;--ink:#E7ECF2;--muted:#9AA7B5;--line:#2A3442;--card:#161E29;
      --good:#7FCFA8;--goodbg:#173528;--mixed:#E9C27A;--mixedbg:#3A2C12;--bad:#F0A39B;--badbg:#3D1D1A;--mutedbg:#1F2833}}
    body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;
      font-variant-numeric:tabular-nums}
    main{max-width:1180px;margin:0 auto;padding:40px 24px 64px}
    h1{font-size:2.2rem;margin:0 0 4px;letter-spacing:-.02em} h2{font-size:1.2rem;margin:36px 0 12px}
    .sub{color:var(--muted);margin:0}
    .scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
    table{border-collapse:collapse;width:100%;font-size:.9rem} table.small{width:auto;background:var(--card);border:1px solid var(--line)}
    th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    th{position:sticky;top:0;background:var(--card);font-weight:600;color:var(--muted);font-size:.82rem;line-height:1.25}
    th:first-child,td:first-child{min-width:250px} th,td{padding:8px 10px}
    td.n{text-align:right;white-space:nowrap} td.d{white-space:nowrap} td.dim{color:var(--muted)} .basis{color:var(--muted);font-size:.8rem}
    .v{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:600;font-size:.85rem;white-space:nowrap}
    .v.good{color:var(--good);background:var(--goodbg)} .v.mixed{color:var(--mixed);background:var(--mixedbg)}
    .v.bad{color:var(--bad);background:var(--badbg)} .v.muted{color:var(--muted);background:var(--mutedbg)}
    ul.now{list-style:none;padding:0;margin:0} ul.now li{margin:6px 0}
    .note{color:var(--muted);font-size:.9rem;max-width:80ch} summary{cursor:pointer;font-weight:600;margin:8px 0} ul.fails{color:var(--muted);font-size:.9rem}
    footer{color:var(--muted);font-size:.85rem;margin-top:40px}"""
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Evidence check</title><style>{css}</style></head><body><main><h1>Evidence check</h1>"
            f"<p class='sub'>{e(period)}</p><p class='sub'>{e(base_line)}</p>{''.join(parts)}"
            f"<footer>Built {datetime.now():%d %b %Y, %H:%M} from {e(str(store))}. Research framework, not investment advice.</footer>"
            f"</main></body></html>")
    path = store / "evidence_report.html"
    path.write_text(html, encoding="utf-8")
    return path



def default_store() -> Path:
    """data/store inside the project when run from scripts/, otherwise next to this file."""
    here = Path(__file__).resolve().parent
    return (here.parent if here.name == "scripts" else here) / "data" / "store"

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(default_store()))
    ap.add_argument("--since", default=None, help="evaluate from this date, e.g. 1960-01")
    args = ap.parse_args(argv)
    store = Path(args.store)
    P = load_panel(store)
    if args.since:
        P = P[P.index >= pd.Timestamp(args.since)]
    S, spx = build_signals(P)
    if spx.notna().sum() < 120:
        msg = "Not enough S&P 500 history in the data to score signals."
        print(f"\n{msg} Check the fetch summary for failed downloads, then run again.")
        path = write_html(store, None, msg, "")
        print(f"Report with data coverage: {path}")
        return 1
    fwd = {h: forward(spx, h) for h in (1, 3, 12, 120)}
    dip = worst_dip_12m(spx)
    base = pd.DataFrame({"r12": fwd[12]}).dropna()
    rows = []
    for name, (flag, direction, basis) in S.items():
        r = score(flag, fwd, dip, direction, 120 if "CAPE" in name else 0)
        now = flag.dropna()
        status_now = "ON" if len(now) and now.iloc[-1] == 1 else ("off" if len(now) else "n/a")
        rows.append({"signal": name, "expected": direction, "basis": basis, "now": status_now, **r})
    out = pd.DataFrame(rows)
    out.to_csv(store / "evidence_scorecard.csv", index=False)

    print(f"\nEvidence check  S&P 500 month-ends {spx.dropna().index[0]:%Y-%m} to {spx.dropna().index[-1]:%Y-%m}; "
          f"base rate: 12-month return positive {(base.r12 > 0).mean():.0%} of the time, average {base.r12.mean():+.1%}\n")
    hdr = f"{'Signal':<52}{'From':>8}{'Eps':>5}{'12m ON':>9}{'12m OFF':>9}{'Up% ON':>8}{'Up% OFF':>8}{'Dip ON':>8}{'Dip OFF':>8}  {'Now':<4} Verdict"
    print(hdr)
    print("-" * len(hdr))
    for _, r in out.iterrows():
        if "verdict" not in r or pd.isna(r.get("verdict")):
            print(f"{r.signal[:51]:<52}{'':>8}{'':>5}{'no data':>9}")
            continue
        if r.verdict == "Never on":
            print(f"{r.signal[:51]:<52}{r['from']:>8}{0:>5}{'never switched on in this history':>42}  {r.now:<4}")
            continue
        print(f"{r.signal[:51]:<52}{r['from']:>8}{int(r.episodes):>5}{pct(r.r12_on):>9}{pct(r.r12_off):>9}"
              f"{r.pos12_on:>8.0%}{r.pos12_off:>8.0%}{pct(r.dip_on, 0):>8}{pct(r.dip_off, 0):>8}  {r.now:<4} {r.verdict}")
    if "r_long_on" in out.columns:
        c = out[out.signal.str.contains("CAPE")].iloc[0]
        if pd.notna(c.get("r_long_on")):
            print(f"\nCAPE 30+ over 10 years: average {pct(c.r_long_on)} total price change when ON vs {pct(c.r_long_off)} when OFF.")
    print("\nColumns: Eps = separate episodes; 12m ON/OFF = average S&P 500 price return over the next 12 months;"
          "\nUp% = share of months followed by a positive 12-month return; Dip = average worst fall within 12 months."
          "\nPrice returns only (no dividends). Overlapping windows: judge by episodes, not months.")

    md = ["# Evidence check", "",
          f"S&P 500 month-ends {spx.dropna().index[0]:%Y-%m} to {spx.dropna().index[-1]:%Y-%m}. "
          f"Base rate: next-12-month return positive {(base.r12 > 0).mean():.0%} of months, average {base.r12.mean():+.1%}.", "",
          "| Signal | Expected | From | Episodes | 12m ON | 12m OFF | Up% ON | Up% OFF | Dip ON | Dip OFF | Now | Verdict | Basis |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        if pd.isna(r.get("verdict")):
            md.append(f"| {r.signal} | {r.expected} | | | no data | | | | | | {r.now} | | {r.basis} |")
            continue
        if r.verdict == "Never on":
            md.append(f"| {r.signal} | {r.expected} | {r['from']} | 0 | | | | | | | {r.now} | Never on | {r.basis} |")
            continue
        md.append(f"| {r.signal} | {r.expected} | {r['from']} | {int(r.episodes)} | {pct(r.r12_on)} | {pct(r.r12_off)} | "
                  f"{r.pos12_on:.0%} | {r.pos12_off:.0%} | {pct(r.dip_on, 0)} | {pct(r.dip_off, 0)} | {r.now} | {r.verdict} | {r.basis} |")
    md += ["", "Price returns only. 12-month windows overlap, so episodes matter more than months. "
           "'Holds up' = effect in the expected direction in both halves of the sample and at least 4 episodes. "
           "This scores single signals; it is not a backtest of the full engine."]
    (store / "evidence_report.md").write_text("\n".join(md))
    period = f"S&P 500 month-ends {spx.dropna().index[0]:%b %Y} to {spx.dropna().index[-1]:%b %Y}"
    base_line = (f"Base rate: the next 12 months were positive {(base.r12 > 0).mean():.0%} of the time, "
                 f"average {base.r12.mean():+.1%}.")
    html_path = write_html(store, out, period, base_line)
    print(f"\nSaved {store / 'evidence_scorecard.csv'}, {store / 'evidence_report.md'} and {html_path}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Leave immediately: every file is already written, and skipping interpreter teardown avoids
    # crashes some compiled libraries (and daemon threads) cause while Python shuts down.
    os._exit(code or 0)
