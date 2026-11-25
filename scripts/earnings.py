#!/usr/bin/env python3
"""
earnings.py - the earnings lab: is there an edge in entering stocks right after earnings announcements?

Announcement times: SEC 8-K filings with Item 2.02 ("Results of Operations"), accepted-at timestamps (2001-).
Phase 1 (daily, 2001-): enter at the reaction day's open or close, exit 0-60 trading days later.
Phase 2 (intraday, 2016-, needs ALPACA_KEY_ID / ALPACA_SECRET_KEY): enter 1-30 minutes after the release in after-hours,
         or at / after the next open; exit 1 minute to the close. Only the needed minute prices are kept.

Every return is market-adjusted (minus SPY over the same window) and net of costs. Signals use only information known at
the entry moment. Statistics are clustered by announcement date. Verdicts: Solid = positive net return in both periods,
t >= 3 in the later one (Harvey, Liu & Zhu 2016 hurdle for many tests) and better than equally large moves on days without
earnings (control); Not proven = positive with t >= 2 in the later period; Fails otherwise.
Research context: Martineau (2022) finds no post-earnings drift for large stocks since 2006; Christensen, Timmermann &
Veliyev (2025) document jumps within seconds of announcements. The lab tests whether anything is left after costs.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEC_UA = {"User-Agent": "Macro Navigator personal research tool " + (os.environ.get("SEC_CONTACT") or "macro-navigator@users.noreply.github.com").strip(),
          "Accept-Encoding": "gzip, deflate"}
ALPACA = "https://data.alpaca.markets/v2/stocks/bars"
ET = "America/New_York"
DAILY_EXITS = (0, 1, 2, 5, 10, 20, 60)               # trading days after the reaction day (0 = that day's close)
REG_ENTRIES = (0, 1, 5, 15, 30, 60)                    # minutes after 09:30
REG_HOLDS = (1, 5, 15, 30, 60)
AH_ENTRIES = (1, 5, 15, 30)                            # minutes after the release (after-close announcements)
COST = {"daily": 0.0010, "ah": 0.0040, "open15": 0.0020, "reg": 0.0010}
SPLIT_DAILY = ("2001-01-01", "2012-12-31", "2013-01-01")
SPLIT_INTRA = ("2016-01-01", "2020-12-31", "2021-01-01")
ETFS = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC", "SMH", "GLD", "TLT"}


# --------------------------------------------------------------------------- storage (parquet if available, else pickle)

def _save(df: pd.DataFrame, path: Path) -> None:
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path)
    except Exception:
        df.to_pickle(path.with_suffix(".pkl"))


def _load(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        pass
    pk = path.with_suffix(".pkl")
    return pd.read_pickle(pk) if pk.exists() else pd.DataFrame()


def _exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".pkl").exists()


def _mtime(path: Path) -> float:
    for p in (path, path.with_suffix(".pkl")):
        if p.exists():
            return p.stat().st_mtime
    return 0.0


# --------------------------------------------------------------------------- SEC announcement times

def sec_json(url: str, cache: Path, ttl_h: float):
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 3600 < ttl_h:
        return json.loads(cache.read_text())
    r = requests.get(url, headers=SEC_UA, timeout=60)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    time.sleep(0.12)                                   # under the SEC's 10 requests a second
    return r.json()


def ticker_ciks(store: Path) -> dict:
    d = sec_json("https://www.sec.gov/files/company_tickers.json", store / "sec" / "company_tickers.json", 168)
    return {str(v["ticker"]).upper().replace(".", "-"): int(v["cik_str"]) for v in d.values()}


def _rows(block: dict, ticker: str) -> list:
    out = []
    for form, acc, items in zip(block.get("form", []), block.get("acceptanceDateTime", []), block.get("items", [])):
        if form == "8-K" and "2.02" in str(items) and acc:
            out.append((ticker, str(acc)))
    return out


def announcements(store: Path, tickers: list, budget_s: float = 600) -> pd.DataFrame:
    t0 = time.time()
    ciks = ticker_ciks(store)
    rows = []
    for n_done, tk in enumerate(tickers):
        if n_done and n_done % 50 == 0:
            print(f"  SEC filings: {n_done}/{len(tickers)} companies", flush=True)
        cik = ciks.get(tk)
        if cik is None:
            continue
        cache = store / "sec" / "subs" / f"{cik}.json"
        try:
            if time.time() - t0 > budget_s:             # out of time: use what is cached; the rest comes next run
                if not cache.exists():
                    continue
                subs = json.loads(cache.read_text())
            else:
                subs = sec_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", cache, 160)
        except Exception:
            continue
        rows += _rows(subs.get("filings", {}).get("recent", {}), tk)
        for f in subs.get("filings", {}).get("files", []):
            if str(f.get("filingTo", "")) < "2001-01-01":
                continue
            try:
                older = sec_json(f"https://data.sec.gov/submissions/{f['name']}", store / "sec" / "subs" / f["name"], 1e9)
            except Exception:
                continue
            rows += _rows(older, tk)
    if not rows:
        return pd.DataFrame(columns=["ticker", "ts"])
    df = pd.DataFrame(rows, columns=["ticker", "raw"]).drop_duplicates()
    ts = pd.to_datetime(df.raw.str.replace("Z", "", regex=False), errors="coerce")
    df = df.assign(ts=ts).dropna(subset=["ts"])
    # EDGAR writes these as ...Z; decide from the data whether they are really UTC or Eastern clock time
    h = df.ts.dt.hour
    eastern_like = ((h >= 16) & (h < 18)).mean() + ((h >= 6) & (h < 9)).mean()
    utc_like = ((h >= 20) & (h < 22)).mean() + ((h >= 10) & (h < 13)).mean()
    if utc_like > eastern_like:
        df["ts"] = df.ts.dt.tz_localize("UTC").dt.tz_convert(ET).dt.tz_localize(None)
    df = df.sort_values(["ticker", "ts"])
    keep = df.groupby("ticker").ts.diff().dt.days.fillna(999) > 20      # one announcement per quarter (drop amendments)
    return df[keep][["ticker", "ts"]].reset_index(drop=True)


def classify(ann: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Reaction day = first session that could trade on the news; timing bmo / amc / intraday."""
    days = pd.DatetimeIndex(days).sort_values()
    d = ann.ts.dt.normalize()
    mins = ann.ts.dt.hour * 60 + ann.ts.dt.minute
    pos = days.searchsorted(d)                                           # first trading day >= announcement date
    is_td = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == d)
    timing = np.where(~is_td, "closed", np.where(mins < 570, "bmo", np.where(mins >= 960, "amc", "intraday")))
    rpos = np.where(timing == "amc", pos + 1, pos)
    ok = rpos < len(days)
    out = ann[ok].copy()
    out["timing"] = np.where(timing[ok] == "closed", "bmo", timing[ok])
    out["reaction"] = days[rpos[ok]]
    out["rpos"] = rpos[ok]
    out["prevday"] = days[np.maximum(rpos[ok] - 1, 0)]
    return out[(out.timing != "intraday") & (out.rpos >= 1)].reset_index(drop=True)


# --------------------------------------------------------------------------- daily prices (open, close, volume)

def daily_prices(store: Path, tickers: list) -> dict:
    """Wide frames of adjusted open / close / volume; refreshed weekly, full history once."""
    import yfinance as yf
    paths = {k: store / f"earn_{k}.parquet" for k in ("open", "close", "volume")}
    old = {k: _load(p) for k, p in paths.items()}
    fresh = _exists(paths["close"]) and (time.time() - _mtime(paths["close"])) / 86400 < 6
    have = set(old["close"].columns) if not old["close"].empty else set()
    tried_path = store / "earn_no_data.json"
    no_data = set(json.loads(tried_path.read_text())) if tried_path.exists() else set()
    missing = sorted(set(tickers) - have - no_data)
    if fresh and not missing:
        return old
    groups = [(missing, "max")] + ([] if fresh else [(sorted(set(tickers) & have), "6mo")])
    new = {k: [] for k in paths}
    for names, period in groups:
        for i in range(0, len(names), 80):
            chunk = names[i: i + 80]
            try:
                raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True, progress=False, group_by="column", threads=True)
            except Exception:
                continue
            if raw is None or raw.empty:
                continue
            for k, col in (("open", "Open"), ("close", "Close"), ("volume", "Volume")):
                part = raw[col] if isinstance(raw.columns, pd.MultiIndex) else raw[[col]].rename(columns={col: chunk[0]})
                part.index = pd.to_datetime(part.index).tz_localize(None)
                new[k].append(part)
    got = set().union(*[set(f.dropna(axis=1, how="all").columns) for f in new["close"]]) if new["close"] else set()
    tried_path.write_text(json.dumps(sorted(no_data | (set(missing) - got))))
    if missing:
        print(f"  daily prices: {len(set(missing) & got)} of {len(missing)} new companies found (the rest have no data left)", flush=True)
    out = {}
    for k, p in paths.items():
        frames = [old[k]] + new[k] if not old[k].empty else new[k]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            out[k] = pd.DataFrame()
            continue
        merged = frames[0]
        for f in frames[1:]:
            merged = f.combine_first(merged)
        merged = merged.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        _save(merged, p)
        out[k] = merged
    return out


# --------------------------------------------------------------------------- index membership at the time (survivorship)

def load_membership(path) -> pd.DataFrame | None:
    """CSV with ticker,start_date,end_date (blank end = still a member)."""
    if not path or not Path(path).exists():
        return None
    m = pd.read_csv(path)
    m["ticker"] = m.ticker.astype(str).str.replace(".", "-", regex=False)
    m["start"] = pd.to_datetime(m.start_date, errors="coerce")
    m["end"] = pd.to_datetime(m.end_date, errors="coerce").fillna(pd.Timestamp("2100-01-01"))
    return m[["ticker", "start", "end"]].dropna()


def member_at(df: pd.DataFrame, mem: pd.DataFrame) -> pd.Series:
    """True where the company was in the index on the reaction date."""
    if df.empty:
        return pd.Series(dtype=bool)
    x = df[["ticker", "reaction"]].reset_index().merge(mem, on="ticker", how="left")
    x["ok"] = (x.reaction >= x.start) & (x.reaction <= x.end)
    return x.groupby("index").ok.any().reindex(df.index, fill_value=False)


# --------------------------------------------------------------------------- statistics

def cluster_stats(dates: pd.Series, r: pd.Series) -> dict | None:
    """Mean and t-statistic with one observation per announcement date (events on the same day are not independent)."""
    m = r.notna()
    if m.sum() < 30:
        return None
    per_day = r[m].groupby(dates[m]).mean()
    n = len(per_day)
    if n < 15 or per_day.std() == 0:
        return None
    se = per_day.std() / math.sqrt(n)
    return {"events": int(m.sum()), "dates": n, "mean": float(r[m].mean()), "t": float(per_day.mean() / se), "se": float(se),
            "hit": float((r[m] > 0).mean())}


def vs_control(late, control_late) -> dict | None:
    """Earnings-specific part: the trade's return minus the same trade after equal-sized moves on ordinary days."""
    if not late or not control_late:
        return None
    d = late["mean"] - control_late["mean"]
    se = math.sqrt(late["se"] ** 2 + control_late["se"] ** 2)
    return {"mean": d, "t": d / se if se else 0.0}


def verdict(early, late, control_late) -> str:
    if not late:
        return "Too little data"
    vc = vs_control(late, control_late)
    beats = control_late is None or (vc is not None and vc["mean"] > 0 and vc["t"] >= 2)
    if late["mean"] > 0 and late["t"] >= 3 and early and early["mean"] > 0 and beats:
        return "Solid"
    if late["mean"] > 0 and late["t"] >= 2:
        return "Not proven"
    return "Fails"


def grid_cells(ev: pd.DataFrame, ctrl: pd.DataFrame, specs: list, split: tuple) -> list:
    """specs: (label, entry, exit, condition column, condition value, strategy, return column, cost)."""
    early = (ev.reaction >= split[0]) & (ev.reaction <= split[1])
    late = ev.reaction >= split[2]
    if ctrl is not None and len(ctrl):
        c_late = ctrl.reaction >= split[2]
    cells = []
    for label, entry, exit_, cond, val, strat, rcol, cost in specs:
        if rcol not in ev:
            continue
        sign = 1.0 if strat == "continuation" else -1.0
        dirn = 1.0 if val == "big_up" else -1.0
        sel = ev[cond] == val
        r = sign * dirn * ev[rcol] - cost
        e, l = cluster_stats(ev.reaction[sel & early], r[sel & early]), cluster_stats(ev.reaction[sel & late], r[sel & late])
        cl = None
        if ctrl is not None and len(ctrl) and rcol in ctrl and cond in ctrl:
            cs = ctrl[cond] == val
            cl = cluster_stats(ctrl.reaction[cs & c_late], (sign * dirn * ctrl[rcol] - cost)[cs & c_late])
        cells.append({"question": label, "entry": entry, "exit": exit_, "condition": val, "strategy": strat,
                      "cost": cost, "early": e, "late": l, "control_late": cl, "vs_control": vs_control(l, cl),
                      "verdict": verdict(e, l, cl)})
    return cells


# --------------------------------------------------------------------------- phase 1: daily

def daily_events(ann: pd.DataFrame, px: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    O, C, V = px["open"], px["close"], px["volume"]
    if C.empty or "SPY" not in C:
        return pd.DataFrame(), pd.DataFrame()
    days = C.index
    ev = classify(ann, days)
    ev = ev[ev.ticker.isin(C.columns)]
    rets = C.pct_change()
    sd60 = rets.rolling(60).std().shift(1)
    vol50 = V.rolling(50).mean().shift(1)
    spy_c, spy_o = C["SPY"].values, O["SPY"].values
    spy_ma = C["SPY"].rolling(200).mean().values
    rows = []
    for tk, g in ev.groupby("ticker"):
        c, o = C[tk].values, O[tk].values
        s, v, vb = sd60[tk].values, V[tk].values, vol50[tk].values
        for rp, react, timing in zip(g.rpos.values, g.reaction.values, g.timing.values):
            if rp < 1 or np.isnan(c[rp - 1]) or np.isnan(o[rp]) or np.isnan(c[rp]) or np.isnan(s[rp]) or s[rp] == 0:
                continue
            row = {"ticker": tk, "reaction": pd.Timestamp(react), "timing": timing,
                   "gap": o[rp] / c[rp - 1] - 1, "day1": c[rp] / c[rp - 1] - 1, "sd": s[rp],
                   "vol_ratio": v[rp] / vb[rp] if vb[rp] and not np.isnan(vb[rp]) else np.nan,
                   "spx_broken": bool(spy_c[rp - 1] < spy_ma[rp - 1]) if not np.isnan(spy_ma[rp - 1]) else None}
            for k in DAILY_EXITS:
                j = rp + k
                if j < len(days) and not np.isnan(c[j]):
                    row[f"o_{k}"] = (c[j] / o[rp]) - (spy_c[j] / spy_o[rp])                    # open entry
                    if k > 0:
                        row[f"c_{k}"] = (c[j] / c[rp]) - (spy_c[j] / spy_c[rp])                # close entry
            rows.append(row)
    E = pd.DataFrame(rows)
    if E.empty:
        return E, E
    E["gap_z"] = E.gap / E.sd
    E["day1_z"] = E.day1 / E.sd
    for col, z in (("gap_cond", "gap_z"), ("day1_cond", "day1_z")):
        E[col] = np.where(E[z] >= 2, "big_up", np.where(E[z] <= -2, "big_down", "small"))
    # control: equally large daily moves with no announcement within 5 trading days
    near = set()
    for tk, g in ev.groupby("ticker"):
        for rp in g.rpos.values:
            for k in range(-5, 6):
                near.add((tk, rp + k))
    crow = []
    zc = rets / sd60
    rng = np.random.default_rng(3)
    for tk in E.ticker.unique():
        z = zc[tk].values
        cand = np.where(np.abs(z) >= 2)[0]
        cand = [p for p in cand if (tk, p) not in near and p + 1 < len(days)]
        if len(cand) > 40:
            cand = sorted(rng.choice(cand, 40, replace=False))
        c, o = C[tk].values, O[tk].values
        for rp in cand:
            if np.isnan(c[rp - 1]) or np.isnan(o[rp]) or np.isnan(c[rp]):
                continue
            row = {"ticker": tk, "reaction": days[rp], "day1_cond": "big_up" if z[rp] > 0 else "big_down",
                   "gap_cond": "big_up" if (o[rp] / c[rp - 1] - 1) / sd60[tk].values[rp] >= 2 else
                               ("big_down" if (o[rp] / c[rp - 1] - 1) / sd60[tk].values[rp] <= -2 else "small")}
            for k in DAILY_EXITS:
                j = rp + k
                if j < len(days) and not np.isnan(c[j]):
                    row[f"o_{k}"] = (c[j] / o[rp]) - (spy_c[j] / spy_o[rp])
                    if k > 0:
                        row[f"c_{k}"] = (c[j] / c[rp]) - (spy_c[j] / spy_c[rp])
            crow.append(row)
    return E, pd.DataFrame(crow)


def daily_specs() -> list:
    specs = []
    for strat in ("continuation", "reversal"):
        for val in ("big_up", "big_down"):
            for k in DAILY_EXITS:
                specs.append(("Enter at the reaction-day open after a big gap", "open", f"{k}d" if k else "same-day close",
                              "gap_cond", val, strat, f"o_{k}", COST["daily"]))
                if k > 0:
                    specs.append(("Enter at the reaction-day close after a big first-day move", "close", f"{k}d",
                                  "day1_cond", val, strat, f"c_{k}", COST["daily"]))
    return specs


def market_split(E: pd.DataFrame, C: pd.DataFrame) -> list:
    """The best-known daily question (continuation of a big first-day move over 5 days) split by the S&P's trend."""
    out = []
    for state, mask in (("S&P trend intact", E.spx_broken == False), ("S&P trend broken", E.spx_broken == True)):  # noqa: E712
        for val in ("big_up", "big_down"):
            sel = mask & (E.day1_cond == val)
            r = (1 if val == "big_up" else -1) * E["c_5"] - COST["daily"] if "c_5" in E else pd.Series(dtype=float)
            late = sel & (E.reaction >= SPLIT_DAILY[2])
            out.append({"state": state, "condition": val, "late": cluster_stats(E.reaction[late], r[late])})
    return out


# --------------------------------------------------------------------------- phase 2: intraday (Alpaca)

def alpaca_headers():
    k, s = os.environ.get("ALPACA_KEY_ID", "").strip(), os.environ.get("ALPACA_SECRET_KEY", "").strip()
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s} if k and s else None


def _bars(symbols: str, start: pd.Timestamp, end: pd.Timestamp, H: dict) -> dict:
    params = {"symbols": symbols, "timeframe": "1Min", "start": start.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
              "end": end.tz_convert("UTC").isoformat().replace("+00:00", "Z"), "feed": "sip", "adjustment": "raw", "limit": 10000}
    out = {}
    for _ in range(40):
        _pace()
        r = requests.get(ALPACA, params=params, headers=H, timeout=30)
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        d = r.json()
        for sym, bars in (d.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars)
        tok = d.get("next_page_token")
        if not tok:
            break
        params["page_token"] = tok
    return out


_LAST = [0.0]


def _pace(gap: float = 0.31):
    """At most about 190 requests a minute (Alpaca free plan: 200)."""
    wait = _LAST[0] + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _LAST[0] = time.time()


def _frame(bars: list) -> pd.DataFrame:
    """Minute bars (open, close) indexed by bar start, Eastern time without zone."""
    if not bars:
        return pd.DataFrame(columns=["o", "c"])
    t = pd.to_datetime([b["t"] for b in bars], utc=True).tz_convert(ET).tz_localize(None)
    f = pd.DataFrame({"o": [b["o"] for b in bars], "c": [b["c"] for b in bars]}, index=t).sort_index()
    return f[~f.index.duplicated()]


def _price_at(f: pd.DataFrame, t: pd.Timestamp, after: pd.Timestamp | None = None):
    """Last trade price at time t (close of the last bar starting before t); NaN if nothing since `after`."""
    w = f[f.index < t]
    if after is not None:
        w = w[w.index >= after]
    return float(w.c.iloc[-1]) if len(w) else np.nan


def intraday_points(ev_row, bars: dict, sym: str) -> dict | None:
    s, m = _frame(bars.get(sym, [])), _frame(bars.get("SPY", []))
    if s.empty or m.empty:
        return None
    react = pd.Timestamp(ev_row.reaction)
    pday = pd.Timestamp(ev_row.prevday)
    close_t = pday + pd.Timedelta(hours=16)
    out = {"prev": _price_at(s, close_t), "prev_spy": _price_at(m, close_t)}
    open_t = react + pd.Timedelta(hours=9, minutes=30)
    first = s[(s.index >= open_t) & (s.index < open_t + pd.Timedelta(minutes=5))]
    first_m = m[(m.index >= open_t) & (m.index < open_t + pd.Timedelta(minutes=5))]
    if first.empty or first_m.empty:
        return None
    out["r0"], out["r0_spy"] = float(first.o.iloc[0]), float(first_m.o.iloc[0])
    for k in sorted(set(REG_ENTRIES) | {e + h for e in REG_ENTRIES for h in REG_HOLDS}):
        if k == 0:
            continue
        t = open_t + pd.Timedelta(minutes=k)
        out[f"r{k}"], out[f"r{k}_spy"] = _price_at(s, t, open_t), _price_at(m, t, open_t)
    ct = react + pd.Timedelta(hours=16)
    out["rc"], out["rc_spy"] = _price_at(s, ct, open_t), _price_at(m, ct, open_t)
    if ev_row.timing == "amc":
        rel = pd.Timestamp(ev_row.ts).ceil("min")
        for k in sorted(set(AH_ENTRIES) | {60}):
            t = rel + pd.Timedelta(minutes=k)
            out[f"a{k}"], out[f"a{k}_spy"] = _price_at(s, t, rel), _price_at(m, t, close_t)
    return out


def intraday_fetch(store: Path, ev: pd.DataFrame, budget_s: float) -> pd.DataFrame:
    H = alpaca_headers()
    path = store / "earn_intraday.parquet"
    have = _load(path)
    if H is None:
        return have
    done = set(have["key"]) if not have.empty else set()
    todo = ev[(ev.reaction >= SPLIT_INTRA[0]) & ~(ev.ticker + "|" + ev.reaction.astype(str)).isin(done)]
    todo = todo.sort_values("reaction", ascending=False)                # newest first
    t0, rows, total = time.time(), [], len(done) + len(todo)
    # companies reporting in the same window share one request (up to 8 symbols plus SPY)
    for (prevday, react), grp in todo.groupby(["prevday", "reaction"], sort=False):
        if time.time() - t0 > budget_s:
            break
        react = pd.Timestamp(react)
        start = pd.Timestamp(prevday).replace(hour=15, minute=55).tz_localize(ET)
        end = react.replace(hour=16, minute=1).tz_localize(ET)
        recs = list(grp.itertuples())
        for i in range(0, len(recs), 8):
            chunk = recs[i: i + 8]
            syms = [r.ticker.replace("-", ".") for r in chunk]
            try:
                bars = _bars(",".join(syms + ["SPY"]), start, end, H)
            except Exception:
                continue
            for row, asym in zip(chunk, syms):
                pts = intraday_points(row, bars, asym)
                rows.append({"key": f"{row.ticker}|{react.date()}", "ticker": row.ticker, "reaction": react, "timing": row.timing,
                             "ts": pd.Timestamp(row.ts), **(pts or {"missing": 1.0})})
            if len(rows) // 250 != (len(rows) - len(chunk)) // 250:
                print(f"  minute bars: {len(done) + len(rows)} of {total} announcements", flush=True)
            if len(rows) // 1000 != (len(rows) - len(chunk)) // 1000:     # checkpoint: an interrupted run loses nothing
                _save(pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if not have.empty else pd.DataFrame(rows), path)
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if not have.empty else pd.DataFrame(rows)
        _save(have, path)
    return have


def intraday_cells(I: pd.DataFrame, sd: pd.Series) -> list:
    if I.empty or "r0" not in I:
        return []
    I = I[I.get("missing").isna()] if "missing" in I else I
    I = I.copy()
    sd = sd[~sd.index.duplicated()] if len(sd) else sd
    I["sd"] = sd.reindex(list(zip(I.ticker, pd.to_datetime(I.reaction)))).values if len(sd) else np.nan
    specs, ev = [], I

    def ret(a, b, a_spy, b_spy):
        return (ev[b] / ev[a]) - (ev[b_spy] / ev[a_spy])
    cols = {}
    for e in REG_ENTRIES:
        ek = f"r{e}"
        z = (ev[ek] / ev["prev"] - 1) / ev["sd"]
        cols[f"cond_r{e}"] = np.where(z >= 2, "big_up", np.where(z <= -2, "big_down", "small"))
        for h in REG_HOLDS:
            if f"r{e + h}" in ev:
                cols[f"ret_r{e}_{h}"] = ret(ek, f"r{e + h}", f"{ek}_spy", f"r{e + h}_spy")
        cols[f"ret_r{e}_c"] = ret(ek, "rc", f"{ek}_spy", "rc_spy")
    for e in AH_ENTRIES:
        ek = f"a{e}"
        if ek not in ev:
            continue
        z = (ev[ek] / ev["prev"] - 1) / ev["sd"]
        cols[f"cond_a{e}"] = np.where(z >= 2, "big_up", np.where(z <= -2, "big_down", "small"))
        if e < 60:
            cols[f"ret_a{e}_60"] = (ev["a60"] / ev[ek]) - (ev["a60_spy"] / ev[f"{ek}_spy"])
        cols[f"ret_a{e}_o"] = (ev["r0"] / ev[ek]) - (ev["r0_spy"] / ev[f"{ek}_spy"])
        cols[f"ret_a{e}_c"] = (ev["rc"] / ev[ek]) - (ev["rc_spy"] / ev[f"{ek}_spy"])
    ev = pd.concat([ev, pd.DataFrame(cols, index=ev.index)], axis=1)
    for strat in ("continuation", "reversal"):
        for val in ("big_up", "big_down"):
            for e in REG_ENTRIES:
                cost = COST["open15"] if e <= 15 else COST["reg"]
                for h in REG_HOLDS:
                    specs.append(("Enter after the open (minutes after 09:30)", f"open+{e}m", f"hold {h}m", f"cond_r{e}", val, strat, f"ret_r{e}_{h}", cost))
                specs.append(("Enter after the open (minutes after 09:30)", f"open+{e}m", "to the close", f"cond_r{e}", val, strat, f"ret_r{e}_c", cost))
            for e in AH_ENTRIES:
                for x, lab in (("60", "to release+60m"), ("o", "to the next open"), ("c", "to the next close")):
                    if e < 60 or x != "60":
                        specs.append(("Enter in after-hours after the release", f"release+{e}m", lab, f"cond_a{e}", val, strat, f"ret_a{e}_{x}", COST["ah"]))
    return grid_cells(ev, None, specs, SPLIT_INTRA)


# --------------------------------------------------------------------------- run

def summarise(cells: list, what: str) -> str:
    solid = [c for c in cells if c["verdict"] == "Solid"]
    np_ = [c for c in cells if c["verdict"] == "Not proven"]
    if not cells:
        return f"{what}: not run yet."
    if solid:
        best = max(solid, key=lambda c: c["late"]["mean"])
        return (f"{what}: {len(solid)} of {len(cells)} variants passed every test. Strongest: {best['strategy']} after a "
                f"{best['condition'].replace('_', ' ')}, entry {best['entry']}, exit {best['exit']}: "
                f"{best['late']['mean'] * 100:+.2f}% per trade after costs (t = {best['late']['t']:.1f}).")
    return (f"{what}: none of {len(cells)} variants passed every test after costs"
            + (f"; {len(np_)} looked promising in recent years only (not proven)." if np_ else "."))


def run(store: Path, tickers: list, budget_s: float = 1500, sec_budget_s: float | None = None,
        membership: pd.DataFrame | None = None, current: list | None = None) -> dict:
    t0 = time.time()
    stocks = [t for t in tickers if not str(t).startswith("^") and t not in ETFS]
    ann = announcements(store, stocks, budget_s=sec_budget_s if sec_budget_s is not None else min(600, budget_s / 2))
    if ann.empty:
        return {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "error": "no SEC announcement data"}
    px = daily_prices(store, sorted(set(stocks) | {"SPY"}))
    E, Ctl = daily_events(ann, px)
    survivorship = None
    if len(E) and membership is not None:
        E["pit"], Ctl["pit"] = member_at(E, membership).values, member_at(Ctl, membership).values
        cur = set(current or [])
        E["current"], Ctl["current"] = E.ticker.isin(cur).values, Ctl.ticker.isin(cur).values
        cells_today = grid_cells(E[E.current], Ctl[Ctl.current], daily_specs(), SPLIT_DAILY)
        Em, Cm = E[E.pit], Ctl[Ctl.pit]
        survivorship = {"events_members_at_the_time": int(len(Em)), "events_today_members_full_history": int(E.current.sum()),
                        "former_member_companies_with_data": int(E[~E.current].ticker.nunique()),
                        "cells_today_members": cells_today}
    else:
        Em, Cm = E, Ctl
    cells = grid_cells(Em, Cm, daily_specs(), SPLIT_DAILY) if len(Em) else []
    split = market_split(Em, px["close"]) if len(Em) else []
    if len(E):
        _save(E, store / "earn_events.parquet")
    # phase 2
    intra_cells, coverage = [], None
    ev2 = classify(ann, px["close"].index) if not px["close"].empty else pd.DataFrame()
    if len(ev2) and membership is not None:
        ev2 = ev2[member_at(ev2, membership).values]                      # members at the time only
    if len(ev2):
        I = intraday_fetch(store, ev2, budget_s=max(0, budget_s - (time.time() - t0)))
        total = int((ev2.reaction >= SPLIT_INTRA[0]).sum())
        coverage = {"fetched": int(len(I)), "total": total, "alpaca_keys": alpaca_headers() is not None}
        if len(I):
            sd = E.set_index(["ticker", "reaction"]).sd if len(E) else pd.Series(dtype=float)
            intra_cells = intraday_cells(I, sd)
    res = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "seconds": round(time.time() - t0),
           "announcements": int(len(ann)), "stocks": int(ann.ticker.nunique()),
           "daily": {"events": int(len(Em)), "control_events": int(len(Cm)), "periods": "2001–2012 vs 2013–now", "cells": cells,
                     "survivorship": survivorship,
                     "by_market": split, "summary": summarise(cells, "Days after earnings")},
           "intraday": {"coverage": coverage, "periods": "2016–2020 vs 2021–now", "cells": intra_cells,
                        "summary": summarise(intra_cells, "Minutes and hours after earnings") if intra_cells else
                        ("Minutes and hours: add the Alpaca keys to start." if not (coverage or {}).get("alpaca_keys")
                         else "Minutes and hours: downloading history; results appear once enough events are in.")},
           "method": ("Market-adjusted (minus SPY), net of costs (after-hours 0.40%, first 15 minutes 0.20%, otherwise 0.10% round "
                      "trip), signals known at entry only, t-statistics clustered by announcement date. Solid = positive in both "
                      "periods, t >= 3 recently, and the earnings-specific part (minus the same trade after equally large moves "
                      "on days without earnings) positive with t >= 2." + (" Counted only while the company was in the S&P 500."
                      if membership is not None else ""))}
    return res


def run_if_stale(store: Path, tickers: list, max_age_days: float = 6.5, budget_s: float = 1500, force: bool = False) -> dict | None:
    path = store / "earnings_results.json"
    stale = not path.exists() or (time.time() - path.stat().st_mtime) / 86400 >= max_age_days
    intraday_pending = False
    if path.exists():
        try:
            old = json.loads(path.read_text())
            cov = (old.get("intraday") or {}).get("coverage") or {}
            intraday_pending = bool(alpaca_headers()) and cov.get("fetched", 0) < 0.98 * max(1, cov.get("total", 1))
        except Exception:
            pass
    if not (stale or intraday_pending or force):
        return json.loads(path.read_text())
    try:
        res = run(store, tickers, budget_s=budget_s)
    except Exception as err:
        print(f"  earnings lab skipped ({err.__class__.__name__}: {err})")
        return json.loads(path.read_text()) if path.exists() else None
    from engine import finite
    path.write_text(json.dumps(finite(res), indent=1, allow_nan=False, default=str))
    d, i = res.get("daily", {}), res.get("intraday", {})
    print(f"  earnings lab: {res.get('announcements', 0)} announcements, {d.get('events', 0)} daily events, "
          f"intraday {((i.get('coverage') or {}).get('fetched', 0))}/{((i.get('coverage') or {}).get('total', 0))} in {res.get('seconds', 0)}s")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(HERE.parent / "data" / "store"))
    ap.add_argument("--tickers", default="", help="comma list; default: the site universe")
    a = ap.parse_args()
    if a.tickers:
        tk = [t.strip() for t in a.tickers.split(",") if t.strip()]
    else:
        tk = [u["s"] for u in json.loads((Path(a.store) / "site_universe.json").read_text())]
    r = run_if_stale(Path(a.store), tk, force=True)
    print(json.dumps({k: v for k, v in (r or {}).items() if k in ("announcements", "stocks", "seconds")}, indent=1))
    print((r or {}).get("daily", {}).get("summary"))
    print((r or {}).get("intraday", {}).get("summary"))


# --------------------------------------------------------------------------- the announcement premium (holding INTO the news)
# Savor & Wilson (2016), Frazzini & Lamont (2007): stocks earn more around scheduled announcements. Strict version uses
# predicted dates (last year's same-quarter reaction day + 52 weeks), as Savor & Wilson do, because firms that delay or
# bring forward announcements may do so for reasons tied to the news. Entry at the close k trading days before the
# (predicted or actual) reaction day, only if that is still before the news; exits before the news, after the reaction
# day, or one day later. Control: the same trade 30 trading days later (mid-quarter, no announcement in the window).

PREM_ENTRIES = (1, 2, 5, 10)
PREM_EXITS = (("just before the news", -1), ("after the reaction day", 0), ("one day after", 1))


def premium_rows(ann: pd.DataFrame, px: dict, membership: pd.DataFrame | None = None, cost: float = COST["daily"]) -> pd.DataFrame:
    C = px["close"]
    if C.empty or "SPY" not in C:
        return pd.DataFrame()
    days = C.index
    spy = C["SPY"].values
    ev = classify(ann, days)
    ev = ev[ev.ticker.isin(C.columns)].sort_values(["ticker", "reaction"]).reset_index(drop=True)
    pred = np.full(len(ev), -1)
    year = np.timedelta64(364, "D")
    for tk, g in ev.groupby("ticker"):
        r = g.reaction.values.astype("datetime64[D]")
        for i, rj in enumerate(r):
            # last year's announcement nearest to 52 weeks earlier (extra 8-K results filings must not shift the count)
            lo, hi = rj - np.timedelta64(400, "D"), rj - np.timedelta64(330, "D")
            cand = r[(r >= lo) & (r <= hi)]
            if len(cand):
                best = cand[np.argmin(np.abs((cand - (rj - year)).astype(int)))]
                pred[g.index[i]] = days.searchsorted(np.datetime64(best + year, "ns"))
    ev["pred_pos"] = pred
    if membership is not None:
        ev = ev[member_at(ev, membership).values]
    rows = []
    for tk, g in ev.groupby("ticker"):
        c = C[tk].values
        ann_pos = set(g.rpos.tolist())
        busy = set()
        for p in ann_pos:
            busy.update(range(p - 1, p + 2))
        for rp, pp, react in zip(g.rpos.values, g.pred_pos.values, g.reaction.values):
            for version, anchor in (("expected date", pp), ("actual date", rp)):
                if anchor < 0:
                    continue
                for k in PREM_ENTRIES:
                    e = anchor - k
                    if e < 61 or e > rp - 1 or np.isnan(c[e]):
                        continue
                    for lab, off in PREM_EXITS:
                        x = rp + off
                        if x <= e or x >= len(days) or np.isnan(c[x]):
                            continue
                        ret = c[x] / c[e] - spy[x] / spy[e] - cost
                        ce, cx = e + 30, x + 30
                        ctrl = np.nan
                        if cx < len(days) and not np.isnan(c[ce]) and not np.isnan(c[cx]) and not (busy & set(range(ce, cx + 1))):
                            ctrl = c[cx] / c[ce] - spy[cx] / spy[ce] - cost
                        rows.append((version, k, lab, pd.Timestamp(react), ret, ctrl, x - e, anchor - rp))
    return pd.DataFrame(rows, columns=["version", "k", "exit", "reaction", "ret", "ctrl", "days_held", "date_error"])


def premium_cells(R: pd.DataFrame, split=SPLIT_DAILY) -> list:
    if R.empty:
        return []
    out = []
    for (version, k, lab), g in R.groupby(["version", "k", "exit"], sort=False):
        early = (g.reaction >= split[0]) & (g.reaction <= split[1])
        late = g.reaction >= split[2]
        e, l = cluster_stats(g.reaction[early], g.ret[early]), cluster_stats(g.reaction[late], g.ret[late])
        cl = cluster_stats(g.reaction[late], g.ctrl[late])
        out.append({"version": version, "entry_days_before": int(k), "exit": lab, "avg_days_held": float(g.days_held.mean()),
                    "early": e, "late": l, "control_late": cl, "vs_control": vs_control(l, cl), "verdict": verdict(e, l, cl)})
    order = {"expected date": 0, "actual date": 1}
    return sorted(out, key=lambda c: (order[c["version"]], c["entry_days_before"], [x[0] for x in PREM_EXITS].index(c["exit"])))


def premium_by_year(R: pd.DataFrame, version="expected date", k=5, lab="after the reaction day") -> list:
    g = R[(R.version == version) & (R.k == k) & (R.exit == lab)]
    out = []
    for y, h in g.groupby(g.reaction.dt.year):
        out.append({"year": int(y), "mean": float(h.ret.mean()), "control": float(h.ctrl.mean()) if h.ctrl.notna().any() else None,
                    "events": int(len(h))})
    return out


def premium_study(store: Path, tickers: list, membership: pd.DataFrame | None = None) -> dict:
    t0 = time.time()
    stocks = [t for t in tickers if not str(t).startswith("^") and t not in ETFS]
    ann = announcements(store, stocks, budget_s=0)                           # cached from the earnings study
    px = daily_prices(store, sorted(set(stocks) | {"SPY"}))
    R = premium_rows(ann, px, membership)
    cells = premium_cells(R)
    solid = [c for c in cells if c["verdict"] == "Solid"]
    strict = [c for c in cells if c["version"] == "expected date"]
    best = max((c for c in strict if c.get("late")), key=lambda c: c["late"]["mean"], default=None)
    if solid:
        b = max(solid, key=lambda c: c["late"]["mean"])
        summary = (f"{len(solid)} of {len(cells)} variants passed every test. Strongest: enter {b['entry_days_before']} days before the "
                   f"{b['version']}, exit {b['exit']}: {b['late']['mean'] * 100:+.2f}% per trade after costs (t = {b['late']['t']:.1f}).")
    else:
        summary = f"None of {len(cells)} variants passed every test after costs." + (
            f" Best strict variant recently: {best['late']['mean'] * 100:+.2f}% per trade (t = {best['late']['t']:.1f})." if best else "")
    strict_rows = R[(R.version == "expected date") & (R.k == 1) & (R.exit == "after the reaction day")] if len(R) else R
    accuracy = {"within_1_day": float((strict_rows.date_error.abs() <= 1).mean()) if len(strict_rows) else None,
                "within_5_days": float((strict_rows.date_error.abs() <= 5).mean()) if len(strict_rows) else None,
                "median_error_days": float(strict_rows.date_error.median()) if len(strict_rows) else None}
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "seconds": round(time.time() - t0),
            "announcements": int(len(ann)), "trades": int(len(R)), "cells": cells, "summary": summary,
            "date_accuracy": accuracy,
            "by_year": {"expected date": premium_by_year(R, "expected date"), "actual date": premium_by_year(R, "actual date")},
            "method": ("Long the stock from the close k trading days before the reaction day to the exit, minus the S&P over the same days, "
                       "after 0.10% round-trip cost. Expected date = last year's same-quarter reaction day + 52 weeks (no hindsight); "
                       "actual date assumes the date was published before entry (usual for S&P 500 companies, not checked). Control = "
                       "the same trade 30 trading days later with no announcement in the window. Clustered by announcement date; "
                       "Solid = positive in both periods, t >= 3 recently, and beating the control with t >= 2. Members at the time only.")}
