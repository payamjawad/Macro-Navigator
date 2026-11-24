"""Data access: FRED, Yahoo Finance (with a Stooq fallback), Shiller CAPE, local archive, demo data.

Every fetcher fails soft: a missing series is reported in `status` instead of crashing the run.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
ARCHIVE = DATA / "archive"

# FRED series -> publication lag in calendar days, measured from FRED's observation date
# (monthly data are dated the 1st of the month they describe).
FRED_SERIES: dict[str, int] = {
    "PCEPILFE": 60,              # core PCE price index
    "CPILFESL": 45,              # core CPI index
    "DFEDTARU": 1,               # fed funds target, upper bound
    "DFF": 1,                    # effective fed funds
    "DGS2": 1, "DGS10": 1, "DFII10": 1, "T10Y3M": 1, "T10Y2Y": 1, "T5YIFR": 1,
    "SAHMREALTIME": 36,          # Sahm rule, real time
    "UNRATE": 36,
    "ICSA": 5,                   # initial claims (weekly)
    "GACDFSA066MSFRBPHI": 20,    # Philadelphia Fed activity (ISM substitute)
    "PERMIT": 48,                # building permits (LEI substitute)
    "BAMLH0A0HYM2": 1,           # ICE BofA HY OAS: FRED keeps only 3 years since Apr 2026
    "BAA10Y": 1,                 # Moody's Baa minus 10Y: long-history credit proxy
    "NFCI": 5,                   # Chicago Fed financial conditions (weekly)
    "CPFF": 1,                   # 3M commercial paper minus fed funds
    "SOFR": 1, "IORB": 1,
    "DCOILBRENTEU": 1,           # Brent
    "VIXCLS": 1,                 # VIX backup
    "DTWEXBGS": 1,               # broad dollar index
}

# Series FRED truncates, so we keep our own growing copy in data/archive/.
ARCHIVED = ["BAMLH0A0HYM2"]

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def load_yaml(name: str) -> dict:
    with open(CONFIG / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def universe_tickers(universe: dict) -> list[str]:
    tickers: list[str] = [universe.get("benchmark", "SPY"), universe.get("global_benchmark", "ACWI")]
    for group in ("us_indices", "us_sectors", "us_industries", "global_markets", "global_indices", "cross_asset"):
        tickers += [row["ticker"] for row in universe.get(group, [])]
    tickers += universe.get("market_internals", [])
    return list(dict.fromkeys(tickers))  # de-duplicate, keep order


@dataclass
class RawData:
    fred: dict[str, pd.Series] = field(default_factory=dict)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    cape: float | None = None
    status: dict[str, str] = field(default_factory=dict)   # name -> "ok" or a problem description
    demo: bool = False


# ---------------------------------------------------------------- FRED

def fetch_fred(series_id: str, api_key: str, start: str, session: requests.Session | None = None) -> pd.Series:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start}
    sess = session or requests.Session()
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = sess.get(url, params=params, timeout=30)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            values = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce")  # "." -> NaN
            idx = pd.to_datetime([o["date"] for o in obs])
            return pd.Series(values.to_numpy(), index=idx, name=series_id).dropna()
        except Exception as err:  # network hiccups, rate limits
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"FRED {series_id}: {last_err}")


def merge_archive(series_id: str, fresh: pd.Series | None) -> pd.Series | None:
    """Union the fresh download with our archived copy; fresh values win on overlapping dates."""
    path = ARCHIVE / f"{series_id}.csv"
    old = None
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    if fresh is None or fresh.empty:
        return old
    merged = fresh if old is None else pd.concat([old[~old.index.isin(fresh.index)], fresh]).sort_index()
    return merged.rename(series_id)


def save_archives(raw: RawData) -> None:
    if raw.demo:
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for sid in ARCHIVED:
        s = raw.fred.get(sid)
        if s is not None and not s.empty:
            s.rename(sid).to_frame().to_csv(ARCHIVE / f"{sid}.csv")


# ---------------------------------------------------------------- prices

def _close_from_yf(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in set(lvl0):
            close = df["Close"]
        else:
            close = df.xs("Close", axis=1, level=1)
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def fetch_stooq(ticker: str, start: str) -> pd.Series | None:
    """Best-effort fallback for US ETFs only; Stooq may change or restrict this endpoint."""
    if ticker.startswith("^") or "=" in ticker:
        return None
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or not resp.text.startswith("Date"):
            return None
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")
        return df.loc[df.index >= pd.Timestamp(start), "Close"].rename(ticker)
    except Exception:
        return None


def fetch_prices(tickers: list[str], start: str, status: dict[str, str]) -> pd.DataFrame:
    import yfinance as yf  # imported lazily so tests and demo mode do not need network

    frames: list[pd.DataFrame] = []
    try:
        df = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True, group_by="column")
        frames.append(_close_from_yf(df, tickers))
    except Exception as err:
        status["yahoo_bulk"] = f"bulk download failed: {err}"
    prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    prices = prices.loc[:, ~prices.columns.duplicated()] if not prices.empty else prices

    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    for t in missing:  # retry one by one, then Stooq
        series = None
        try:
            single = yf.download(t, start=start, auto_adjust=True, progress=False)
            c = _close_from_yf(single, [t])
            if not c.empty:
                series = c.iloc[:, 0].rename(t)
        except Exception:
            series = None
        if series is None or series.dropna().empty:
            series = fetch_stooq(t, start)
            if series is not None:
                status[t] = "ok (Stooq fallback)"
        if series is not None and not series.dropna().empty:
            prices = prices.join(series, how="outer") if not prices.empty else series.to_frame()
        else:
            status[t] = "no price data"
    for t in tickers:
        status.setdefault(t, "ok")
    return prices.sort_index()


# ---------------------------------------------------------------- CAPE

def fetch_cape(status: dict[str, str], manual: float | None) -> float | None:
    try:
        resp = requests.get(SHILLER_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        sheet = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data", header=None)
        header_row = next(i for i in range(15) if "CAPE" in [str(v).strip() for v in sheet.iloc[i].tolist()])
        cols = [str(v).strip() for v in sheet.iloc[header_row].tolist()]
        cape = pd.to_numeric(sheet.iloc[header_row + 1:, cols.index("CAPE")], errors="coerce").dropna()
        status["CAPE"] = "ok (Shiller data)"
        return float(cape.iloc[-1])
    except Exception as err:
        if manual is not None:
            status["CAPE"] = f"manual override ({err.__class__.__name__} on download)"
            return float(manual)
        status["CAPE"] = f"unavailable: {err.__class__.__name__}; set manual_overrides.cape"
        return None


# ---------------------------------------------------------------- orchestration

def load_raw(fred_key: str | None, years: int = 6, demo: bool = False) -> RawData:
    if demo:
        return make_demo_data(years=years)
    if not fred_key:
        raise ValueError("FRED_API_KEY is missing. Add it as a secret (see README) or use demo mode.")
    thresholds = load_yaml("thresholds.yaml")
    universe = load_yaml("universe.yaml")
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    raw = RawData()
    sess = requests.Session()
    for sid in FRED_SERIES:
        try:
            s = fetch_fred(sid, fred_key, start if sid not in ARCHIVED else "1990-01-01", sess)
            raw.status[sid] = "ok"
        except Exception as err:
            s = None
            raw.status[sid] = f"failed: {err}"
        if sid in ARCHIVED:
            s = merge_archive(sid, s)
            if s is not None and raw.status[sid] != "ok":
                raw.status[sid] = "archive only (FRED download failed)"
        if s is not None and not s.empty:
            raw.fred[sid] = s
    raw.prices = fetch_prices(universe_tickers(universe), start, raw.status)
    raw.cape = fetch_cape(raw.status, (thresholds.get("manual_overrides") or {}).get("cape"))
    return raw


# ---------------------------------------------------------------- demo data

def make_demo_data(years: int = 6, seed: int = 7) -> RawData:
    """Synthetic but internally consistent data so the app and tests run without keys or network."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    days = pd.bdate_range(end - pd.DateOffset(years=years), end)
    months = pd.date_range(days[0].replace(day=1), end, freq="MS")
    weeks = pd.date_range(days[0], end, freq="W-SAT")
    n, t = len(days), np.linspace(0, 1, len(days))

    def walk(start, vol, drift=0.0):
        return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))

    fred: dict[str, pd.Series] = {}
    # a hiking cycle in the middle third, then stabilisation
    ff = np.clip(np.round((0.25 + 5.0 * np.clip((t - 0.25) / 0.3, 0, 1) - 1.25 * np.clip((t - 0.75) / 0.2, 0, 1)) * 4) / 4, 0.25, None)
    fred["DFEDTARU"] = pd.Series(ff, days)
    fred["DFF"] = pd.Series(ff - 0.17, days)
    fred["DGS2"] = pd.Series(ff + 0.3 * np.sin(6 * t) + rng.normal(0, 0.03, n), days)
    fred["DGS10"] = pd.Series(1.5 + 2.8 * np.clip(t * 1.4, 0, 1) + rng.normal(0, 0.04, n), days)
    fred["T10Y2Y"] = fred["DGS10"] - fred["DGS2"]
    fred["T10Y3M"] = fred["DGS10"] - (pd.Series(ff, days) + 0.05)
    fred["DFII10"] = fred["DGS10"] - 2.3
    fred["T5YIFR"] = pd.Series(2.2 + 0.2 * np.sin(4 * t), days)
    hy = 330 + 60 * np.sin(9 * t) + 450 * np.exp(-((t - 0.18) / 0.02) ** 2) + rng.normal(0, 6, n)
    fred["BAMLH0A0HYM2"] = pd.Series(hy / 100, days)
    fred["BAA10Y"] = pd.Series((hy / 100) * 0.45 + 0.4, days)
    fred["CPFF"] = pd.Series(0.10 + rng.normal(0, 0.02, n), days)
    fred["SOFR"] = pd.Series(ff - 0.2, days)
    fred["IORB"] = pd.Series(ff - 0.1, days)
    fred["DCOILBRENTEU"] = pd.Series(walk(65, 0.02), days)
    fred["VIXCLS"] = pd.Series(16 + 25 * np.exp(-((t - 0.18) / 0.015) ** 2) + np.abs(rng.normal(0, 2, n)), days)
    fred["DTWEXBGS"] = pd.Series(walk(115, 0.003), days)
    m = len(months)
    tm = np.linspace(0, 1, m)
    fred["PCEPILFE"] = pd.Series(100 * np.cumprod(1 + (0.0016 + 0.0025 * np.exp(-((tm - 0.45) / 0.15) ** 2))), months)
    fred["CPILFESL"] = pd.Series(100 * np.cumprod(1 + (0.0018 + 0.0030 * np.exp(-((tm - 0.45) / 0.15) ** 2))), months)
    fred["UNRATE"] = pd.Series(3.8 + 0.4 * tm + rng.normal(0, 0.05, m), months)
    fred["SAHMREALTIME"] = pd.Series(np.clip(0.05 + 0.25 * tm + rng.normal(0, 0.03, m), 0, None), months)
    fred["GACDFSA066MSFRBPHI"] = pd.Series(5 + 10 * np.sin(5 * tm) + rng.normal(0, 4, m), months)
    fred["PERMIT"] = pd.Series(1400 * np.exp(np.cumsum(rng.normal(0, 0.02, m))), months)
    fred["ICSA"] = pd.Series(220_000 + 15_000 * np.sin(np.linspace(0, 8, len(weeks))) + rng.normal(0, 5_000, len(weeks)), weeks)
    fred["NFCI"] = pd.Series(-0.45 + 0.5 * np.exp(-((np.linspace(0, 1, len(weeks)) - 0.18) / 0.02) ** 2), weeks)

    universe = load_yaml("universe.yaml")
    tickers = universe_tickers(universe)
    market = walk(3000, 0.010, 0.0004)
    shock = 1 - 0.30 * np.exp(-((t - 0.18) / 0.012) ** 2)
    prices = {}
    for i, tk in enumerate(tickers):
        beta = 0.6 + (i % 7) * 0.12
        idio = np.exp(np.cumsum(rng.normal(0, 0.006, n)))
        prices[tk] = (market / market[0]) ** beta * idio * shock ** beta * (50 + 10 * i)
    prices["^GSPC"] = market * shock
    prices["^VIX"] = fred["VIXCLS"].to_numpy()
    prices["^VIX3M"] = fred["VIXCLS"].to_numpy() * 0.92 + 2.0
    prices["^MOVE"] = 95 + 40 * np.exp(-((t - 0.18) / 0.02) ** 2) + rng.normal(0, 3, n)
    status = {k: "demo" for k in list(fred) + tickers}
    return RawData(fred=fred, prices=pd.DataFrame(prices, index=days), cape=36.0, status=status, demo=True)


def fred_key_from_env() -> str | None:
    return os.environ.get("FRED_API_KEY") or None
