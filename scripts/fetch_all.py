#!/usr/bin/env python3
"""
fetch_all.py - fast, parallel, incremental download of every dataset the Macro Navigator uses.

    export FRED_API_KEY=your_key        # free: fred.stlouisfed.org -> My Account -> API Keys
    python fetch_all.py                 # first run: full history; later runs: only what is new
    python fetch_all.py --full          # rebuild from scratch (archived ICE spreads are kept)
    python fetch_all.py --only fred     # any of: fred, yahoo, files (comma-separated)
    python fetch_all.py --vintages      # + ALFRED first-release values for revised macro series
    python fetch_all.py --breadth       # + current S&P 500 members (for breadth studies)

Everything runs at once: 65 FRED series, 97 Yahoo tickers and 8 research files are
downloaded in parallel. Output goes to ./data/store (change with --out):

    fred/*.parquet            one file per FRED series (history grows; ICE spreads never dropped)
    yahoo_prices.parquet      adjusted daily closes, one column per ticker
    files/*.parquet           Shiller, Ken French, Fed EBP, NY Fed GSCPI, BoE, ECB, Goyal (manual)
    panel_daily.parquet       every daily/weekly series on one calendar (raw, not filled)
    panel_monthly.parquet     every series at month-end, ready for research
    catalog.csv               what was fetched: source, first/last date, rows, status, seconds

Needs: pandas, requests, yfinance, pyarrow (parquet), xlrd (Shiller .xls), openpyxl (.xlsx).
"""
from __future__ import annotations

import argparse
import faulthandler
import importlib
import io
import os
import subprocess
import re
import sys
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

faulthandler.enable()   # if a compiled library crashes, print where it happened

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 macro-navigator-fetch/1.0"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# --------------------------------------------------------------------------- catalog
# FRED id -> (group, frequency, description). Frequency drives the incremental window.
FRED = {
    # inflation, money, wages
    "CPIAUCSL": ("inflation", "M", "CPI, all items (1947-)"),
    "CPILFESL": ("inflation", "M", "Core CPI (1957-)"),
    "PCEPI": ("inflation", "M", "PCE price index (1959-)"),
    "PCEPILFE": ("inflation", "M", "Core PCE price index (1959-)"),
    "T5YIFR": ("inflation", "D", "5y5y forward inflation expectation (2003-)"),
    "T10YIE": ("inflation", "D", "10-year breakeven inflation (2003-)"),
    "MICH": ("inflation", "M", "Michigan 1-year inflation expectations (1978-)"),
    "AHETPI": ("inflation", "M", "Average hourly earnings, production workers (1964-)"),
    "M2SL": ("inflation", "M", "M2 money stock (1959-)"),
    # policy and rates
    "DFEDTARU": ("rates", "D", "Fed funds target, upper bound (2008-)"),
    "DFEDTAR": ("rates", "D", "Fed funds target, single value (1982-2008)"),
    "DFF": ("rates", "D", "Effective fed funds, daily (1954-)"),
    "FEDFUNDS": ("rates", "M", "Effective fed funds, monthly (1954-)"),
    "TB3MS": ("rates", "M", "3-month T-bill, monthly (1934-)"),
    "DTB3": ("rates", "D", "3-month T-bill, daily (1954-)"),
    "GS1": ("rates", "M", "1-year Treasury, monthly (1953-)"),
    "GS10": ("rates", "M", "10-year Treasury, monthly (1953-)"),
    "DGS2": ("rates", "D", "2-year Treasury (1976-)"),
    "DGS10": ("rates", "D", "10-year Treasury (1962-)"),
    "DGS3MO": ("rates", "D", "3-month Treasury, constant maturity (1981-)"),
    "DFII10": ("rates", "D", "10-year TIPS real yield (2003-)"),
    "T10Y3M": ("rates", "D", "10Y minus 3M spread (1982-)"),
    "T10Y2Y": ("rates", "D", "10Y minus 2Y spread (1976-)"),
    "THREEFYTP10": ("rates", "D", "10-year term premium, Kim-Wright (1990-)"),
    "SOFR": ("rates", "D", "Secured overnight financing rate (2018-)"),
    "IORB": ("rates", "D", "Interest on reserve balances (2021-)"),
    # Fed balance sheet and liquidity
    "WALCL": ("liquidity", "W", "Fed total assets (2002-)"),
    "WTREGEN": ("liquidity", "W", "Treasury General Account (2002-)"),
    "RRPONTSYD": ("liquidity", "D", "Overnight reverse repo (2003-)"),
    "WRESBAL": ("liquidity", "W", "Reserve balances (2002-)"),
    # growth and labour
    "UNRATE": ("growth", "M", "Unemployment rate (1948-)"),
    "PAYEMS": ("growth", "M", "Nonfarm payrolls (1939-)"),
    "SAHMREALTIME": ("growth", "M", "Sahm rule, real-time (1959-)"),
    "ICSA": ("growth", "W", "Initial jobless claims (1967-)"),
    "CCSA": ("growth", "W", "Continuing claims (1967-)"),
    "INDPRO": ("growth", "M", "Industrial production (1919-)"),
    "GACDFSA066MSFRBPHI": ("growth", "M", "Philadelphia Fed activity index (1968-)"),
    "PERMIT": ("growth", "M", "Building permits (1960-)"),
    "HOUST": ("growth", "M", "Housing starts (1959-)"),
    "HTRUCKSSAAR": ("growth", "M", "Heavy truck sales (1967-)"),
    "AWHMAN": ("growth", "M", "Average weekly hours, manufacturing (1939-)"),
    "UMCSENT": ("growth", "M", "University of Michigan consumer sentiment (1952-)"),
    "GDPC1": ("growth", "Q", "Real GDP (1947-)"),
    "USREC": ("growth", "M", "NBER recession indicator (1854-)"),
    "RECPROUSM156N": ("growth", "M", "Smoothed recession probability, Chauvet-Piger (1967-)"),
    # credit and financial conditions
    "BAMLH0A0HYM2": ("credit", "D", "ICE BofA high-yield OAS (FRED keeps 3 years; archived here)"),
    "BAMLH0A3HYC": ("credit", "D", "ICE BofA CCC and lower OAS (archived)"),
    "BAMLC0A0CM": ("credit", "D", "ICE BofA investment-grade OAS (archived)"),
    "BAA": ("credit", "M", "Moody's Baa corporate yield (1919-)"),
    "AAA": ("credit", "M", "Moody's Aaa corporate yield (1919-)"),
    "BAA10Y": ("credit", "D", "Baa minus 10-year Treasury (1986-)"),
    "NFCI": ("credit", "W", "Chicago Fed financial conditions index (1971-)"),
    "ANFCI": ("credit", "W", "Adjusted NFCI (1971-)"),
    "STLFSI4": ("credit", "W", "St. Louis Fed financial stress index (1993-)"),
    "CPFF": ("credit", "D", "3-month commercial paper minus fed funds (1997-)"),
    "DRTSCILM": ("credit", "Q", "Banks tightening C&I standards, SLOOS (1990-)"),
    "TEDRATE": ("credit", "D", "TED spread (1986-2022, discontinued)"),
    # markets, commodities, FX
    "DCOILWTICO": ("markets", "D", "WTI crude, daily (1986-)"),
    "DCOILBRENTEU": ("markets", "D", "Brent crude, daily (1987-)"),
    "WTISPLC": ("markets", "M", "WTI crude, monthly (1946-)"),
    "PCOPPUSDM": ("markets", "M", "Copper price, monthly (1990-)"),
    "VIXCLS": ("markets", "D", "VIX (1990-)"),
    "VXOCLS": ("markets", "D", "VXO, old VIX (1986-2021)"),
    "DTWEXBGS": ("markets", "D", "Broad trade-weighted dollar (2006-)"),
    "DEXUSUK": ("markets", "D", "US dollars per pound (1971-)"),
    # other countries
    "IRLTLT01GBM156N": ("global", "M", "UK 10-year gilt yield (1960-)"),
    "IRLTLT01DEM156N": ("global", "M", "Germany 10-year yield (1956-)"),
    "IRLTLT01JPM156N": ("global", "M", "Japan 10-year yield (1989-)"),
}
ARCHIVED = {"BAMLH0A0HYM2", "BAMLH0A3HYC", "BAMLC0A0CM"}       # never drop old rows
VINTAGE_SERIES = ["PAYEMS", "UNRATE", "CPILFESL", "PCEPILFE", "INDPRO", "GDPC1", "PERMIT", "HTRUCKSSAAR", "AWHMAN"]

YAHOO = [
    # long-history indices and market internals
    "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT", "^VIX", "^VIX3M", "^VVIX", "^MOVE", "^SKEW", "^TNX", "^IRX",
    # US indices, sectors, industries (ETFs)
    "SPY", "QQQ", "DIA", "RSP", "MDY", "IWM",
    "XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE",
    "SMH", "IGV", "KRE", "XHB", "XBI", "ITA", "KBE", "XOP", "IYT",
    # factors and styles
    "IWD", "IWF", "MTUM", "QUAL", "USMV", "VLUE",
    # global markets
    "ACWI", "EFA", "EEM", "VGK", "EWU", "EWG", "EWQ", "EWJ", "MCHI", "FXI", "INDA", "EWT", "EWY", "EWZ",
    "EWC", "EWA", "EWW", "EWH", "EWS",
    "^FTSE", "^FTMC", "^GDAXI", "^STOXX50E", "^FCHI", "^N225", "^HSI", "^GSPTSE", "^AXJO", "^BSESN",
    # bonds, credit, real assets
    "TLT", "IEF", "SHY", "TIP", "HYG", "JNK", "LQD", "EMB", "BIL", "GLD", "SLV", "DBC", "USO", "VNQ",
    # futures and FX
    "CL=F", "BZ=F", "GC=F", "HG=F", "NG=F", "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "JPY=X", "BTC-USD",
]

BOE_URL = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
           "&Datefrom=01/Jan/1975&Dateto=now&SeriesCodes=IUDBEDR&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
FILES = {  # name -> (urls to try in order, parser name, description)
    "shiller": (["shillerdata.com", "http://www.econ.yale.edu/~shiller/data/ie_data.xls"], "shiller",
                "Shiller: S&P price, dividends, earnings, CPI, 10Y, CAPE (monthly, 1871-)"),
    "french_12_industries": (["https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_CSV.zip"],
                             "french", "Ken French 12 industry portfolios, value-weighted monthly returns (1926-)"),
    "french_factors": (["https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"],
                       "french", "Fama-French market, size, value factors and T-bill (1926-)"),
    "french_momentum": (["https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"],
                        "french", "Momentum factor (1927-)"),
    "fed_ebp": (["https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"], "csv_dated",
                "Fed GZ spread, excess bond premium, recession probability (monthly, 1973-)"),
    "nyfed_gscpi": (["https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"],
                    "gscpi", "NY Fed Global Supply Chain Pressure Index (monthly, 1997-)"),
    "boe_bank_rate": ([BOE_URL], "boe", "Bank of England Bank Rate (1975-)"),
    "ecb_deposit_rate": (["https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV?format=csvdata"],
                         "ecb", "ECB deposit facility rate (1999-)"),
}
FILE_TTL_HOURS = 20  # these files change at most monthly


# --------------------------------------------------------------------------- storage helpers
try:
    import pyarrow  # noqa: F401
    EXT = ".parquet"
except ImportError:  # still works, just slower and bigger
    EXT = ".csv.gz"


SAVE_LOCK = threading.Lock()


def save(df: pd.DataFrame | pd.Series, path: Path) -> None:
    with SAVE_LOCK:              # one write at a time: some parquet builds dislike parallel writers
        _save(df, path)


def _save(df: pd.DataFrame | pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.to_frame() if isinstance(df, pd.Series) else df
    if EXT == ".parquet":
        df.to_parquet(path.with_suffix(".parquet"))
    else:
        df.to_csv(Path(str(path.with_suffix("")) + ".csv.gz"))


def _paths(path: Path) -> list[Path]:
    """Files this environment can read, preferred first (no parquet without pyarrow)."""
    p, c = path.with_suffix(".parquet"), Path(str(path.with_suffix("")) + ".csv.gz")
    return [p, c] if EXT == ".parquet" else [c]


def load(path: Path) -> pd.DataFrame | None:
    for f in _paths(path):
        if f.exists():
            if f.suffix == ".parquet":
                return pd.read_parquet(f)
            return pd.read_csv(f, index_col=0, parse_dates=True)
    return None


def age_hours(path: Path) -> float:
    for f in _paths(path):
        if f.exists():
            return (time.time() - f.stat().st_mtime) / 3600
    return float("inf")


def make_session(pool: int = 32) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": UA})
    return s


class RateLimiter:
    """FRED allows 120 requests a minute per key; stay just under it without slowing small runs."""

    def __init__(self, per_minute: int = 110):
        self.per_minute, self.calls, self.lock = per_minute, deque(), threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()
                if len(self.calls) < self.per_minute:
                    self.calls.append(now)
                    return
                sleep_for = 60 - (now - self.calls[0]) + 0.05
            time.sleep(sleep_for)


# --------------------------------------------------------------------------- FRED

def _fred_api(sess, limiter, sid, key, start=None, extra=None):
    params = {"series_id": sid, "api_key": key, "file_type": "json"}
    if start:
        params["observation_start"] = start
    params.update(extra or {})
    limiter.wait()
    r = sess.get(FRED_API, params=params, timeout=40)
    if r.status_code == 400:
        msg = r.json().get("error_message", "bad request") if "json" in r.headers.get("content-type", "") else r.text[:120]
        raise RuntimeError(f"FRED says: {msg}")
    r.raise_for_status()
    return r.json().get("observations", [])


def _fred_csv(sess, sid, start=None):
    params = {"id": sid}
    if start:
        params["cosd"] = start
    r = sess.get(FRED_CSV, params=params, timeout=40)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    date_col = df.columns[0]
    return pd.Series(pd.to_numeric(df[sid], errors="coerce").to_numpy(), index=pd.to_datetime(df[date_col]), name=sid)


def fetch_fred_series(sess, limiter, sid, key, out_dir, full):
    t0 = time.time()
    freq = FRED[sid][1]
    path = out_dir / "fred" / sid
    old = load(path)
    old_s = old.iloc[:, 0] if old is not None and len(old.columns) else None
    start = None
    if old_s is not None and not old_s.empty and (not full or sid in ARCHIVED):
        if not full:
            window = 30 if freq in ("D", "W") else 400   # re-pull recent history to pick up revisions
            start = (old_s.index.max() - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    via = ""
    if key:
        try:
            obs = _fred_api(sess, limiter, sid, key, start)
            new = pd.Series(pd.to_numeric([o["value"] for o in obs], errors="coerce"),
                            index=pd.to_datetime([o["date"] for o in obs]), name=sid)
        except Exception as err:            # rejected key, blocked or throttled: try the public download instead
            new, via = _fred_csv(sess, sid, start), f" via public CSV (API: {str(err)[:60]})"
    else:
        new = _fred_csv(sess, sid, start)
    new = new.dropna()
    if old_s is not None and not old_s.empty and (start is not None or sid in ARCHIVED):
        keep = old_s[old_s.index < new.index.min()] if not new.empty else old_s
        merged = pd.concat([keep, new]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = new
    merged.index.name = "date"
    save(merged.rename(sid), path)
    added = len(merged) - (0 if old_s is None else len(old_s))
    return sid, merged, (f"ok (+{max(added, 0)} rows)" if old_s is not None else "ok") + via, time.time() - t0


def vintages_due(out_dir: Path, days: float = 6.5) -> bool:
    """First-release data change rarely: refresh them about once a week (and whenever one is missing)."""
    folder = out_dir / "fred_first_release"
    for sid in VINTAGE_SERIES:
        hits = list(folder.glob(f"{sid}.*")) if folder.exists() else []
        if not hits or (time.time() - max(h.stat().st_mtime for h in hits)) / 86400 > days:
            return True
    return False


def fetch_vintage(sess, limiter, sid, key, out_dir):
    """First-release value of each observation and the date it was published (ALFRED)."""
    t0 = time.time()
    obs = _fred_api(sess, limiter, sid, key, extra={"realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                                                     "output_type": 4})
    df = pd.DataFrame({"value": pd.to_numeric([o["value"] for o in obs], errors="coerce"),
                       "first_release": pd.to_datetime([o.get("realtime_start") for o in obs])},
                      index=pd.to_datetime([o["date"] for o in obs]))
    df.index.name = "date"
    save(df, out_dir / "fred_first_release" / sid)
    return f"vintage:{sid}", df, "ok", time.time() - t0


# --------------------------------------------------------------------------- Yahoo

def _closes(df, tickers):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        close = df["Close"] if "Close" in lvl0 else df.xs("Close", axis=1, level=1)
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.dropna(how="all")


def fetch_yahoo(tickers, out_dir, full, name="yahoo_prices", threads=True):
    import yfinance as yf
    t0 = time.time()
    path = out_dir / name
    old = None if full else load(path)
    kw = dict(auto_adjust=True, progress=False, threads=threads, group_by="column")
    status = {}
    adjusted = []
    if old is None or old.empty:
        new = _closes(yf.download(tickers, period="max", **kw), tickers)
        merged, refetch = new, []
    else:
        start = (old.index.max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        new = _closes(yf.download(tickers, start=start, **kw), tickers)
        refetch = [t for t in tickers if t not in old.columns or old[t].dropna().empty]
        adjusted = []
        for t in tickers:
            if t in old.columns and t in new.columns and t not in refetch:
                both = pd.concat([old[t], new[t]], axis=1, keys=["o", "n"]).dropna()
                # adjusted history shifts after a dividend or split: re-pull that ticker in full
                if len(both) and (both.o / both.n - 1).abs().max() > 0.002:
                    adjusted.append(t)
        refetch += adjusted
        merged = old.combine_first(new)
        merged.update(new)
    if refetch:
        extra = _closes(yf.download(refetch, period="max", **kw), refetch)
        for t in refetch:
            if t in extra.columns:
                merged = merged.drop(columns=[t], errors="ignore").join(extra[[t]], how="outer")
    missing = [t for t in tickers if t not in merged.columns or merged[t].dropna().empty]
    for t in missing[:60]:
        for wait in (2, 6):
            time.sleep(wait)
            try:
                one = _closes(yf.download(t, period="max", auto_adjust=True, progress=False, threads=False), [t])
            except Exception:
                one = pd.DataFrame()
            if t in one.columns and one[t].notna().any():
                merged = merged.drop(columns=[t], errors="ignore").join(one[[t]], how="outer")
                break
    merged = merged.sort_index()
    merged.index.name = "date"
    for t in tickers:
        ok = t in merged.columns and merged[t].notna().any()
        status[t] = "ok" if ok else "no data"
    save(merged, path)
    return merged, status, time.time() - t0, len(adjusted)


def sp500_members(sess) -> list[str]:
    html = sess.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    return [str(t).replace(".", "-") for t in table["Symbol"].tolist()]


# --------------------------------------------------------------------------- research files

def parse_french(content: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.strip().startswith(",") and re.match(r"^\s*\d{6}\s*,", lines[i + 1]):
            cols = [c.strip() for c in line.split(",")[1:]]
            dates, rows = [], []
            for row in lines[i + 1:]:
                m = re.match(r"^\s*(\d{6})\s*,(.*)$", row)
                if not m:
                    break
                dates.append(m.group(1))
                rows.append([float(v) for v in m.group(2).split(",")])
            idx = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
            df = pd.DataFrame(rows, index=idx, columns=cols[: len(rows[0])])
            return df.replace([-99.99, -999.0], np.nan)
    raise ValueError("monthly table not found in the French file")


def parse_shiller(content: bytes) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None)
    hdr = next(i for i in range(min(20, len(raw))) if str(raw.iloc[i, 0]).strip() == "Date")
    names = [str(v).strip() for v in raw.iloc[hdr].tolist()]
    body = raw.iloc[hdr + 1:]
    dates, keep = [], []
    for i, v in enumerate(body.iloc[:, 0].tolist()):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            break
        if not np.isfinite(fv) or fv < 1800:      # blank rows or notes after the data end
            break
        s = f"{fv:.2f}"
        dates.append(pd.Timestamp(year=int(s[:4]), month=int(s[5:7]), day=1) + pd.offsets.MonthEnd(0))
        keep.append(i)
    body = body.iloc[keep]

    def col(*labels):
        for lab in labels:
            for j, n in enumerate(names):
                if n == lab:
                    return pd.to_numeric(body.iloc[:, j], errors="coerce").to_numpy()
        return np.full(len(body), np.nan)

    df = pd.DataFrame({"P": col("P"), "D": col("D"), "E": col("E"), "CPI": col("CPI"),
                       "GS10": col("Rate GS10", "GS10"), "CAPE": col("CAPE", "P/E10"), "TR_CAPE": col("TR CAPE")},
                      index=pd.DatetimeIndex(dates))
    return df.dropna(how="all")


def parse_csv_dated(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    idx = pd.to_datetime(df.iloc[:, 0]) + pd.offsets.MonthEnd(0)
    out = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    out.index = idx
    return out


def parse_gscpi(content: bytes) -> pd.DataFrame:
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    for _, raw in sheets.items():
        for i in range(min(15, len(raw))):
            cells = [str(v) for v in raw.iloc[i].tolist()]
            hits = [j for j, c in enumerate(cells) if "gscpi" in c.lower()]
            if not hits:
                continue
            body = raw.iloc[i + 1:]
            for dj in range(raw.shape[1]):
                d = pd.to_datetime(body.iloc[:, dj], errors="coerce")
                if d.notna().mean() > 0.8:
                    v = pd.to_numeric(body.iloc[:, hits[0]], errors="coerce")
                    s = pd.Series(v.to_numpy(), index=d + pd.offsets.MonthEnd(0)).dropna()
                    s = s[s.index.notna()]
                    if len(s) > 100:
                        return s.rename("GSCPI").to_frame()
    raise ValueError("GSCPI column not found")


def parse_boe(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8", errors="ignore")
    if "<html" in text[:500].lower():
        raise ValueError("BoE returned a web page instead of CSV")
    df = pd.read_csv(io.StringIO(text))
    idx = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors="coerce")
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=idx).dropna()
    return s[s.index.notna()].rename("BANK_RATE").to_frame()


def parse_ecb(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(), index=pd.to_datetime(df["TIME_PERIOD"]))
    return s.dropna().rename("DFR").to_frame()


def parse_goyal(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = next((s for s in xl.sheet_names if s.lower().startswith("month")), xl.sheet_names[0])
    df = xl.parse(sheet)
    key = next(c for c in df.columns if str(c).lower() in ("yyyymm", "date"))
    idx = pd.to_datetime(df[key].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    out = df.drop(columns=[key]).apply(pd.to_numeric, errors="coerce")
    out.index = idx
    return out


PARSERS = {"french": parse_french, "shiller": parse_shiller, "csv_dated": parse_csv_dated,
           "gscpi": parse_gscpi, "boe": parse_boe, "ecb": parse_ecb}


def _shiller_link(sess) -> str | None:
    html = sess.get("https://shillerdata.com/", timeout=30).text
    links = re.findall(r"""https?://[^"'\s>]+ie_data\.xls[^"'\s>]*""", html)
    return links[0].replace("&amp;", "&") if links else None


def fetch_file(sess, name, out_dir, full, manual_dir):
    t0 = time.time()
    urls, parser, _ = FILES[name]
    path = out_dir / "files" / name
    if not full and age_hours(path) < FILE_TTL_HOURS:
        df = load(path)
        return name, df, "cached", time.time() - t0
    errors, df = [], None
    if name == "shiller" and (manual_dir / "ie_data.xls").exists():
        df = parse_shiller((manual_dir / "ie_data.xls").read_bytes())
    for url in ([] if df is not None else urls):
        try:
            if url == "shillerdata.com":
                url = _shiller_link(sess)
                if not url:
                    raise ValueError("no ie_data.xls link on shillerdata.com")
            r = sess.get(url, timeout=60)
            r.raise_for_status()
            df = PARSERS[parser](r.content)
            break
        except Exception as err:
            errors.append(f"{err.__class__.__name__}: {str(err)[:80]}")
    if df is None:
        old = load(path)
        if old is not None:
            return name, old, "stale cache (download failed: " + "; ".join(errors) + ")", time.time() - t0
        raise RuntimeError("; ".join(errors))
    df.index.name = "date"
    save(df, path)
    return name, df, "ok", time.time() - t0


# --------------------------------------------------------------------------- panels

def build_panels(out_dir, fred, prices, files):
    daily_cols, monthly_cols = {}, {}
    for sid, s in fred.items():
        freq = FRED[sid][1]
        if freq in ("D", "W"):
            daily_cols[f"fred:{sid}"] = s
            monthly_cols[f"fred:{sid}"] = s.resample("ME").last()
        elif freq == "M":
            monthly_cols[f"fred:{sid}"] = s.set_axis(s.index + pd.offsets.MonthEnd(0))
        else:  # quarterly: placed at the month FRED dates it (start of quarter), never in the future
            monthly_cols[f"fred:{sid}"] = s.set_axis(s.index + pd.offsets.MonthEnd(0))
    if prices is not None and not prices.empty:
        for t in prices.columns:
            daily_cols[f"px:{t}"] = prices[t].dropna()
            monthly_cols[f"px:{t}"] = prices[t].dropna().resample("ME").last()
    prefix = {"shiller": "shiller", "french_12_industries": "ind", "french_factors": "ff", "french_momentum": "ff",
              "fed_ebp": "ebp", "nyfed_gscpi": "gscpi", "boe_bank_rate": "boe", "ecb_deposit_rate": "ecb", "goyal": "goyal"}
    for name, df in files.items():
        if df is None or df.empty:
            continue
        for c in df.columns:
            s = df[c].dropna()
            if s.empty:
                continue
            key = f"{prefix.get(name, name)}:{c}"
            if name in ("boe_bank_rate", "ecb_deposit_rate"):
                daily_cols[key] = s
                monthly_cols[key] = s.resample("ME").last()
            else:
                monthly_cols[key] = s.groupby(s.index + pd.offsets.MonthEnd(0)).last()
    daily = pd.DataFrame(daily_cols).sort_index() if daily_cols else pd.DataFrame()
    monthly = pd.DataFrame(monthly_cols).sort_index() if monthly_cols else pd.DataFrame()
    if not daily.empty:
        daily = daily[daily.index.day_of_week < 5]
    cap = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)      # no rows beyond the current month
    daily = daily[daily.index <= cap] if not daily.empty else daily
    monthly = monthly[monthly.index <= cap] if not monthly.empty else monthly
    daily.index.name = monthly.index.name = "date"
    save(daily, out_dir / "panel_daily")
    save(monthly, out_dir / "panel_monthly")
    return daily.shape, monthly.shape


# --------------------------------------------------------------------------- main


def default_store() -> Path:
    """data/store inside the project when run from scripts/, otherwise next to this file."""
    here = Path(__file__).resolve().parent
    return (here.parent if here.name == "scripts" else here) / "data" / "store"

GROUPS = ("yahoo", "fred", "files")


def preload(groups) -> None:
    """Load compiled libraries once, in the main thread, before any worker thread starts."""
    mods = ["pyarrow", "pyarrow.parquet"]
    if "files" in groups:
        mods += ["openpyxl", "xlrd"]
    if "yahoo" in groups:
        mods += ["yfinance"]
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            pass


def _row(group, name, data, status, secs, desc=""):
    has = data is not None and len(data)
    return {"source": group, "name": name, "description": desc,
            "first": data.index.min() if has else None, "last": data.index.max() if has else None,
            "rows": len(data) if data is not None else 0, "status": status,
            "seconds": round(secs, 2) if secs is not None else None}


def run_groups(groups, out, args, key, safe=False) -> list[dict]:
    """Do the downloading for the given groups inside this process."""
    preload(groups)
    rows, manual = [], out.parent / "manual"
    workers = 2 if safe else args.workers
    sess, limiter = make_session(workers + 8), RateLimiter()

    if "yahoo" in groups:                      # yfinance runs in the main thread
        import logging
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # per-ticker noise goes to catalog.csv instead
        t0 = time.time()
        try:
            prices, ystat, ysecs, nref = fetch_yahoo(YAHOO, out, args.full, threads=not safe)
            for t, st in ystat.items():
                ser = prices[t].dropna() if t in prices.columns else pd.Series(dtype=float)
                rows.append(_row("yahoo", t, ser, st, None, "adjusted daily close"))
            ok = sum(1 for v in ystat.values() if v == "ok")
            extra = f"; {nref} re-pulled after dividends/splits" if nref else ""
            print(f"yahoo: {ok}/{len(ystat)} tickers ok in {ysecs:.1f}s{extra}", flush=True)
        except Exception as err:
            rows.append(_row("yahoo", "bulk download", None, f"failed: {str(err)[:140]}", time.time() - t0))
            print(f"yahoo: failed: {str(err)[:140]}", flush=True)

    jobs = []
    if "fred" in groups:
        jobs += [("fred", sid, fetch_fred_series, (sess, limiter, sid, key, out, args.full)) for sid in FRED]
        if (args.vintages or vintages_due(out)) and key:
            jobs += [("fred", f"vintage:{sid}", fetch_vintage, (sess, limiter, sid, key, out)) for sid in VINTAGE_SERIES]
    if "files" in groups:
        jobs += [("files", name, fetch_file, (sess, name, out, args.full, manual)) for name in FILES]
    if jobs:
        t0 = {g: time.time() for g in ("fred", "files")}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fn, *a): (g, n) for g, n, fn, a in jobs}
            for fut in as_completed(futures):
                g, n = futures[fut]
                try:
                    name, data, status, secs = fut.result()
                    desc = FRED[name][2] if name in FRED else FILES.get(name, (None, None, name))[2]
                    rows.append(_row(g, name, data, status, secs, desc))
                except Exception as err:
                    rows.append(_row(g, n, None, f"failed: {str(err)[:140]}", None))
        for g in ("fred", "files"):
            if g in groups:
                sub = [r for r in rows if r["source"] == g]
                ok = sum(1 for r in sub if str(r["status"]).startswith(("ok", "cached")))
                print(f"{g}: {ok}/{len(sub)} ok in {time.time() - t0[g]:.1f}s", flush=True)
    return rows


def describe_exit(rc: int) -> str:
    if rc == -11:
        return "crashed: segmentation fault in a compiled library"
    if rc == -6:
        return "crashed: abort in a compiled library"
    if rc < 0:
        return f"stopped by signal {-rc}"
    return f"stopped with error code {rc}"


def child_cmd(group, args, safe):
    cmd = [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()), "--child",
           "--only", group, "--out", args.out, "--workers", str(args.workers)]
    if args.full:
        cmd.append("--full")
    if args.vintages and group == "fred":
        cmd.append("--vintages")
    if safe:
        cmd.append("--safe")
    return cmd


def run_children(groups, args, safe=False) -> dict:
    """One process per group, in parallel; their output is streamed with a prefix."""
    procs, readers = {}, []
    for g in groups:
        p = subprocess.Popen(child_cmd(g, args, safe), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=os.environ.copy())
        procs[g] = p

        def pump(stream, tag=g):
            for line in stream:
                line = line.rstrip()
                if line:
                    print(f"  [{tag}] {line}", flush=True)
        th = threading.Thread(target=pump, args=(p.stdout,), daemon=True)
        th.start()
        readers.append(th)
    codes = {g: p.wait() for g, p in procs.items()}
    for th in readers:
        th.join(timeout=5)
    return codes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fast parallel fetch of every Macro Navigator dataset.")
    ap.add_argument("--out", default=str(default_store()))
    ap.add_argument("--full", action="store_true", help="re-download full history (ICE archive kept)")
    ap.add_argument("--only", default="fred,yahoo,files", help="comma list of: fred, yahoo, files")
    ap.add_argument("--vintages", action="store_true", help="also fetch ALFRED first-release data")
    ap.add_argument("--breadth", action="store_true", help="also fetch current S&P 500 members")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--inprocess", action="store_true", help="no child processes (debugging)")
    ap.add_argument("--safe", action="store_true", help="one connection at a time, no Yahoo threads")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    out = Path(args.out)
    args.out = str(out)
    manual = out.parent / "manual"
    out.mkdir(parents=True, exist_ok=True)
    manual.mkdir(parents=True, exist_ok=True)
    key = (os.environ.get("FRED_API_KEY") or "").strip().strip('"').strip("'") or None
    groups = [g for g in GROUPS if g in {x.strip() for x in args.only.split(",")}]

    if args.child:                              # worker process: download, write a catalog part, exit
        rows = run_groups(groups, out, args, key, safe=args.safe)
        pd.DataFrame(rows).to_csv(out / f"catalog_part_{'_'.join(groups)}.csv", index=False)
        return 0

    t_start = time.time()
    print(f"Macro Navigator fetch  {datetime.now():%Y-%m-%d %H:%M}  ->  {out}", flush=True)
    if "fred" in groups and not key:
        print("  No FRED_API_KEY set: using FRED's public CSV endpoint (slower; vintages unavailable).", flush=True)
    for g in GROUPS:
        part = out / f"catalog_part_{g}.csv"
        if part.exists():
            part.unlink()
    rows: list[dict] = []
    if args.inprocess:
        rows = run_groups(groups, out, args, key, safe=args.safe)
    else:
        print(f"  Downloading {', '.join(groups)} in parallel, one process each...", flush=True)
        codes = run_children(groups, args, safe=args.safe)
        crashed = [g for g, rc in codes.items() if rc != 0]
        for g in crashed:
            print(f"  {g} {describe_exit(codes[g])}; retrying it in safe mode (slower, one connection at a time)...", flush=True)
        if crashed:
            codes2 = run_children(crashed, args, safe=True)
            for g, rc in codes2.items():
                if rc != 0:
                    print(f"  {g} {describe_exit(rc)} again in safe mode; its saved data from earlier runs is used.", flush=True)
                    rows.append({"source": g, "name": "(whole group)", "description": "", "first": None, "last": None,
                                 "rows": 0, "status": f"failed: {describe_exit(rc)} (see the log above)", "seconds": None})
        for g in groups:
            part = out / f"catalog_part_{g}.csv"
            if part.exists():
                rows += pd.read_csv(part).to_dict(orient="records")
                part.unlink()

    preload(["files"])
    sess = make_session(8)
    if args.breadth:
        t0 = time.time()
        try:
            members = sp500_members(sess)
            fetch_yahoo(members, out, args.full, name="sp500_members_prices", threads=not args.safe)
            print(f"  Breadth: {len(members)} S&P 500 members saved ({time.time() - t0:.1f}s; current members only, so survivorship-biased).")
        except Exception as err:
            print(f"  Breadth fetch failed: {err}")

    file_data: dict = {}
    for p in sorted(manual.glob("PredictorData*.xls*")):
        try:
            file_data["goyal"] = parse_goyal(p)
            save(file_data["goyal"], out / "files" / "goyal")
            rows.append(_row("files", "goyal", file_data["goyal"], "ok (manual file)", 0, f"Goyal-Welch predictors ({p.name})"))
        except Exception as err:
            rows.append(_row("files", "goyal", None, f"failed: {err}", None, p.name))

    # panels are built from everything on disk, so a partial or crashed run still yields complete panels
    fred_data = {}
    for sid in FRED:
        d = load(out / "fred" / sid)
        if d is not None and len(d.columns):
            fred_data[sid] = d.iloc[:, 0]
    for name in list(FILES) + ["goyal"]:
        if name not in file_data:
            d = load(out / "files" / name)
            if d is not None:
                file_data[name] = d
    prices = load(out / "yahoo_prices")
    t0 = time.time()
    dshape, mshape = build_panels(out, fred_data, prices, file_data)
    t_panel = time.time() - t0

    cat = pd.DataFrame(rows)
    if cat.empty:
        cat = pd.DataFrame(columns=["source", "name", "description", "first", "last", "rows", "status", "seconds"])
    cat = cat.sort_values(["source", "name"])
    cat.to_csv(out / "catalog.csv", index=False)
    total = time.time() - t_start
    print()
    for group in ("fred", "yahoo", "files"):
        sub = cat[cat.source == group]
        if sub.empty:
            continue
        st = sub.status.astype(str)
        ok, cached = st.str.startswith(("ok", "cached")).sum(), st.str.startswith("cached").sum()
        extra = f" ({cached} cached)" if cached else ""
        print(f"  {group:<6} {ok:>3}/{len(sub):<3} ok{extra}")
    print(f"  panels  daily {dshape[0]:,} x {dshape[1]}   monthly {mshape[0]:,} x {mshape[1]}   ({t_panel:.1f}s)")
    print(f"  total   {total:.1f}s")
    bad = cat[~cat.status.astype(str).str.startswith(("ok", "cached"))]
    if len(bad):
        print(f"\n  {len(bad)} item(s) need attention (details in catalog.csv):")
        for _, r in bad.head(25).iterrows():
            print(f"    {r.source}:{r['name']}  {r.status}")
    if not any(manual.glob("PredictorData*.xls*")):
        print("\n  Optional: put the Goyal-Welch file (PredictorData*.xlsx, from sites.google.com/view/agoyal145) in "
              f"{manual} and rerun to add the 1871- predictor set.")
    return 0 if mshape[0] > 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Leave immediately: every file is already written, and skipping interpreter teardown avoids
    # crashes some compiled libraries (and daemon threads) cause while Python shuts down.
    os._exit(code or 0)
