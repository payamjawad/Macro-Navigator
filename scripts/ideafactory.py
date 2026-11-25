#!/usr/bin/env python3
"""
ideafactory.py - hundreds of home-made single-stock ideas, tested with a discovery / holdout protocol.

Protocol (fixed before looking at results):
 1. Ideas are generated systematically from building blocks; every feature is tried both ways (buy highest, buy lowest)
    and held 1 or 3 months. Universe: companies while they were in the S&P 500. Portfolio: top 10% by the feature, equal
    weight, 0.10% per name replaced. Measure: risk-adjusted alpha against the average member (stocksignals.evaluate rules).
 2. Discovery period (price ideas 1996-2012; fundamentals 2010-2017): an idea qualifies with alpha t >= 3 and the top 10%
    beating the bottom 10%. Near-duplicates (return correlation > 0.7 with a better qualifier) are merged; at most 20 go on.
 3. Luck benchmark: 150 random persistent signals through the same pipeline give the share of ideas that qualify by chance.
 4. Holdout (price 2013-now; fundamentals 2018-now), used once and only for the qualifiers: survives if alpha > 0 with
    t >= the one-sided Bonferroni bar for the number carried (about 2.8 for 20). The report also gives one aggregate:
    whether discovery rank predicted holdout performance across all ideas.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import earnings as ea
import stocksignals as ss

warnings.filterwarnings("ignore", category=RuntimeWarning)
PRICE_SPLIT = ("1996-01-01", "2012-12-31", "2013-01-01")
FUND_SPLIT = ("2010-01-01", "2017-12-31", "2018-01-01")
MAX_CARRY = 20
D = {"1w": 5, "1m": 21, "2m": 42, "3m": 63, "6m": 126, "9m": 189, "12m": 252, "18m": 378, "24m": 504, "36m": 756, "60m": 1260}


# --------------------------------------------------------------------------- helpers

def _bcast(s: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(s.values[:, None], like.shape[1], axis=1), index=like.index, columns=like.columns)


def _rcov(x: pd.DataFrame, y, w: int, mp: int):
    if isinstance(y, pd.Series):
        y = _bcast(y.reindex(x.index), x)
    mx, my = x.rolling(w, min_periods=mp).mean(), y.rolling(w, min_periods=mp).mean()
    return (x * y).rolling(w, min_periods=mp).mean() - mx * my


def _rcorr(x: pd.DataFrame, y, w: int, mp: int):
    if isinstance(y, pd.Series):
        y = _bcast(y.reindex(x.index), x)
    c = _rcov(x, y, w, mp)
    return c / np.sqrt(x.rolling(w, min_periods=mp).var(ddof=0) * y.rolling(w, min_periods=mp).var(ddof=0))


def bonferroni_t(n: int, alpha: float = 0.05) -> float:
    """One-sided bar for n ideas carried to the holdout (about 2.8 for 20)."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / max(1, n))


# --------------------------------------------------------------------------- price, volume, event features

def price_features(C: pd.DataFrame, O: pd.DataFrame, V: pd.DataFrame, me: pd.DatetimeIndex, ev: pd.DataFrame | None):
    """Yields (key, family, description, monthly frame). Built one at a time to keep memory small."""
    stocks = [c for c in C.columns if c != "SPY"]
    Cs, Os, Vs = C[stocks], O.reindex(columns=stocks), V.reindex(columns=stocks)
    spy = C["SPY"].pct_change(fill_method=None)
    R = Cs.pct_change(fill_method=None)
    DV = Cs * Vs
    ON = Os / Cs.shift(1) - 1
    ID = Cs / Os - 1
    at = lambda df: df.reindex(me).astype("float32")  # noqa: E731
    for k in ("1w", "1m", "2m", "3m", "6m", "9m", "12m", "18m", "24m", "36m", "60m"):
        yield f"ret_{k}", "Returns", f"{k} return", at(Cs / Cs.shift(D[k]) - 1)
    for k in ("3m", "6m", "12m"):
        yield f"ret_{k}_skip1m", "Returns", f"{k} return skipping the last month", at(Cs.shift(21) / Cs.shift(D[k]) - 1)
    yield "ret_36_12", "Returns", "return from 3 years to 1 year ago", at(Cs.shift(252) / Cs.shift(756) - 1)
    yield "ret_60_12", "Returns", "return from 5 years to 1 year ago", at(Cs.shift(252) / Cs.shift(1260) - 1)
    for k in ("1m", "3m", "6m", "12m"):
        yield f"vol_{k}", "Risk", f"daily volatility over {k}", at(R.rolling(D[k], min_periods=D[k] * 2 // 3).std())
    for k in ("3m", "12m"):
        yield f"downvol_{k}", "Risk", f"downside volatility over {k}", at(np.sqrt((R.clip(upper=0) ** 2).rolling(D[k], min_periods=D[k] * 2 // 3).mean()))
        yield f"upvol_{k}", "Risk", f"upside volatility over {k}", at(np.sqrt((R.clip(lower=0) ** 2).rolling(D[k], min_periods=D[k] * 2 // 3).mean()))
    v21, v63, v252 = (R.rolling(w, min_periods=w * 2 // 3).std() for w in (21, 63, 252))
    yield "volratio_1m_12m", "Risk", "1-month volatility vs 12-month", at(v21 / v252)
    yield "volratio_3m_12m", "Risk", "3-month volatility vs 12-month", at(v63 / v252)
    for k in ("3m", "12m"):
        yield f"skew_{k}", "Risk", f"skew of daily returns over {k}", at(R.rolling(D[k], min_periods=D[k] * 2 // 3).skew())
    yield "kurt_12m", "Risk", "fat tails (kurtosis) over 12m", at(R.rolling(252, min_periods=168).kurt())
    for k in ("1m", "3m"):
        yield f"max_{k}", "Risk", f"largest daily gain over {k}", at(R.rolling(D[k], min_periods=D[k] * 2 // 3).max())
        yield f"min_{k}", "Risk", f"largest daily loss over {k}", at(R.rolling(D[k], min_periods=D[k] * 2 // 3).min())
    cov = _rcov(R, spy, 252, 168)
    vm = spy.rolling(252, min_periods=168).var(ddof=0)
    beta = cov.div(vm, axis=0)
    yield "beta_12m", "Market link", "beta to the S&P over 12m (daily)", at(beta)
    yield "corr_12m", "Market link", "correlation with the S&P over 12m", at(_rcorr(R, spy, 252, 168))
    ivar = R.rolling(252, min_periods=168).var(ddof=0) - beta ** 2 * _bcast(vm.reindex(R.index), R)
    yield "idio_12m", "Market link", "volatility not explained by the S&P (12m)", at(np.sqrt(ivar.clip(lower=0)))
    cov63 = _rcov(R, spy, 63, 42)
    b63 = cov63.div(spy.rolling(63, min_periods=42).var(ddof=0), axis=0)
    iv63 = R.rolling(63, min_periods=42).var(ddof=0) - b63 ** 2 * _bcast(spy.rolling(63, min_periods=42).var(ddof=0).reindex(R.index), R)
    yield "idio_3m", "Market link", "volatility not explained by the S&P (3m)", at(np.sqrt(iv63.clip(lower=0)))
    yield "beta_3m_vs_12m", "Market link", "beta over 3m minus beta over 12m", at(b63 - beta)
    up = _bcast(spy > 0, R)
    upc = R.where(up).rolling(252, min_periods=60).mean() / _bcast(spy.where(spy > 0).rolling(252, min_periods=60).mean(), R)
    dnc = R.where(~up & R.notna()).rolling(252, min_periods=60).mean() / _bcast(spy.where(spy < 0).rolling(252, min_periods=60).mean(), R)
    yield "upcapture_12m", "Market link", "share of S&P up days captured (12m)", at(upc)
    yield "downcapture_12m", "Market link", "share of S&P down days captured (12m)", at(dnc)
    yield "capture_gap_12m", "Market link", "up capture minus down capture (12m)", at(upc - dnc)
    for w in (20, 50, 100, 200):
        yield f"vs_ma{w}", "Trend", f"price vs its {w}-day average", at(Cs / Cs.rolling(w, min_periods=w * 2 // 3).mean() - 1)
    ma50, ma200 = Cs.rolling(50, min_periods=35).mean(), Cs.rolling(200, min_periods=140).mean()
    yield "ma50_vs_ma200", "Trend", "50-day average vs 200-day average", at(ma50 / ma200 - 1)
    hi, lo = Cs.rolling(252, min_periods=200).max(), Cs.rolling(252, min_periods=200).min()
    yield "from_low_12m", "Trend", "distance above the 12-month low", at(Cs / lo - 1)
    yield "range_pos_12m", "Trend", "position in the 12-month range", at((Cs - lo) / (hi - lo))
    yield "days_above_ma200", "Trend", "share of the last year above the 200-day average", at((Cs > ma200).astype(float).where(ma200.notna()).rolling(252, min_periods=168).mean())
    dv21, dv63, dv252 = (DV.rolling(w, min_periods=w * 2 // 3).mean() for w in (21, 63, 252))
    yield "dollarvol_1m_12m", "Volume", "1-month trading value vs 12-month", at(dv21 / dv252)
    yield "dollarvol_3m_12m", "Volume", "3-month trading value vs 12-month", at(dv63 / dv252)
    yield "dollarvol_level", "Volume", "trading value (size proxy, 3m)", at(np.log(dv63))
    yield "dollarvol_cv_3m", "Volume", "how uneven trading value is (3m)", at(DV.rolling(63, min_periods=42).std() / dv63)
    yield "illiquidity_1m", "Volume", "price impact per dollar traded (1m)", at((R.abs() / DV).rolling(21, min_periods=14).mean())
    yield "illiquidity_12m", "Volume", "price impact per dollar traded (12m)", at((R.abs() / DV).rolling(252, min_periods=168).mean())
    vrel = Vs / Vs.rolling(63, min_periods=42).mean()
    yield "price_volume_corr_3m", "Volume", "correlation of returns with unusual volume (3m)", at(_rcorr(R, vrel, 63, 42))
    yield "updays_value_share_3m", "Volume", "share of trading value on up days (3m)", at(DV.where(R > 0).rolling(63, min_periods=42).sum() / DV.rolling(63, min_periods=42).sum())
    for k in ("1m", "3m", "12m"):
        yield f"overnight_{k}", "Overnight", f"overnight returns over {k}", at(ON.rolling(D[k], min_periods=D[k] * 2 // 3).sum())
        yield f"intraday_{k}", "Overnight", f"daytime returns over {k}", at(ID.rolling(D[k], min_periods=D[k] * 2 // 3).sum())
    yield "overnight_minus_day_12m", "Overnight", "overnight minus daytime returns (12m)", at(ON.rolling(252, min_periods=168).sum() - ID.rolling(252, min_periods=168).sum())
    for k in ("1m", "3m", "12m"):
        yield f"updays_{k}", "Consistency", f"share of up days over {k}", at((R > 0).astype(float).where(R.notna()).rolling(D[k], min_periods=D[k] * 2 // 3).mean())
    Pm = Cs.reindex(me)
    Rm = Pm.pct_change(fill_method=None)
    yield "upmonths_12m", "Consistency", "share of up months over 12m", ((Rm > 0).astype(float).where(Rm.notna()).rolling(12, min_periods=10).mean()).astype("float32")
    for k in ("3m", "12m"):
        yield f"autocorr_{k}", "Consistency", f"day-to-day return persistence over {k}", at(_rcorr(R, R.shift(1), D[k], D[k] * 2 // 3))
    yield "gaps_12m", "Gaps", "number of opening gaps over 3% (12m)", at((ON.abs() > 0.03).astype(float).where(ON.notna()).rolling(252, min_periods=168).sum())
    yield "avg_gap_3m", "Gaps", "average opening gap size (3m)", at(ON.abs().rolling(63, min_periods=42).mean())
    seas = lambda ks: (sum(Rm.shift(12 * k - 1).fillna(0) for k in ks) / sum(Rm.shift(12 * k - 1).notna().astype(int) for k in ks).replace(0, np.nan))  # noqa: E731
    yield "seasonal_1_5y", "Seasonality", "same calendar month, past 1-5 years", seas(range(1, 6)).astype("float32")
    yield "seasonal_6_10y", "Seasonality", "same calendar month, past 6-10 years", seas(range(6, 11)).astype("float32")
    other = (Rm.rolling(11, min_periods=9).sum() - 0)  # last 11 months exclude the coming calendar month's last-year value
    yield "other_months_avg", "Seasonality", "average of the other 11 months last year", (other / 11).astype("float32")
    if ev is not None and len(ev):
        yield from earnings_features(ev, C, me, stocks)


def earnings_features(ev: pd.DataFrame, C: pd.DataFrame, me: pd.DatetimeIndex, stocks: list):
    """Point-in-time features from announcement reaction days (known after the reaction-day close)."""
    days = C.index
    spy = C["SPY"].values
    rows = []
    for tk, g in ev[ev.ticker.isin(stocks)].groupby("ticker"):
        c = C[tk].values
        for rp in g.rpos.values:
            if rp < 1 or rp >= len(days) or np.isnan(c[rp]) or np.isnan(c[rp - 1]):
                continue
            rows.append((tk, days[rp], c[rp] / c[rp - 1] - (spy[rp] / spy[rp - 1]), c[rp]))
    if not rows:
        return
    E = pd.DataFrame(rows, columns=["ticker", "date", "react", "px"]).sort_values("date")
    E["abs4"] = E.groupby("ticker").react.transform(lambda s: s.abs().rolling(4, min_periods=2).mean())
    E["me"] = pd.Series(me, index=me).reindex(E.date, method="bfill").values             # first month-end on/after the date
    last = E.dropna(subset=["me"]).groupby(["me", "ticker"]).last()
    def panel(col):
        P = last[col].unstack().reindex(index=me, columns=stocks)
        return P.ffill(limit=4)                                                          # carry for up to 4 months
    react = panel("react")
    yield "earn_last_reaction", "Earnings", "last earnings-day move vs the S&P", react.astype("float32")
    yield "earn_avg_abs_move", "Earnings", "typical earnings-day move (last 4)", panel("abs4").astype("float32")
    pxr = panel("px")
    Pm = C[stocks].reindex(me)
    yield "earn_drift_since", "Earnings", "return since the last earnings day", (Pm / pxr - 1).astype("float32")
    dates = last["date"].unstack().reindex(index=me, columns=stocks).ffill(limit=4)
    now = pd.DataFrame(np.repeat(me.values[:, None], len(stocks), axis=1), index=me, columns=stocks)
    age = (now - dates).apply(lambda col: col.dt.days) / 30.0
    yield "earn_months_since", "Earnings", "time since the last earnings day", age.astype("float32")


REL_LIST = {"ret_1m", "ret_3m", "ret_6m", "ret_12m_skip1m", "vol_12m", "idio_12m", "beta_12m", "overnight_12m", "intraday_12m",
            "updays_12m", "dollarvol_1m_12m", "vs_ma200", "range_pos_12m", "skew_12m", "max_1m", "illiquidity_12m",
            "upcapture_12m", "downcapture_12m", "volratio_1m_12m", "earn_last_reaction"}
CHG_LIST = {"vol_12m", "beta_12m", "idio_12m", "dollarvol_1m_12m", "updays_12m", "skew_12m", "corr_12m", "overnight_12m",
            "range_pos_12m", "illiquidity_12m"}


# --------------------------------------------------------------------------- company fundamentals (SEC XBRL, first-reported values)

CONCEPTS = {
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"], "q"),
    "gross_profit": (["GrossProfit"], "q"),
    "op_income": (["OperatingIncomeLoss"], "q"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "q"),
    "rnd": (["ResearchAndDevelopmentExpense"], "q"),
    "net_income_a": (["NetIncomeLoss", "ProfitLoss"], "a"),
    "revenue_a": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                   "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"], "a"),
    "cfo_a": (["NetCashProvidedByUsedInOperatingActivities"], "a"),
    "capex_a": (["PaymentsToAcquirePropertyPlantAndEquipment"], "a"),
    "dividends_a": (["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "a"),
    "buybacks_a": (["PaymentsForRepurchaseOfCommonStock"], "a"),
    "assets": (["Assets"], "i"),
    "equity": (["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "i"),
    "liabilities": (["Liabilities"], "i"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], "i"),
    "inventory": (["InventoryNet"], "i"),
    "receivables": (["AccountsReceivableNetCurrent"], "i"),
    "shares": (["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"], "i"),
    "float_usd": (["EntityPublicFloat"], "i"),
}


def parse_companyfacts(js: dict) -> pd.DataFrame:
    """Rows: key, end, start, val, filed. Only the concepts in CONCEPTS; durations sorted into quarter / year."""
    out = []
    facts = js.get("facts", {})
    for key, (tags, kind) in CONCEPTS.items():
        for pri, tag in enumerate(tags):
            for ns in ("us-gaap", "dei"):
                node = facts.get(ns, {}).get(tag)
                if not node:
                    continue
                for unit, items in node.get("units", {}).items():
                    if not (unit == "USD" or unit == "shares"):
                        continue
                    for it in items:
                        end, filed = it.get("end"), it.get("filed")
                        if not end or not filed or it.get("val") is None:
                            continue
                        start = it.get("start")
                        if kind in ("q", "a"):
                            if not start:
                                continue
                            dur = (pd.Timestamp(end) - pd.Timestamp(start)).days
                            if kind == "q" and not 80 <= dur <= 100:
                                continue
                            if kind == "a" and not 350 <= dur <= 380:
                                continue
                        out.append((key, pri, end, filed, float(it["val"])))
    if not out:
        return pd.DataFrame(columns=["key", "end", "filed", "val"])
    df = pd.DataFrame(out, columns=["key", "pri", "end", "filed", "val"])
    df["end"], df["filed"] = pd.to_datetime(df.end), pd.to_datetime(df.filed)
    # first-reported value for each period: earliest filing, then the preferred tag
    df = df.sort_values(["key", "end", "filed", "pri"]).groupby(["key", "end"], as_index=False).first()
    return df[["key", "end", "filed", "val"]]


def fetch_fundamentals(store: Path, tickers: list, budget_s: float = 3600) -> dict:
    """Compact first-reported facts per ticker, cached; downloads SEC company facts once (gzip)."""
    t0 = time.time()
    ciks = ea.ticker_ciks(store)
    folder = store / "sec" / "facts"
    folder.mkdir(parents=True, exist_ok=True)
    out, n_new = {}, 0
    for i, tk in enumerate(tickers):
        cik = ciks.get(tk)
        if cik is None:
            continue
        path = folder / f"{cik}.pkl"
        if path.exists():
            out[tk] = pd.read_pickle(path)
            continue
        if time.time() - t0 > budget_s:
            continue
        try:
            r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=ea.SEC_UA, timeout=60)
            time.sleep(0.12)
            if r.status_code == 404:
                pd.DataFrame(columns=["key", "end", "filed", "val"]).to_pickle(path)
                continue
            r.raise_for_status()
            df = parse_companyfacts(r.json())
        except Exception:
            continue
        df.to_pickle(path)
        out[tk] = df
        n_new += 1
        if n_new % 50 == 0:
            print(f"  company financials: {len(out)} companies ({n_new} downloaded this run)", flush=True)
    return out


def _asof(df: pd.DataFrame, key: str, me: pd.DatetimeIndex, yoy: bool = False) -> pd.Series:
    """Latest period's value known at each month-end (by filing date); with yoy, the value one year before that period."""
    import bisect
    f = df[df.key == key].sort_values("filed")
    if f.empty:
        return pd.Series(np.nan, index=me)
    ends, vals_by_end, latest = [], {}, None
    out = []
    for row in f.itertuples():
        e = row.end
        if e not in vals_by_end:
            vals_by_end[e] = row.val
            bisect.insort(ends, e)
        latest = e if latest is None or e > latest else latest
        if yoy:
            target = latest - pd.Timedelta(days=365)
            i = bisect.bisect_left(ends, target)
            best = None
            for j in (i - 1, i):
                if 0 <= j < len(ends) and abs((ends[j] - target).days) <= 20:
                    if best is None or abs((ends[j] - target).days) < abs((best - target).days):
                        best = ends[j]
            v = vals_by_end[best] if best is not None else np.nan
        else:
            v = vals_by_end[latest]
        out.append((row.filed, v, latest))
    s = pd.DataFrame(out, columns=["filed", "v", "end"]).groupby("filed").last()
    idx = s.index.union(me)
    res = s.v.reindex(idx).ffill().reindex(me)
    period_end = pd.to_datetime(s.end.reindex(idx).ffill().reindex(me))
    age = (pd.Series(me, index=me) - period_end).dt.days
    return res.where(age <= 550)                                                      # drop data older than about 18 months


def fundamental_features(facts: dict, C: pd.DataFrame, me: pd.DatetimeIndex):
    stocks = [t for t in facts if t in C.columns]
    def panel(key, yoy=False):
        return pd.DataFrame({t: _asof(facts[t], key, me, yoy) for t in stocks}, index=me)
    P = C[stocks].reindex(me)
    rev, rev_y = panel("revenue"), panel("revenue", True)
    gp, gp_y = panel("gross_profit"), panel("gross_profit", True)
    oi, oi_y = panel("op_income"), panel("op_income", True)
    ni, ni_y = panel("net_income"), panel("net_income", True)
    rnd = panel("rnd")
    ni_a, rev_a, cfo, capex = panel("net_income_a"), panel("revenue_a"), panel("cfo_a"), panel("capex_a")
    div, bb = panel("dividends_a"), panel("buybacks_a")
    assets, assets_y = panel("assets"), panel("assets", True)
    equity, liab, cash = panel("equity"), panel("liabilities"), panel("cash")
    inv, inv_y, recv, recv_y = panel("inventory"), panel("inventory", True), panel("receivables"), panel("receivables", True)
    sh, sh_y = panel("shares"), panel("shares", True)
    # market value: public float (dollars, split-free) carried forward with the split-adjusted price ratio
    fl = {}
    for t in stocks:
        f = facts[t][facts[t].key == "float_usd"].sort_values("filed")
        if f.empty:
            continue
        s = pd.Series(np.nan, index=me)
        for row in f.itertuples():
            px0 = C[t].asof(row.end) if row.end >= C.index[0] else np.nan
            s.loc[s.index >= row.filed] = row.val / px0 if px0 and not np.isnan(px0) else np.nan
        fl[t] = s
    mcap = (pd.DataFrame(fl, index=me).reindex(columns=stocks) * P)
    g = lambda a, b: (a - b) / b.abs().where(b.abs() > 0)  # noqa: E731
    rg = g(rev, rev_y)
    gm, gm_y = gp / rev, gp_y / rev_y
    om, om_y = oi / rev, oi_y / rev_y
    shchg = (sh / sh_y - 1).where(((sh / sh_y) < 1.4) & ((sh / sh_y) > 0.7))                    # big jumps are splits
    feats = [
        ("f_revenue_growth", "Growth", "revenue growth vs a year ago (quarter)", rg),
        ("f_revenue_accel", "Growth", "revenue growth now minus 3 months ago", rg - rg.shift(3)),
        ("f_earnings_growth", "Growth", "net income growth vs a year ago (quarter)", g(ni, ni_y)),
        ("f_op_income_growth", "Growth", "operating income growth vs a year ago", g(oi, oi_y)),
        ("f_gross_margin", "Profitability", "gross margin", gm),
        ("f_gross_margin_chg", "Profitability", "gross margin vs a year ago", gm - gm_y),
        ("f_op_margin", "Profitability", "operating margin", om),
        ("f_op_margin_chg", "Profitability", "operating margin vs a year ago", om - om_y),
        ("f_net_margin", "Profitability", "net margin", ni / rev),
        ("f_roa", "Profitability", "return on assets (annual)", ni_a / assets),
        ("f_roe", "Profitability", "return on equity (annual)", (ni_a / equity).where(equity > 0)),
        ("f_asset_turnover", "Profitability", "sales per dollar of assets", rev_a / assets),
        ("f_gross_profit_assets", "Profitability", "gross profit per dollar of assets (quarter x4)", 4 * gp / assets),
        ("f_accruals", "Quality", "earnings not backed by cash (annual)", (ni_a - cfo) / assets),
        ("f_cash_conversion", "Quality", "operating cash flow vs net income", (cfo / ni_a).where(ni_a > 0)),
        ("f_inventory_vs_sales", "Quality", "inventory growth minus sales growth", g(inv, inv_y) - rg),
        ("f_receivables_vs_sales", "Quality", "receivables growth minus sales growth", g(recv, recv_y) - rg),
        ("f_asset_growth", "Investment", "asset growth vs a year ago", g(assets, assets_y)),
        ("f_capex_assets", "Investment", "capital spending per dollar of assets", capex / assets),
        ("f_rnd_sales", "Investment", "R&D per dollar of sales", rnd / rev),
        ("f_share_change", "Capital", "share count change vs a year ago", shchg),
        ("f_buyback_yield", "Capital", "buybacks vs market value", bb / mcap),
        ("f_dividend_yield", "Capital", "dividends vs market value", div / mcap),
        ("f_leverage", "Balance sheet", "liabilities per dollar of assets", liab / assets),
        ("f_cash_assets", "Balance sheet", "cash per dollar of assets", cash / assets),
        ("f_earnings_yield", "Valuation", "annual earnings vs market value", ni_a / mcap),
        ("f_sales_yield", "Valuation", "annual sales vs market value", rev_a / mcap),
        ("f_book_yield", "Valuation", "book value vs market value", equity / mcap),
        ("f_fcf_yield", "Valuation", "free cash flow vs market value", (cfo - capex) / mcap),
        ("f_size", "Valuation", "market value (log)", np.log(mcap.where(mcap > 0))),
    ]
    for key, fam, desc, F in feats:
        yield key, fam, desc, F.replace([np.inf, -np.inf], np.nan).astype("float32")


F_REL = {"f_revenue_growth", "f_gross_margin", "f_op_margin", "f_roa", "f_accruals", "f_asset_growth", "f_leverage",
         "f_earnings_yield", "f_sales_yield", "f_book_yield", "f_fcf_yield", "f_rnd_sales"}


# --------------------------------------------------------------------------- testing

def hac_alpha(y: pd.Series, x: pd.Series, lags: int = 3) -> dict | None:
    """Alpha from y = a + b x with Newey-West standard errors (allows month-to-month correlation)."""
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    n = len(d)
    if n < 36:
        return None
    X = np.column_stack([np.ones(n), d.x.values])
    b, *_ = np.linalg.lstsq(X, d.y.values, rcond=None)
    e = d.y.values - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * n / (n - 2)
    return {"alpha": float(b[0]), "t": float(b[0] / math.sqrt(V[0, 0])), "beta": float(b[1]), "months": n}


def stats(pf: pd.DataFrame, split: tuple, part: str):
    when = pd.Series(pf.index + pd.offsets.MonthEnd(1), index=pf.index)
    sel = (when >= split[0]) & (when <= split[1]) if part == "discovery" else (when >= split[2])
    p = pf[sel]
    a = hac_alpha(p.top, p.avg)
    tb = ss.mean_t(p.top - p.bottom)
    return a, tb, (p.top - p.avg)


def run_block(name: str, feature_iter, mask: pd.DataFrame, F1: pd.DataFrame, split: tuple, groups, rel_list, chg_list,
              n_placebo: int = 150, seed: int = 7) -> dict:
    t0 = time.time()
    ideas, series_d = [], {}
    holdout_cache = {}
    for key, fam, desc, F in feature_iter:
        for vkey, vdesc, G in (variants_generic(key, desc, F, mask, groups, rel_list, chg_list)):
            for sign, word in ((1, "highest"), (-1, "lowest")):
                for hold in (1, 3):
                    pf = ss.portfolios(sign * G, F1, mask, hold=hold)
                    if len(pf) < 60:
                        continue
                    a, tb, d = stats(pf, split, "discovery")
                    if not a:
                        continue
                    idea_id = f"{vkey}|{word}|{hold}m"
                    ideas.append({"id": idea_id, "feature": vkey, "family": fam, "description": f"Buy the {word}: {vdesc}",
                                  "hold": hold, "disc_alpha": a["alpha"], "disc_t": a["t"], "disc_top_minus_bottom": (tb or {}).get("mean"),
                                  "turnover": float(pf.turnover.mean())})
                    series_d[idea_id] = d
                    ah, tbh, _ = stats(pf, split, "holdout")
                    holdout_cache[idea_id] = (ah, tbh)
    # luck benchmark: random persistent signals through the same pipeline
    rng = np.random.default_rng(seed)
    pl_t = []
    for i in range(n_placebo):
        phi = (0.0, 0.5, 0.9)[i % 3]
        Z = np.zeros(mask.shape, dtype="float32")
        e = rng.standard_normal(mask.shape).astype("float32")
        Z[0] = e[0]
        for k in range(1, mask.shape[0]):
            Z[k] = phi * Z[k - 1] + e[k]
        G = pd.DataFrame(Z, index=mask.index, columns=mask.columns).where(mask)
        pf = ss.portfolios(G, F1, mask, hold=1 + (i % 2) * 2)
        if len(pf) < 60:
            continue
        a, _, _ = stats(pf, split, "discovery")
        if a:
            pl_t.append(a["t"])
    pl_t = np.array(pl_t)
    share_chance = None
    if len(pl_t) >= 10 and pl_t.std() > 0:
        from statistics import NormalDist
        share_chance = 1 - NormalDist(float(pl_t.mean()), float(pl_t.std())).cdf(3.0)     # smooth estimate of P(t >= 3) by luck
    # selection in discovery
    qual = [x for x in ideas if x["disc_t"] >= 3 and (x["disc_top_minus_bottom"] or -1) > 0]
    qual.sort(key=lambda x: -x["disc_t"])
    carried = []
    for x in qual:
        if len(carried) >= MAX_CARRY:
            break
        s = series_d[x["id"]]
        if any(abs(s.corr(series_d[c["id"]])) > 0.7 for c in carried if len(pd.concat([s, series_d[c["id"]]], axis=1, sort=True).dropna()) > 24):
            continue
        carried.append(x)
    bar = max(2.5, bonferroni_t(len(carried))) if carried else None
    for x in carried:
        ah, tbh = holdout_cache[x["id"]]
        x["hold_alpha"], x["hold_t"] = (ah or {}).get("alpha"), (ah or {}).get("t")
        x["hold_top_minus_bottom"] = (tbh or {}).get("mean")
        if not ah:
            x["verdict"] = "Too little data"
        elif ah["alpha"] > 0 and ah["t"] >= bar:
            x["verdict"] = "Survived"
        elif ah["alpha"] > 0:
            x["verdict"] = "Weakened"
        else:
            x["verdict"] = "Failed"
    # aggregate only: did discovery rank predict the holdout, across all ideas?
    dt = pd.Series({x["id"]: x["disc_t"] for x in ideas})
    ht = pd.Series({k: (v[0] or {}).get("t") for k, v in holdout_cache.items()}).reindex(dt.index)
    ok = ht.notna()
    rank_corr = float(dt[ok].rank().corr(ht[ok].rank())) if ok.sum() > 20 else None
    q_hi = dt[ok] >= dt[ok].quantile(0.9)
    agg = {"rank_corr": rank_corr,
           "holdout_t_top10pct_of_ideas": float(ht[ok][q_hi].mean()) if q_hi.any() else None,
           "holdout_t_all_ideas": float(ht[ok].mean()) if ok.any() else None}
    fam = {}
    for x in ideas:
        f = fam.setdefault(x["family"], {"ideas": 0, "disc_t_ge_3": 0})
        f["ideas"] += 1
        f["disc_t_ge_3"] += int(x["disc_t"] >= 3)
    hist_edges = [-99, -3, -2, -1, 0, 1, 2, 3, 4, 99]
    hist = {"edges": hist_edges[1:-1], "ideas": np.histogram([x["disc_t"] for x in ideas], bins=hist_edges)[0].tolist(),
            "placebo": np.histogram(pl_t, bins=hist_edges)[0].tolist() if len(pl_t) else []}
    return {"block": name, "periods": {"discovery": f"{split[0][:4]}–{split[1][:4]}", "holdout": f"{split[2][:4]}–now"},
            "tests": len(ideas), "qualified": len(qual), "expected_by_chance": (share_chance * len(ideas)) if share_chance is not None else None,
            "placebo_runs": int(len(pl_t)), "placebo_share_t_ge_3": share_chance, "carried": carried, "bar_t": bar,
            "aggregate": agg, "families": fam, "histogram": hist, "seconds": round(time.time() - t0),
            "ideas": [{"id": x["id"], "feature": x["feature"], "family": x["family"], "disc_t": x["disc_t"],
                       "sign": 1 if "|highest|" in x["id"] else -1} for x in ideas]}


def variants_generic(key, desc, F, mask, groups, rel_list, chg_list):
    F = F.reindex(columns=mask.columns).where(mask)
    yield key, desc, F
    if groups is not None and key in rel_list:
        g_ = groups.reindex(F.columns)
        has = g_.notna()
        if has.sum() >= 30:
            sub = F.loc[:, has]
            med = sub.T.groupby(g_[has]).transform("median").T
            n = sub.notna().T.groupby(g_[has]).transform("sum").T
            yield f"{key}_vs_industry", f"{desc}, relative to its industry", (sub - med).where(n >= 3).reindex(columns=F.columns)
    if key in chg_list:
        yield f"{key}_change_3m", f"change in {desc} over 3 months", F - F.shift(3)


def run(store: Path, tickers: list, membership: pd.DataFrame | None = None, fundamentals: bool = True, n_placebo: int = 150,
        fund_budget_s: float = 3600, price: bool = True, min_fund_companies: int = 100) -> dict:
    t0 = time.time()
    stocks = [t for t in tickers if not str(t).startswith("^") and t not in ea.ETFS]
    px = ea.daily_prices(store, sorted(set(stocks) | {"SPY"}))
    keep = [c for c in px["close"].columns if c in set(stocks) | {"SPY"}]
    C = px["close"][keep].sort_index()
    C = C[C.index >= "1990-01-01"]
    O = px["open"].reindex(index=C.index, columns=keep)
    V = px["volume"].reindex(index=C.index, columns=keep)
    me = ss.month_ends(C.index)
    names = [c for c in keep if c != "SPY"]
    M = ss.membership_matrix(membership, me, names)
    mask = M.reindex(index=me, columns=names).fillna(False).astype(bool)
    Rm = C[names].reindex(me).pct_change(fill_method=None)
    F1 = Rm.shift(-1).where(mask)
    sic = ss.sic_codes(store, names)
    groups = pd.Series(sic) if sic else None
    ev = None
    try:
        ann = ea.announcements(store, stocks, budget_s=0)
        if len(ann):
            ev = ea.classify(ann, C.index)
    except Exception:
        pass
    price_res = None
    if price:
        print("  price, volume and event ideas (a few minutes)...", flush=True)
        price_res = run_block("Price, volume and event ideas", price_features(C, O, V, me, ev), mask, F1, PRICE_SPLIT, groups,
                              REL_LIST, CHG_LIST, n_placebo=n_placebo)
    fund = None
    if fundamentals:
        print("  company financials (first run downloads SEC filings, 15-30 minutes)...", flush=True)
        facts = fetch_fundamentals(store, names, budget_s=fund_budget_s)
        facts = {t: df for t, df in facts.items() if len(df)}
        if len(facts) >= min_fund_companies:
            fund = run_block("Company fundamentals", fundamental_features(facts, C, me), mask, F1, FUND_SPLIT, groups,
                             F_REL, set(), n_placebo=n_placebo)
            fund["companies_with_financials"] = len(facts)
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "seconds": round(time.time() - t0),
            "price": price_res, "fundamentals": fund,
            "method": ("Top 10% of S&P 500 members at the time by each idea, equal weight, held 1 or 3 months, 0.10% per name "
                       "replaced; alpha = risk-adjusted return against the average member. Qualify in discovery with t >= 3 and "
                       "top beating bottom; near-duplicates merged; at most 20 carried; holdout used once for those only, with a "
                       "Bonferroni bar for the number carried. Luck benchmark: random persistent signals through the same steps.")}
