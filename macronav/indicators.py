"""Turn raw series into the booklet's signals on one business-day calendar.

Each macro series is shifted by its publication lag before it is used, so a value only appears
on the day it would have been known.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import FRED_SERIES, RawData


def _on_calendar(s: pd.Series | None, cal: pd.DatetimeIndex, lag_days: int = 0) -> pd.Series:
    if s is None or s.dropna().empty:
        return pd.Series(np.nan, index=cal)
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index) + pd.Timedelta(days=lag_days)
    s = s[~s.index.duplicated(keep="last")]
    joined = s.reindex(s.index.union(cal)).sort_index().ffill()
    return joined.reindex(cal)


def _monthly_fill_gaps(s: pd.Series | None) -> pd.Series | None:
    """Monthly index levels with a missing month (e.g. Oct 2025 shutdown) are interpolated."""
    if s is None or s.empty:
        return s
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    return full.interpolate(limit=2)


def _consecutive_true(b: pd.Series) -> pd.Series:
    b = b.fillna(False).astype(bool)
    groups = (~b).cumsum()
    return b.astype(int).groupby(groups).cumsum()


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def recession_probability(spread_10y3m: float) -> float:
    """NY Fed / Estrella-Mishkin probit: P(recession in 12 months) from the 10Y-3M spread (pp)."""
    if spread_10y3m is None or (isinstance(spread_10y3m, float) and math.isnan(spread_10y3m)):
        return float("nan")
    return norm_cdf(-0.5333 - 0.6330 * spread_10y3m)


def build_panel(raw: RawData, universe: dict) -> pd.DataFrame:
    prices = raw.prices.copy()
    end = pd.Timestamp.today().normalize()
    if not prices.empty:
        end = max(end, prices.index.max())
    starts = [s.index.min() for s in raw.fred.values() if s is not None and not s.empty]
    if not prices.empty:
        starts.append(prices.index.min())
    if not starts:
        raise ValueError("No data was downloaded. Check the FRED_API_KEY secret and the network, then retry.")
    start = min(starts)
    cal = pd.bdate_range(start, end)
    f = raw.fred
    lag = FRED_SERIES
    P = pd.DataFrame(index=cal)

    # --- inflation (computed on native monthly data, then lagged)
    pce = _monthly_fill_gaps(f.get("PCEPILFE"))
    cpi = _monthly_fill_gaps(f.get("CPILFESL"))
    P["core_pce_yoy"] = _on_calendar(None if pce is None else (pce / pce.shift(12) - 1) * 100, cal, lag["PCEPILFE"])
    P["core_cpi_3m_ann"] = _on_calendar(None if cpi is None else ((cpi / cpi.shift(3)) ** 4 - 1) * 100, cal, lag["CPILFESL"])
    P["core_cpi_yoy"] = _on_calendar(None if cpi is None else (cpi / cpi.shift(12) - 1) * 100, cal, lag["CPILFESL"])
    brent = _on_calendar(f.get("DCOILBRENTEU"), cal, 1)
    P["brent"] = brent
    P["brent_yoy"] = (brent / brent.shift(261) - 1) * 100
    P["breakeven_5y5y"] = _on_calendar(f.get("T5YIFR"), cal, 1)

    # --- policy and rates
    ff = _on_calendar(f.get("DFEDTARU"), cal, 1)
    P["ff_upper"] = ff
    P["ff_change_3m"] = ff - ff.shift(65)
    P["ff_change_6m"] = ff - ff.shift(130)
    P["ff_change_12m"] = ff - ff.shift(261)
    P["dgs2"] = _on_calendar(f.get("DGS2"), cal, 1)
    P["dgs10"] = _on_calendar(f.get("DGS10"), cal, 1)
    P["real_10y"] = _on_calendar(f.get("DFII10"), cal, 1)
    P["two_y_minus_ff"] = P["dgs2"] - ff
    P["real_ff"] = _on_calendar(f.get("DFF"), cal, 1) - P["core_pce_yoy"]
    P["t10y3m"] = _on_calendar(f.get("T10Y3M"), cal, 1)
    P["t10y2y"] = _on_calendar(f.get("T10Y2Y"), cal, 1)
    P["dgs10_change_6m"] = P["dgs10"] - P["dgs10"].shift(130)
    P["recession_prob_12m"] = P["t10y3m"].rolling(21, min_periods=5).mean().apply(recession_probability)
    d2, d10 = P["dgs2"].diff(20), P["dgs10"].diff(20)
    pattern = np.select(
        [(d2.abs() >= d10.abs()) & (d2 > 0), (d2.abs() >= d10.abs()) & (d2 <= 0),
         (d10.abs() > d2.abs()) & (d10 > 0), (d10.abs() > d2.abs()) & (d10 <= 0)],
        ["bear flattener", "bull steepener", "bear steepener", "bull flattener"], default="flat")
    P["curve_pattern"] = pd.Series(pattern, index=cal).where(d2.notna() & d10.notna(), None)
    P["inverted_within_6m"] = (P["t10y3m"] < 0).astype(float).rolling(130, min_periods=1).max().astype(bool)

    # --- growth and labour
    P["sahm"] = _on_calendar(f.get("SAHMREALTIME"), cal, lag["SAHMREALTIME"])
    P["unrate"] = _on_calendar(f.get("UNRATE"), cal, lag["UNRATE"])
    icsa = f.get("ICSA")
    if icsa is not None and not icsa.empty:
        ma4 = icsa.rolling(4, min_periods=4).mean()
        P["claims_vs_low"] = _on_calendar(ma4 / ma4.rolling(52, min_periods=26).min() - 1, cal, lag["ICSA"])
        P["claims_4wk"] = _on_calendar(ma4, cal, lag["ICSA"])
    else:
        P["claims_vs_low"] = np.nan
        P["claims_4wk"] = np.nan
    P["philly"] = _on_calendar(f.get("GACDFSA066MSFRBPHI"), cal, lag["GACDFSA066MSFRBPHI"])
    permit = f.get("PERMIT")
    P["permits_yoy"] = _on_calendar(None if permit is None else (permit / permit.shift(12) - 1) * 100, cal, lag["PERMIT"])

    # --- credit and funding
    hy = _on_calendar(f.get("BAMLH0A0HYM2"), cal, 1) * 100
    P["hy_bp"] = hy
    P["hy_min_126"] = hy.rolling(126, min_periods=60).min()
    P["hy_max_126"] = hy.rolling(126, min_periods=60).max()
    P["hy_10d_ago"] = hy.shift(10)
    P["hy_change_63"] = hy - hy.shift(63)
    P["baa10y"] = _on_calendar(f.get("BAA10Y"), cal, 1)
    P["nfci"] = _on_calendar(f.get("NFCI"), cal, lag["NFCI"])
    P["cp_ff_bp"] = _on_calendar(f.get("CPFF"), cal, 1) * 100
    P["sofr_iorb_bp"] = (_on_calendar(f.get("SOFR"), cal, 1) - _on_calendar(f.get("IORB"), cal, 1)) * 100
    dollar = _on_calendar(f.get("DTWEXBGS"), cal, 1)
    P["dollar_yoy"] = (dollar / dollar.shift(261) - 1) * 100

    # --- markets
    def px(t):
        return prices[t].reindex(cal).ffill() if t in prices.columns else pd.Series(np.nan, index=cal)

    vix = px("^VIX")
    if vix.dropna().empty:
        vix = _on_calendar(f.get("VIXCLS"), cal, 0)
    P["vix"] = vix
    P["vix3m"] = px("^VIX3M")
    P["vix_ratio"] = P["vix"] / P["vix3m"]
    P["vix_ratio_min_3d"] = P["vix_ratio"].rolling(3).min()
    P["vix_max_60"] = vix.rolling(60, min_periods=10).max()
    P["vix_avg_5"] = vix.rolling(5).mean()
    P["vix_avg_20"] = vix.rolling(20).mean()
    P["vix_days_above_25"] = _consecutive_true(vix >= 25)
    P["move"] = px("^MOVE")

    spx = px("^GSPC")
    if spx.dropna().empty:
        spx = px(universe.get("benchmark", "SPY"))
    P["spx"] = spx
    P["drawdown_pct"] = (spx / spx.rolling(252, min_periods=20).max() - 1) * 100
    spx_d = spx.dropna()
    if spx_d.empty:
        me_close = pd.Series(dtype=float)
    else:
        grp = spx_d.groupby(spx_d.index.to_period("M"))
        me_close = pd.Series(grp.last().to_numpy(), index=pd.DatetimeIndex(grp.apply(lambda s: s.index.max()).to_numpy()))
        if me_close.index[-1].to_period("M") == pd.Timestamp.today().to_period("M"):
            me_close = me_close.iloc[:-1]          # the current month is not complete yet
    me_sma = me_close.rolling(10, min_periods=10).mean()
    trend_down_me = (me_close < me_sma).where(me_sma.notna())
    above_count = _consecutive_true(me_close > me_sma)
    P["trend_down"] = _on_calendar(trend_down_me.astype(float), cal, 0).astype(float) > 0.5
    P["months_above_sma10"] = _on_calendar(above_count.astype(float), cal, 0).fillna(0)
    P["spx_sma10m"] = _on_calendar(me_sma, cal, 0)
    P["spx_last_month_close"] = _on_calendar(me_close, cal, 0)
    live_sma10 = pd.concat([me_close.tail(9), spx.dropna().tail(1)]).mean() if len(me_close) >= 9 else np.nan
    P.attrs["spx_sma10m_live"] = float(live_sma10) if live_sma10 == live_sma10 else None

    bench = universe.get("benchmark", "SPY")
    kre, spy, tlt = px("KRE"), px(bench), px("TLT")
    P["kre_rel_20d"] = ((kre / kre.shift(20)) - (spy / spy.shift(20))) * 100
    P["stock_bond_corr_60d"] = spy.pct_change().rolling(60).corr(tlt.pct_change())
    sector_tickers = [r["ticker"] for g in ("us_sectors", "us_industries") for r in universe.get(g, [])]
    above = []
    for t in sector_tickers:
        s = px(t)
        above.append((s > s.rolling(200, min_periods=150).mean()).where(s.notna()))
    P["sector_breadth_pct"] = pd.concat(above, axis=1).mean(axis=1) * 100 if above else np.nan
    return P
