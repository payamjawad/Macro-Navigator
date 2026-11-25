import sys, pathlib
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import stocksignals as ss


def _world(persist: float, seed=0, n=150):
    """Daily prices 1998-2025. persist > 0 gives each stock a slowly changing drift (past winners keep winning)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1998-01-02", "2025-12-31")
    T = len(days)
    spy = 0.0003 + rng.normal(0, 0.01, T)
    cols = {"SPY": 100 * np.exp(np.cumsum(spy))}
    for i in range(n):
        drift = np.zeros(T)
        if persist:
            shocks = rng.normal(0, persist, T // 21 + 2)
            level = np.zeros_like(shocks)
            for k in range(1, len(shocks)):
                level[k] = 0.95 * level[k - 1] + shocks[k]            # persistent monthly expected-return component
            drift = np.repeat(level, 21)[:T]
        r = 0.9 * spy + drift + rng.normal(0, 0.015, T)
        cols[f"S{i:03d}"] = 50 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=days)


def _run(C):
    me = ss.month_ends(C.index)
    M = ss.membership_matrix(None, me, [c for c in C.columns if c != "SPY"])
    S, F1, spy, mask = ss.build_signals(C, M)
    spy_m = C["SPY"].reindex(me); broken = spy_m < spy_m.rolling(10).mean()
    out = {}
    for key in ("momentum", "resmom", "high52", "reversal", "low_max", "low_vol", "seasonal", "frog"):
        out[key] = ss.evaluate(ss.portfolios(S[key], F1, mask, hold=1), spy, broken)
    return out


def test_finds_planted_momentum():
    res = _run(_world(persist=0.0006))
    m = res["momentum"]
    assert m["late"]["alpha"]["alpha"] > 0 and m["late"]["alpha"]["t"] > 3 and m["verdict"] == "Solid"


def test_no_false_solid_in_random_worlds():
    solid = 0
    for seed in range(3):
        res = _run(_world(persist=0.0, seed=10 + seed))
        solid += sum(r["verdict"] == "Solid" for r in res.values())
    assert solid == 0


def test_portfolio_cost_and_turnover():
    idx = pd.date_range("2000-01-31", periods=50, freq="ME")
    cols = [f"T{i}" for i in range(40)]
    score = pd.DataFrame(np.tile(np.arange(40), (50, 1)), index=idx, columns=cols, dtype=float)   # same top names every month
    F1 = pd.DataFrame(0.01, index=idx, columns=cols)
    mask = pd.DataFrame(True, index=idx, columns=cols)
    pf = ss.portfolios(score, F1, mask, hold=1)
    assert pf.turnover.iloc[5:].max() == 0 and abs(pf.top.iloc[5:].mean() - 0.01) < 1e-12        # no turnover, no cost
