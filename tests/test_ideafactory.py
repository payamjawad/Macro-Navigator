import sys, pathlib
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ideafactory as ifa, stocksignals as ss
from test_stocksignals import _world


def test_parse_keeps_first_reported_quarters():
    js = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2020-01-01", "end": "2020-03-31", "val": 100.0, "filed": "2020-05-01", "form": "10-Q"},
        {"start": "2020-01-01", "end": "2020-03-31", "val": 90.0, "filed": "2021-05-01", "form": "10-Q"},     # restated later
        {"start": "2020-01-01", "end": "2020-06-30", "val": 210.0, "filed": "2020-08-01", "form": "10-Q"},    # year-to-date: dropped
        {"start": "2020-01-01", "end": "2020-12-31", "val": 450.0, "filed": "2021-02-15", "form": "10-K"}]}}},
        "dei": {"EntityPublicFloat": {"units": {"USD": [{"end": "2020-06-30", "val": 5e9, "filed": "2021-02-15"}]}}}}}
    df = ifa.parse_companyfacts(js)
    q = df[df.key == "revenue"]
    assert len(q) == 1 and q.val.iloc[0] == 100.0                       # first-reported quarter only
    a = df[df.key == "revenue_a"]
    assert len(a) == 1 and a.val.iloc[0] == 450.0
    assert (df.key == "float_usd").sum() == 1


def test_asof_and_year_ago():
    rows = []
    for y in (2019, 2020, 2021):
        for qe, f in (("03-31", "05-01"), ("06-30", "08-01"), ("09-30", "11-01"), ("12-31", "02-15")):
            fy = y + 1 if qe == "12-31" else y
            rows.append(("revenue", pd.Timestamp(f"{y}-{qe}"), pd.Timestamp(f"{fy}-{f}"), float(100 * (y - 2018) + int(qe[:2]))))
    df = pd.DataFrame(rows, columns=["key", "end", "filed", "val"])
    me = pd.date_range("2019-01-31", "2023-12-31", freq="ME")
    now, ago = ifa._asof(df, "revenue", me), ifa._asof(df, "revenue", me, yoy=True)
    assert np.isnan(now[pd.Timestamp("2019-04-30")])                    # nothing filed yet
    assert now[pd.Timestamp("2020-05-31")] == 203.0 and ago[pd.Timestamp("2020-05-31")] == 103.0
    assert now[pd.Timestamp("2022-12-31")] == 312.0                     # 12 months old: still used
    assert np.isnan(now[pd.Timestamp("2023-08-31")])                    # older than about 18 months: dropped


def _inputs(persist, seed):
    C = _world(persist=persist, seed=seed, n=120)
    rng = np.random.default_rng(seed + 1)
    O = C.shift(1) * (1 + rng.normal(0, 0.003, C.shape))
    V = pd.DataFrame(rng.lognormal(13, 0.4, C.shape), index=C.index, columns=C.columns)
    me = ss.month_ends(C.index)
    names = [c for c in C.columns if c != "SPY"]
    mask = pd.DataFrame(True, index=me, columns=names)
    F1 = C[names].reindex(me).pct_change(fill_method=None).shift(-1).where(mask)
    return C, O, V, me, mask, F1


def _block(persist, seed):
    C, O, V, me, mask, F1 = _inputs(persist, seed)
    return ifa.run_block("test", ifa.price_features(C, O, V, me, None), mask, F1, ("1999-01-01", "2012-12-31", "2013-01-01"),
                         None, set(), ifa.CHG_LIST, n_placebo=30)


def test_random_world_has_no_survivors():
    res = _block(0.0, 21)
    assert res["tests"] > 300
    assert sum(x["verdict"] == "Survived" for x in res["carried"]) == 0
    exp = res["expected_by_chance"]
    assert res["qualified"] <= max(8, 3 * exp)                          # about what luck produces


def test_planted_effect_survives():
    res = _block(0.00015, 22)
    surv = [x for x in res["carried"] if x["verdict"] == "Survived"]
    assert surv and any(x["family"] in ("Returns", "Trend", "Consistency", "Overnight") for x in surv)
    assert res["aggregate"]["rank_corr"] > 0.3
