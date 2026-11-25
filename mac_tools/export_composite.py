#!/usr/bin/env python3
"""One-off: register the forward-test composite from the idea factory's DISCOVERY results only.
Rule fixed in advance: the top 10% of ideas by discovery t in each block (price 1996-2012, fundamentals 2010-2017),
limited to features the cloud pipeline can compute, one direction per feature (the better one). Writes config/composite.json."""
import json, math, sys
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import earnings as ea  # noqa: E402
import ideafactory as ifa  # noqa: E402
import composite  # noqa: E402


def pick(block):
    ideas = sorted((block or {}).get("ideas", []), key=lambda x: -x["disc_t"])
    top = ideas[: max(1, math.ceil(0.1 * len(ideas)))] if ideas else []
    out, seen = [], set()
    for x in top:
        if x["family"] not in composite.COMPUTABLE or x["feature"] in seen:
            continue
        seen.add(x["feature"])
        out.append({"feature": x["feature"], "sign": x["sign"], "family": x["family"], "disc_t": round(x["disc_t"], 2)})
    return out


def main():
    store = Path.home() / "Downloads" / "MacroNavigator" / "data" / "store"
    tickers = [t.strip() for t in (HERE / "sp500_all_since_2001.txt").read_text().split() if t.strip()]
    mem = ea.load_membership(HERE.parent / "config" / "sp500_membership.csv")
    print("Recomputing discovery results from the data on this Mac (5-10 minutes)...", flush=True)
    res = ifa.run(store, tickers, mem, fundamentals=True, n_placebo=10, fund_budget_s=1800)
    feats = [dict(f, block="price") for f in pick(res.get("price"))] + [dict(f, block="fundamentals") for f in pick(res.get("fundamentals"))]
    reg = {"registered": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "rule": ("Average of signed cross-sectional percentile ranks of these features; top 10% of current S&P 500 members each "
                    "month-end, equal weight, held one month, 0.10% per name replaced; scored against the average member."),
           "source": "Idea factory discovery results only (price 1996-2012, fundamentals 2010-2017); holdout results not used.",
           "features": feats}
    out = HERE.parent / "config" / "composite.json"
    out.write_text(json.dumps(reg, indent=1))
    print(f"Registered {len(feats)} features ({sum(f['block'] == 'price' for f in feats)} price, "
          f"{sum(f['block'] == 'fundamentals' for f in feats)} fundamentals) -> {out}", flush=True)
    print("Done. Now upload the scripts, site and config folders to GitHub.", flush=True)


if __name__ == "__main__":
    main()
