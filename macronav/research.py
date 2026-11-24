"""Research watch: pulls new papers, scores them against the model's modules, flags what is new.

It does not change the model by itself. It surfaces papers so a person can decide whether a
threshold, signal or module should change (see docs/RESEARCH.md for the review routine).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone

import requests

from .data import DATA, load_yaml

SEEN_PATH = DATA / "research_seen.json"
UA = {"User-Agent": "macro-navigator research watch (personal use)"}


def _fetch_feed(feed: dict) -> list[dict]:
    import feedparser

    if feed.get("type") == "arxiv":
        params = {"search_query": feed["query"], "sortBy": "submittedDate", "sortOrder": "descending",
                  "max_results": feed.get("max_results", 100)}
        resp = requests.get(feed["url"], params=params, timeout=30, headers=UA)
    else:
        resp = requests.get(feed["url"], timeout=30, headers=UA)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    items = []
    for e in parsed.entries:
        when = e.get("published_parsed") or e.get("updated_parsed")
        published = datetime.fromtimestamp(time.mktime(when), tz=timezone.utc) if when else None
        authors = ", ".join(a.get("name", "") for a in e.get("authors", [])) if e.get("authors") else e.get("author", "")
        summary = re.sub(r"<[^>]+>", " ", e.get("summary", "") or "")
        items.append({"title": " ".join((e.get("title") or "").split()), "link": e.get("link", ""),
                      "authors": authors, "summary": " ".join(summary.split())[:600],
                      "published": published.isoformat() if published else None, "source": feed["name"]})
    return items


def _score(item: dict, modules: dict) -> tuple[int, str | None, list[str]]:
    title, text = item["title"].lower(), item["summary"].lower()
    best, best_mod, hits = 0, None, []
    total = 0
    for mod, spec in modules.items():
        s = 0
        for kw in spec["keywords"]:
            k = kw.lower()
            if k in title:
                s += 2 * spec.get("weight", 1)
                hits.append(kw)
            elif k in text:
                s += spec.get("weight", 1)
                hits.append(kw)
        total += s
        if s > best:
            best, best_mod = s, mod
    return total, best_mod, sorted(set(hits))


def research_watch(offline: bool = False) -> dict:
    cfg = load_yaml("research.yaml")
    if offline:
        return {"items": [], "status": {"research": "skipped (demo mode)"}, "checked": None}
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.get("lookback_days", 60))
    seen = set(json.loads(SEEN_PATH.read_text())) if SEEN_PATH.exists() else set()
    status, pool = {}, []
    for feed in cfg.get("feeds", []):
        try:
            got = _fetch_feed(feed)
            pool += got
            status[feed["name"]] = f"ok ({len(got)} items)"
        except Exception as err:
            status[feed["name"]] = f"failed: {err.__class__.__name__}"
    out = []
    for it in pool:
        if it["published"] and datetime.fromisoformat(it["published"]) < cutoff:
            continue
        score, module, hits = _score(it, cfg["modules"])
        if score < cfg.get("min_score", 3):
            continue
        out.append({**it, "score": score, "module": module, "keywords": hits, "new": it["link"] not in seen})
    out.sort(key=lambda r: (r["new"], r["score"]), reverse=True)
    out = out[: cfg.get("max_items", 30)]
    return {"items": out, "status": status, "checked": datetime.now(timezone.utc).isoformat()}


def remember(items: list[dict]) -> None:
    seen = set(json.loads(SEEN_PATH.read_text())) if SEEN_PATH.exists() else set()
    seen |= {i["link"] for i in items}
    SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PATH.write_text(json.dumps(sorted(seen)[-5000:], indent=0))
