"""News headlines via Exa's /search endpoint, used to complement social sentiment."""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

from .. import config


def search_news(query: str, days: int = 10, num_results: int | None = None) -> list[dict]:
    key = config.exa_key()
    published_after = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "query": query,
        "numResults": num_results or config.EXA_NEWS_RESULTS,
        "type": "neural",
        "category": "news",
        "startPublishedDate": published_after,
        "contents": {"text": {"maxCharacters": 600}},
    }
    request = urllib.request.Request(
        f"{config.EXA_BASE}/search",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)

    articles = []
    for result in body.get("results", []):
        articles.append({
            "title": result.get("title") or "",
            "url": result.get("url") or "",
            "published": result.get("publishedDate") or datetime.now(timezone.utc).isoformat(),
            "text": (result.get("text") or "")[:1200],
            "source": "exa",
        })
    return articles


def collect_news(assets: list[dict], days: int = 10, pause: float = 0.3) -> list[dict]:
    rows = []
    for asset in assets:
        query = f"{asset['name']} stock news"
        for article in search_news(query, days=days):
            article["asset"] = asset["ticker"]
            rows.append(article)
        time.sleep(pause)
    return rows
