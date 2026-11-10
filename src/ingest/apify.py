"""Apify ingestion.

The brief asks for scraped posts from two different retail venues per exchange. Rather than
assume a bespoke actor exists for each venue, the pipeline drives Apify's generic
`website-content-crawler` with per-venue start URLs and turns the returned page text into
post records, then optionally adds cashtag chatter (tweet scraper) and Telegram channel
traffic. Every record keeps the actor id it came from so a run is reproducible.

Run it live with:

    set APIFY_TOKEN=...        (PowerShell: $env:APIFY_TOKEN="...")
    python src/run.py --live --days 10
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from .. import config

RUNS_URL = f"{config.APIFY_BASE}/acts/{{actor}}/runs"
DATASET_URL = f"{config.APIFY_BASE}/datasets/{{dataset}}/items"


def _post(url: str, payload: dict, token: str, timeout: int = 320) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _get(url: str, token: str) -> list[dict]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def run_actor(actor_id: str, payload: dict, token: str | None = None, wait_seconds: int = 300) -> list[dict]:
    """Start an actor with waitForFinish and return its dataset items."""
    token = token or config.apify_token()
    quoted = urllib.parse.quote(actor_id, safe="")
    url = RUNS_URL.format(actor=quoted) + f"?token={urllib.parse.quote(token)}&waitForFinish={wait_seconds}"
    started = _post(url, payload, token)
    run = started.get("data", {})
    if run.get("status") not in {"SUCCEEDED", "RUNNING", "READY"}:
        raise RuntimeError(f"actor {actor_id} did not start: {run.get('status')} {run.get('statusMessage')}")

    run_id = run["id"]
    deadline = time.time() + wait_seconds
    while run.get("status") not in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"} and time.time() < deadline:
        time.sleep(5)
        status_url = f"{config.APIFY_BASE}/actor-runs/{run_id}?token={urllib.parse.quote(token)}"
        run = _get_json(status_url, token).get("data", {})
    if run.get("status") != "SUCCEEDED":
        raise RuntimeError(f"actor {actor_id} finished as {run.get('status')}")

    dataset = run["defaultDatasetId"]
    items_url = DATASET_URL.format(dataset=dataset) + f"?token={urllib.parse.quote(token)}&clean=true&limit=5000"
    return _get(items_url, token)


def _get_json(url: str, token: str) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


# --- record shaping -------------------------------------------------------

_TIMESTAMP_HINTS = ("time", "date", "created", "published", "timestamp")


def _first(item: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _as_utc(value) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M", "%b %d, %Y %H:%M"):
            try:
                parsed = datetime.strptime(str(value), fmt)
                break
            except ValueError:
                continue
        else:
            parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def normalise_web_item(item: dict, asset: dict, exchange: str, source: str) -> list[dict]:
    """A crawled forum page yields one long text blob; split it into post-sized chunks so
    each chunk gets its own sentiment score and timestamp estimate."""
    text = _first(item, ("text", "markdown", "content", "body")) or ""
    url = _first(item, ("url", "loadedUrl", "link")) or ""
    published = _first(item, tuple(h for h in _TIMESTAMP_HINTS if h in item)) or item.get("crawledAt")
    stamp = _as_utc(published)

    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if len(c.strip()) >= 25][:60]
    posts = []
    for index, chunk in enumerate(chunks):
        posts.append({
            "exchange": exchange,
            "asset": asset["ticker"],
            "source": source,
            "ts": stamp,
            "text": chunk[:2000],
            "author": f"{source}:{urllib.parse.urlparse(url).path or 'page'}:{index}",
            "likes": 0,
            "reposts": 0,
            "views": 0,
            "followers": 0,
            "url": url,
        })
    return posts


def normalise_tweet(item: dict, asset: dict, exchange: str) -> dict:
    return {
        "exchange": exchange,
        "asset": asset["ticker"],
        "source": "x",
        "ts": _as_utc(_first(item, ("createdAt", "created_at", "timestamp", "date"))),
        "text": (_first(item, ("fullText", "text", "tweet")) or "")[:2000],
        "author": str(_first(item, ("author", "userName", "username")) or "x"),
        "likes": int(_first(item, ("likeCount", "favorite_count", "likes")) or 0),
        "reposts": int(_first(item, ("retweetCount", "retweet_count", "reposts")) or 0),
        "views": int(_first(item, ("viewCount", "views")) or 0),
        "followers": int(_first(item, ("followers", "followersCount")) or 0),
        "url": _first(item, ("url", "twitterUrl")) or "",
    }


def normalise_telegram(item: dict, asset: dict, exchange: str) -> dict:
    return {
        "exchange": exchange,
        "asset": asset["ticker"],
        "source": "telegram",
        "ts": _as_utc(_first(item, ("date", "timestamp", "createdAt"))),
        "text": (_first(item, ("text", "message", "caption")) or "")[:2000],
        "author": str(_first(item, ("channel", "username")) or "telegram"),
        "likes": int(_first(item, ("likes", "reactions")) or 0),
        "reposts": int(_first(item, ("forwards", "shares")) or 0),
        "views": int(_first(item, ("views", "viewCount")) or 0),
        "followers": 0,
        "url": _first(item, ("url", "link")) or "",
    }


# --- payload builders -----------------------------------------------------

def forum_payload(spec: config.SourceSpec, asset: dict) -> dict:
    term = asset["terms"][0]
    target = spec.search_url.format(q=urllib.parse.quote(term))
    return {
        "startUrls": [{"url": target}],
        "maxCrawlPages": 4,
        "crawlerType": "playwright:adaptive",
        "maxCrawlDepth": 1,
        "saveMarkdown": True,
        "removeCookieWarnings": True,
    }


def tweet_payload(asset: dict, days: int) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return {
        "searchTerms": [" ".join(asset["terms"][:2])],
        "since": since,
        "maxItems": 400,
        "sort": "Latest",
        "tweetLanguage": "en",
    }


def telegram_payload(asset: dict, days: int) -> dict:
    return {
        "channelList": ["@hkstockpicks", "@krstockchat"],
        "searchKeywords": asset["terms"][:2],
        "dateFrom": (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d"),
        "maxItems": 300,
    }


def collect_posts(exchange: str, days: int, include_tweets: bool = False,
                  include_telegram: bool = False) -> list[dict]:
    token = config.apify_token()
    posts: list[dict] = []

    for asset in config.assets_for(exchange):
        for source in config.SOURCES[exchange]:
            spec = config.SOURCE_SPECS[source]
            items = run_actor(config.APIFY_ACTORS["web"], forum_payload(spec, asset), token)
            for item in items:
                posts.extend(normalise_web_item(item, asset, exchange, source))

        if include_tweets:
            items = run_actor(config.APIFY_ACTORS["tweets"], tweet_payload(asset, days), token)
            posts.extend(normalise_tweet(i, asset, exchange) for i in items)

        if include_telegram:
            items = run_actor(config.APIFY_ACTORS["telegram"], telegram_payload(asset, days), token)
            posts.extend(normalise_telegram(i, asset, exchange) for i in items)

    return posts
