"""Static configuration: assets, per-exchange social sources, Apify actors, Exa settings.

Everything the pipeline needs to talk to the outside world is described here so the
ingestion layer stays generic: a source is a URL template plus a parser hint, an actor
is an id, and an asset is a ticker with the keywords retail traders actually type.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

HISTORY_DAYS = 10
UTC = "UTC"

# --- Exchanges and assets -------------------------------------------------
# Five liquid names per exchange. `terms` mixes the cashtag, the plain ticker and the
# local-language name, because retail posts rarely use one convention consistently.
ASSETS: dict[str, list[dict]] = {
    "HKEX": [
        {"ticker": "0700.HK", "name": "Tencent Holdings", "terms": ["$0700", "0700.HK", "騰訊", "tencent"]},
        {"ticker": "9988.HK", "name": "Alibaba Group", "terms": ["$9988", "9988.HK", "阿里巴巴", "alibaba"]},
        {"ticker": "1810.HK", "name": "Xiaomi", "terms": ["$1810", "1810.HK", "小米", "xiaomi"]},
        {"ticker": "0941.HK", "name": "China Mobile", "terms": ["$0941", "0941.HK", "中國移動", "china mobile"]},
        {"ticker": "0388.HK", "name": "Hong Kong Exchanges", "terms": ["$0388", "0388.HK", "港交所", "hkex"]},
    ],
    "KRX": [
        {"ticker": "005930.KS", "name": "Samsung Electronics", "terms": ["$005930", "005930", "삼성전자", "samsung"]},
        {"ticker": "000660.KS", "name": "SK Hynix", "terms": ["$000660", "000660", "하이닉스", "hynix"]},
        {"ticker": "035420.KS", "name": "NAVER", "terms": ["$035420", "035420", "네이버", "naver"]},
        {"ticker": "005380.KS", "name": "Hyundai Motor", "terms": ["$005380", "005380", "현대차", "hyundai"]},
        {"ticker": "035720.KS", "name": "Kakao", "terms": ["$035720", "035720", "카카오", "kakao"]},
    ],
}

# Two different retail sources per exchange, as the brief requires.
SOURCES: dict[str, list[str]] = {
    "HKEX": ["lihkg", "aastocks"],
    "KRX": ["naver", "blind"],
}


@dataclass(frozen=True)
class SourceSpec:
    """How to find a given ticker on a given retail venue."""

    label: str
    search_url: str          # {q} is replaced with the asset's primary term, url-quoted
    entry_selector: str      # CSS selector for one post/comment block inside a page
    author_selector: str = ""
    engagement_selector: str = ""
    note: str = ""


SOURCE_SPECS: dict[str, SourceSpec] = {
    "lihkg": SourceSpec(
        label="LIHKG",
        search_url="https://lihkg.com/search?q={q}",
        entry_selector=".thread_item, .reply_content",
        author_selector=".user_info, .thread_username",
        engagement_selector=".like_count, .reply_count",
        note="Hong Kong retail forum; stock channel + free chat.",
    ),
    "aastocks": SourceSpec(
        label="AAStocks",
        search_url="https://www.aastocks.com/en/stocks/analysis/comment.aspx?symbol={q}",
        entry_selector=".newscontent, .comment_content",
        author_selector=".comment_author",
        engagement_selector="",
        note="AAStocks quote pages carry user comments next to market data.",
    ),
    "naver": SourceSpec(
        label="Naver Finance 종목토론실",
        search_url="https://finance.naver.com/item/board.naver?code={q}",
        entry_selector="table.type2 tr td.title",
        author_selector="td.writer",
        engagement_selector="td.view",
        note="One discussion board per KRX listing; the densest retail sentiment feed in Korea.",
    ),
    "blind": SourceSpec(
        label="Blind",
        search_url="https://www.teamblind.com/kr/search?q={q}",
        entry_selector=".post_item, .article_item",
        author_selector=".author",
        engagement_selector=".like_count",
        note="Verified-employee app; company-level discussion.",
    ),
}

# --- Apify ----------------------------------------------------------------
# One generic crawler covers the four forum venues; the two social adapters are used when
# a run should also pull cashtag chatter and Telegram channel traffic.
APIFY_ACTORS: dict[str, str] = {
    "web": "apify/website-content-crawler",
    "tweets": "apidojo/tweet-scraper",
    "telegram": "apify/telegram-scraper",
}

APIFY_BASE = "https://api.apify.com/v2"
EXA_BASE = "https://api.exa.ai"
EXA_NEWS_RESULTS = 8


def apify_token() -> str:
    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("APIFY_TOKEN is not set (see .env.example)")
    return token


def exa_key() -> str:
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("EXA_API_KEY is not set (see .env.example)")
    return key


def assets_for(exchange: str) -> list[dict]:
    return ASSETS[exchange]


@dataclass
class RunConfig:
    days: int = HISTORY_DAYS
    exchanges: list[str] = field(default_factory=lambda: list(ASSETS))
    include_tweets: bool = False
    include_telegram: bool = False
