"""Regenerate the committed fixture snapshot.

Usage (needs network for the price bars, no API keys):

    python -m tools.make_fixtures --days 60

Writes:
    data/fixtures/bars.csv      real Yahoo Finance daily bars for all ten assets
    data/fixtures/social.jsonl  synthetic retail posts from the two venues per exchange
    data/fixtures/news.jsonl    synthetic Exa-style headlines

The social/news text is synthetic because the brief's venues need a paid Apify token and a
crawler run per asset. To keep the validation harness meaningful, the polarity of a day's
posts is tied to the *realised* next-day sign of that ticker 66% of the time. That gives the
model a weak, planted signal to recover and lets a reviewer see walk-forward accuracy move
above the baseline. It is not evidence of a real edge — the report says so in fixture mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

from src import config
from src.ingest.prices import collect

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "data" / "fixtures"

SIGNAL_ACCURACY = 0.66     # P(day's dominant tone matches the realised next-day sign)
TONE_FOLLOW = 0.80         # P(a single post follows the dominant tone)
NEUTRAL_SHARE = 0.08

TEMPLATES: dict[str, dict[str, list[str]]] = {
    "zh": {
        "bullish": [
            "{name} 今日放量突破，短線看好，目標再升一成 🚀",
            "業績超預期，買入等反彈，牛氣沖天",
            "大行上調評級，資金流入，強勢突破阻力",
            "抄底成功，反彈開始，成交急增 📈",
            "板塊輪動到 {name}，加倉等升",
            "低位吸納，利好陸續有來，睇好後市",
        ],
        "bearish": [
            "{name} 跌穿支持，成交萎縮，睇淡後市 📉",
            "利空消息未消化，短期再跌，減倉為妙",
            "高位派貨，小心暴跌，唔好亂接刀",
            "套牢盤太重，反彈無力，繼續尋底",
            "大行下調目標價，熊市未完",
            "資金流出，弱勢持續，止蝕先",
        ],
        "neutral": [
            "橫行待變，等消息再算",
            "今日無乜方向，觀望為主",
            "成交一般，區間上落",
            "{name} 等業績，暫時唔追",
        ],
    },
    "ko": {
        "bullish": [
            "{name} 급등, 외국인 매수세 강해 강세 지속 🚀",
            "실적 호재로 돌파, 추가 상승 기대",
            "저점 매수 기회, 반등 시작 📈",
            "기관 순매수 이어져 상승 탄력",
            "목표가 상향, 강세 유지 전망",
        ],
        "bearish": [
            "{name} 급락 주의, 매도 물량 폭탄 📉",
            "악재 겹쳐 약세 전환, 손절 타이밍",
            "외국인 매도 지속, 추가 하락 우려",
            "고점 대비 조정, 반등 실패",
            "실적 부진으로 투자심리 위축",
        ],
        "neutral": [
            "횡보 중, 방향성 없음",
            "관망세, 거래량 부족",
            "{name} 재료 대기 중",
        ],
    },
    "en": {
        "bullish": [
            "{name} breakout on volume, still bullish above resistance 🚀",
            "Earnings beat, upgrading my target, buying the dip",
            "Strong inflow today, rally looks sustainable",
        ],
        "bearish": [
            "{name} dumping through support, bearish short term 📉",
            "Guidance miss, downgrade risk, selling",
            "Distribution at the highs, expecting a selloff",
        ],
        "neutral": [
            "Choppy, waiting for a catalyst",
            "Volume drying up, range-bound",
        ],
    },
}


def _dominant_tone(next_ret: float | None, rng: random.Random) -> str:
    if next_ret is None or next_ret == 0:
        return rng.choice(["bullish", "bearish"])
    truth = "bullish" if next_ret > 0 else "bearish"
    if rng.random() < SIGNAL_ACCURACY:
        return truth
    return "bearish" if truth == "bullish" else "bullish"


def _post_tone(dominant: str, rng: random.Random) -> str:
    if rng.random() < NEUTRAL_SHARE:
        return "neutral"
    if rng.random() < TONE_FOLLOW:
        return dominant
    return "bullish" if dominant == "bearish" else "bearish"


def _asset_terms(ticker: str) -> dict:
    for assets in config.ASSETS.values():
        for asset in assets:
            if asset["ticker"] == ticker:
                return asset
    raise KeyError(ticker)


def _language(exchange: str, source: str, rng: random.Random) -> str:
    english_chance = {"blind": 0.35, "aastocks": 0.25}.get(source, 0.12)
    return "en" if rng.random() < english_chance else ("ko" if exchange == "KRX" else "zh")


def _engagement(rng: random.Random, source: str) -> dict:
    base = 40 if source in {"naver", "lihkg"} else 8
    likes = int(rng.expovariate(1 / base))
    reposts = int(likes * rng.uniform(0, 0.4))
    views = int(likes * rng.uniform(3, 25) + rng.randint(0, 400))
    followers = int(rng.uniform(25, 4000))
    return {"likes": likes, "reposts": reposts, "views": views, "followers": followers}


def generate_social(bars, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for ticker, group in bars.sort_values(["ticker", "date"]).groupby("ticker"):
        exchange = next(ex for ex, assets in config.ASSETS.items()
                        if ticker in {a["ticker"] for a in assets})
        sources = config.SOURCES[exchange]
        asset = _asset_terms(ticker)
        closes = group["close"].tolist()
        dates = [str(day) for day in group["date"].tolist()]

        for index, date in enumerate(dates):
            next_ret = None
            if index + 1 < len(closes) and closes[index]:
                next_ret = closes[index + 1] / closes[index] - 1
            dominant = _dominant_tone(next_ret, rng)
            n_posts = rng.randint(3, 9)
            for slot in range(n_posts):
                source = sources[slot % len(sources)]
                language = _language(exchange, source, rng)
                tone = _post_tone(dominant, rng)
                template = rng.choice(TEMPLATES[language][tone])
                text = template.format(name=asset["terms"][2] if language != "en" else asset["name"])
                engagement = _engagement(rng, source)
                rows.append({
                    "ticker": ticker,
                    "exchange": exchange,
                    "source": source,
                    "ts": f"{date}T{rng.randint(0, 9) + 1:02d}:{rng.randint(0, 59):02d}:00+00:00",
                    "author": f"{source}_{rng.randint(1, 900)}",
                    "text": text,
                    **engagement,
                    "url": f"https://example.invalid/{source}/{ticker}/{date}/{slot}",
                })
    return rows


def generate_news(bars, social: list[dict], seed: int = 11) -> list[dict]:
    """Headlines only on days that already have chatter, so the news feature is not sparse
    in a way that would leak which dates the model is being tested on."""
    rng = random.Random(seed)
    covered = {(row["ticker"], row["ts"][:10]) for row in social}
    rows: list[dict] = []
    for ticker, group in bars.sort_values(["ticker", "date"]).groupby("ticker"):
        asset = _asset_terms(ticker)
        closes = group["close"].tolist()
        dates = [str(day) for day in group["date"].tolist()]
        for index, date in enumerate(dates):
            if (ticker, date) not in covered or rng.random() > 0.55:
                continue
            next_ret = closes[index + 1] / closes[index] - 1 if index + 1 < len(closes) else None
            tone = _dominant_tone(next_ret, rng)
            if rng.random() < 0.2:
                tone = "neutral"
            text = rng.choice(TEMPLATES["en"][tone]).format(name=asset["name"])
            rows.append({
                "asset": ticker,
                "title": text.split(",")[0],
                "text": text,
                "published": f"{date}T02:15:00+00:00",
                "url": f"https://example.invalid/news/{ticker}/{date}",
                "source": "exa",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=60, help="trading bars per asset")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tickers = [asset["ticker"] for assets in config.ASSETS.values() for asset in assets]
    print(f"fetching {len(tickers)} x {args.days} bars from Yahoo Finance ...")
    bars = collect(tickers, days=args.days, pause=0.5)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    with (FIXTURE_DIR / "bars.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "ticker", "open", "high", "low", "close", "volume"])
        for row in bars.itertuples(index=False):
            writer.writerow([
                row.date, row.ticker, f"{row.open:.4f}", f"{row.high:.4f}",
                f"{row.low:.4f}", f"{row.close:.4f}", int(row.volume or 0),
            ])

    social = generate_social(bars, seed=args.seed)
    news = generate_news(bars, social, seed=args.seed + 4)

    with (FIXTURE_DIR / "social.jsonl").open("w", encoding="utf-8") as handle:
        for row in social:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (FIXTURE_DIR / "news.jsonl").open("w", encoding="utf-8") as handle:
        for row in news:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    span = f"{bars['date'].min()} -> {bars['date'].max()}"
    print(f"bars.csv     {len(bars):5d} rows  ({span})")
    print(f"social.jsonl {len(social):5d} posts")
    print(f"news.jsonl   {len(news):5d} headlines")
    print(f"written to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
