"""End-to-end run: ingest -> panel -> walk-forward -> n+1 predictions -> HTML report.

    python -m src.run                    # fixture snapshot, no keys, no network
    python -m src.run --mode live --days 10
    python -m src.run --mode live --days 10 --tweets --telegram

Outputs land in `reports/`: report.html (self-contained), predictions.csv, panel.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, model, report
from .features import build_panel
from .ingest import fixtures

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCES = [source for group in config.SOURCES.values() for source in group]
EXCHANGE_OF = {asset["ticker"]: exchange for exchange, assets in config.ASSETS.items() for asset in assets}
ALL_TICKERS = list(EXCHANGE_OF)


def _cutoff(bars: pd.DataFrame) -> str:
    return str(min(bars["date"]))


def _trim(rows: list[dict], bars: pd.DataFrame) -> list[dict]:
    """Keep only records inside the price window, so a crawl that returns older pages does
    not add features for dates the model never trains on."""
    start = _cutoff(bars)
    kept = []
    for row in rows:
        stamp = str(row.get("ts") or row.get("published") or "")
        if stamp[:10] >= start:
            kept.append(row)
    return kept


def load_fixture(days: int) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    bars = fixtures.load_bars()
    bars = bars[bars["date"] >= sorted(bars["date"].unique())[-days]].reset_index(drop=True)
    posts = _trim(fixtures.load_posts(), bars)
    news = _trim(fixtures.load_news(), bars)
    return bars, posts, news


def load_live(days: int, tweets: bool, telegram: bool) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    from .ingest import apify, exa, prices

    try:
        config.apify_token()
        config.exa_key()
    except RuntimeError as error:
        sys.exit(f"{error}. Live mode needs APIFY_TOKEN and EXA_API_KEY; "
                 f"run without --mode live for the offline fixture snapshot.")

    bars = prices.collect(ALL_TICKERS, days=days)
    posts: list[dict] = []
    for exchange in config.ASSETS:
        print(f"  apify: {exchange} ({', '.join(config.SOURCES[exchange])})")
        posts.extend(apify.collect_posts(exchange, days, include_tweets=tweets, include_telegram=telegram))
    news: list[dict] = []
    for exchange in config.ASSETS:
        print(f"  exa: news for {exchange}")
        news.extend(exa.collect_news(config.assets_for(exchange), days=days))
    return bars, _trim(posts, bars), _trim(news, bars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Social sentiment -> n+1 price pipeline")
    parser.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    parser.add_argument("--days", type=int, default=40, help="price bars / social window")
    parser.add_argument("--tweets", action="store_true", help="also pull cashtag chatter (live)")
    parser.add_argument("--telegram", action="store_true", help="also pull Telegram channels (live)")
    parser.add_argument("--out", default=str(REPORTS / "report.html"))
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)

    print(f"mode={args.mode} days={args.days}")
    if args.mode == "live":
        bars, posts, news = load_live(args.days, args.tweets, args.telegram)
    else:
        bars, posts, news = load_fixture(args.days)

    print(f"  bars={len(bars)} posts={len(posts)} news={len(news)}")

    panel = build_panel(posts, news, bars, SOURCES)
    panel.to_csv(REPORTS / "panel.csv", index=False)

    results = model.walk_forward(panel, SOURCES)
    overall = model.metrics(results)
    per_exchange = model.by_exchange(results, EXCHANGE_OF)
    predictions = model.predict_next_day(panel, SOURCES)

    out = Path(args.out)
    out.write_text(report.build(panel, predictions, overall, per_exchange, args.mode, args.days),
                   encoding="utf-8")
    predictions.sort_values("prob_up", ascending=False).to_csv(REPORTS / "predictions.csv", index=False)

    print(f"  walk-forward rows={overall.get('n', 0)} dates={overall.get('dates', 0)}")
    print(f"  accuracy={overall.get('accuracy')} balanced={overall.get('balanced_accuracy')} "
          f"auc={overall.get('auc')} majority={overall.get('majority_baseline')}")
    print(f"  mae model={overall.get('mae_model')} naive={overall.get('mae_naive')} "
          f"ret_corr={overall.get('ret_corr')}")
    if not predictions.empty:
        print("  next-day calls:")
        for row in predictions.sort_values("prob_up", ascending=False).itertuples(index=False):
            print(f"    {row.ticker:<11} {row.date}  P(up)={row.prob_up:.3f}  E[r]={row.expected_ret:+.4f}")
    print(f"  report -> {out}")


if __name__ == "__main__":
    main()
