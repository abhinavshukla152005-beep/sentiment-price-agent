"""Turn raw posts, news and bars into one daily feature panel per asset.

Every feature at date D is computed only from information dated on or before D, and the
target is the D+1 close-to-close return. That ordering is what makes the walk-forward test
in `model.py` honest.
"""

from __future__ import annotations

import pandas as pd

from .sentiment import aggregate, score_text

SENTIMENT_COLUMNS = [
    "n_posts", "n_authors", "sent_mean", "sent_std", "sent_weighted",
    "bull_share", "bear_share", "engagement", "disagreement",
]


def _utc_date(value) -> str:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return ""
    return stamp.date().isoformat()


def social_daily(posts: list[dict], sources: list[str]) -> pd.DataFrame:
    """Collapse posts into (ticker, date) rows, keeping a per-source mention count."""
    if not posts:
        return pd.DataFrame(columns=["ticker", "date", *SENTIMENT_COLUMNS])

    frame = pd.DataFrame(posts)
    frame["date"] = frame["ts"].map(_utc_date)
    frame = frame[frame["date"] != ""]

    rows = []
    for (ticker, date), group in frame.groupby(["ticker", "date"]):
        stats = aggregate(group.to_dict("records"))
        stats["ticker"] = ticker
        stats["date"] = date
        for source in sources:
            stats[f"src_{source}"] = int((group["source"] == source).sum())
        rows.append(stats)
    return pd.DataFrame(rows)


def news_daily(news: list[dict]) -> pd.DataFrame:
    if not news:
        return pd.DataFrame(columns=["ticker", "date", "n_news", "news_sent"])

    frame = pd.DataFrame(news)
    frame["date"] = frame["published"].map(_utc_date)
    frame["score"] = (frame["title"].fillna("") + " " + frame["text"].fillna("").str.slice(0, 300)).map(
        lambda text: score_text(text).polarity
    )
    grouped = frame.groupby(["asset", "date"]).agg(n_news=("score", "size"), news_sent=("score", "mean"))
    return grouped.reset_index().rename(columns={"asset": "ticker"})


def price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Price-only features plus the D+1 target. Support/resistance are trailing levels so
    they can be quoted in the report without hindsight."""
    frame = bars.sort_values(["ticker", "date"]).copy()
    frame["date"] = frame["date"].astype(str)

    grouped = frame.groupby("ticker", group_keys=False)
    frame["ret_1d"] = grouped["close"].pct_change()
    frame["ret_5d"] = grouped["close"].pct_change(5)
    frame["vol_10d"] = grouped["ret_1d"].transform(lambda s: s.rolling(10, min_periods=4).std())
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"]
    frame["gap_pct"] = (frame["open"] - grouped["close"].shift(1)) / grouped["close"].shift(1)
    frame["volume_z"] = grouped["volume"].transform(
        lambda s: (s - s.rolling(10, min_periods=5).mean()) / s.rolling(10, min_periods=5).std()
    )
    frame["support"] = grouped["low"].transform(lambda s: s.rolling(20, min_periods=5).min())
    frame["resistance"] = grouped["high"].transform(lambda s: s.rolling(20, min_periods=5).max())
    frame["pos_in_range"] = (frame["close"] - frame["support"]) / (frame["resistance"] - frame["support"])

    frame["target_ret"] = grouped["close"].pct_change().shift(-1)
    frame["target_up"] = (frame["target_ret"] > 0).astype("float")
    frame.loc[frame["target_ret"].isna(), "target_up"] = float("nan")
    return frame


def build_panel(posts: list[dict], news: list[dict], bars: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    prices = price_features(bars)
    social = social_daily(posts, sources)
    news_daily_frame = news_daily(news)

    panel = prices.merge(social, on=["ticker", "date"], how="left")
    panel = panel.merge(news_daily_frame, on=["ticker", "date"], how="left")

    for column in SENTIMENT_COLUMNS:
        panel[column] = panel.get(column, 0)
        panel[column] = panel[column].fillna(0)
    for source in sources:
        column = f"src_{source}"
        panel[column] = panel.get(column, 0).fillna(0)
    panel["has_social"] = (panel["n_posts"] > 0).astype(int)
    panel["n_news"] = panel["n_news"].fillna(0)
    panel["news_sent"] = panel["news_sent"].fillna(0)
    panel["sent_x_news"] = panel["sent_weighted"] * panel["news_sent"]
    panel["attention"] = panel["n_posts"].clip(upper=200) * (1 + panel["sent_weighted"].abs())

    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def feature_columns(sources: list[str]) -> list[str]:
    return [
        "n_posts", "n_authors", "sent_mean", "sent_std", "sent_weighted", "bull_share",
        "bear_share", "engagement", "disagreement", "has_social", "n_news", "news_sent",
        "sent_x_news", "attention", "ret_1d", "ret_5d", "vol_10d", "range_pct", "gap_pct",
        "volume_z", "pos_in_range", *[f"src_{s}" for s in sources],
    ]
