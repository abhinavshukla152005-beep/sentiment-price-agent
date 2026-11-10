"""Load the committed fixture snapshot.

Fixture mode exists so the reviewer can run the whole pipeline — features, walk-forward
validation, n+1 prediction, report — with no Apify token, no Exa key and no network. The
snapshot is:

    data/fixtures/bars.csv     real Yahoo Finance daily bars, frozen at generation time
    data/fixtures/social.jsonl  synthetic retail posts, two venues per exchange
    data/fixtures/news.jsonl    synthetic headlines

The posts are synthetic on purpose and their polarity is correlated with the *realised*
next-day sign about 62% of the time (see `tools/make_fixtures.py`). That is a deliberately
injected, weak signal: it proves the harness can recover an edge when one exists, and it
proves nothing whatsoever about the real market. Every report generated in this mode says
so. Live mode replaces both inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"


def _path(name: str, directory: Path | None) -> Path:
    return (directory or FIXTURE_DIR) / name


def load_bars(directory: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(_path("bars.csv", directory))
    frame["date"] = frame["date"].astype(str)
    return frame[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _load_jsonl(name: str, directory: Path | None) -> list[dict]:
    path = _path(name, directory)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_posts(directory: Path | None = None) -> list[dict]:
    return _load_jsonl("social.jsonl", directory)


def load_news(directory: Path | None = None) -> list[dict]:
    return _load_jsonl("news.jsonl", directory)


def available(directory: Path | None = None) -> bool:
    return all(_path(name, directory).exists() for name in ("bars.csv", "social.jsonl", "news.jsonl"))
