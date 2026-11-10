import numpy as np
import pandas as pd

from src.features import build_panel
from src.model import MIN_TRAIN_DATES, metrics, predict_next_day, walk_forward

SOURCES = ["lihkg", "aastocks"]
BULL = ["bullish breakout on volume", "buying the dip, strong rally", "upgrade, profit growth"]
BEAR = ["bearish dump, selling", "crash risk, downgrade", "loss widens, weak guidance"]

_PANEL: pd.DataFrame | None = None
_RESULTS: pd.DataFrame | None = None


def synthetic_panel(signal: float = 0.9, days: int = 45, tickers: int = 8, seed: int = 5) -> pd.DataFrame:
    """Price walk plus posts whose tone tracks the next day's sign with probability `signal`."""
    rng = np.random.default_rng(seed)
    dates = [str(day) for day in pd.bdate_range("2026-01-05", periods=days).date]

    bar_rows, post_rows = [], []
    for index in range(tickers):
        ticker = f"T{index}"
        close = 50 * np.cumprod(1 + rng.normal(0, 0.015, days))
        for day, date in enumerate(dates):
            bar_rows.append({
                "date": date, "ticker": ticker, "open": close[day], "high": close[day] * 1.01,
                "low": close[day] * 0.99, "close": close[day],
                "volume": float(rng.integers(1_000, 5_000)),
            })
            if day + 1 >= days:
                continue
            up = close[day + 1] > close[day]
            follow = up if rng.random() < signal else not up
            for slot in range(6):
                text = BULL if follow else BEAR
                post_rows.append({
                    "ticker": ticker,
                    "source": SOURCES[slot % len(SOURCES)],
                    "ts": f"{date}T0{slot}:00:00+00:00",
                    "text": text[slot % len(text)],
                    "author": f"u{slot}",
                    "likes": int(rng.integers(0, 30)),
                })
    return build_panel(post_rows, [], pd.DataFrame(bar_rows), SOURCES)


def panel() -> pd.DataFrame:
    """The synthetic panel is deterministic but costs ~1s to build, so it is reused."""
    global _PANEL
    if _PANEL is None:
        _PANEL = synthetic_panel()
    return _PANEL


def results() -> pd.DataFrame:
    global _RESULTS
    if _RESULTS is None:
        _RESULTS = walk_forward(panel(), SOURCES)
    return _RESULTS


def test_walk_forward_recovers_a_planted_signal():
    stats = metrics(results())

    assert stats["n"] > 100
    assert stats["accuracy"] > stats["majority_baseline"]
    assert stats["balanced_accuracy"] > 0.6
    assert stats["auc"] > 0.6


def test_walk_forward_never_tests_on_training_dates():
    dates = sorted(results()["date"].unique())
    all_dates = sorted(panel()["date"].unique())

    # the final bar has no realised next-day return, so it can never be a test date
    assert dates == all_dates[MIN_TRAIN_DATES:-1]
    assert all_dates[-1] not in dates


def test_predict_next_day_covers_every_ticker():
    predictions = predict_next_day(panel(), SOURCES)

    assert set(predictions["ticker"]) == set(panel()["ticker"])
    assert predictions["prob_up"].between(0, 1).all()
    assert predictions["date"].astype(str).max() == panel()["date"].astype(str).max()


def test_metrics_on_empty_frame():
    assert metrics(pd.DataFrame(columns=["target_up"])) == {"n": 0}
