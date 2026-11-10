import numpy as np
import pandas as pd

from src.features import build_panel, feature_columns, price_features


def bars(days: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [str(day) for day in pd.bdate_range("2026-01-05", periods=days).date]
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, days))
    return pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "open": close * (1 + rng.normal(0, 0.002, days)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1_000, 9_000, days).astype(float),
    })


def test_target_is_the_next_close_to_close_return():
    frame = price_features(bars())
    expected = frame["close"].shift(-1) / frame["close"] - 1
    assert np.allclose(frame["target_ret"][:-1], expected[:-1])
    assert np.isnan(frame["target_ret"].iloc[-1])


def test_features_do_not_look_ahead():
    base = bars()
    frame = price_features(base)
    tweaked = base.copy()
    tweaked.loc[tweaked.index[-1], "close"] *= 1.2
    moved = price_features(tweaked)

    features = [c for c in feature_columns(["lihkg"]) if c in frame]
    pd.testing.assert_frame_equal(
        frame.iloc[:-1][features].reset_index(drop=True),
        moved.iloc[:-1][features].reset_index(drop=True),
    )


def test_support_and_resistance_are_trailing():
    frame = price_features(bars()).dropna(subset=["support", "resistance", "pos_in_range"])
    assert (frame["support"] <= frame["close"]).all()
    assert (frame["resistance"] >= frame["close"]).all()
    assert frame["pos_in_range"].between(0, 1).all()


def test_panel_merges_social_on_the_same_day():
    price_frame = price_features(bars(days=3))
    posts = [{
        "ticker": "TEST", "source": "lihkg", "ts": "2026-01-06T03:00:00+00:00",
        "text": "bullish breakout", "author": "a", "likes": 3,
    }]
    panel = build_panel(posts, [], bars(days=3), ["lihkg"])

    assert not panel.duplicated(["ticker", "date"]).any()
    row = panel[panel["date"] == "2026-01-06"].iloc[0]
    assert row["n_posts"] == 1
    assert row["has_social"] == 1
    assert row["src_lihkg"] == 1
    other = panel[panel["date"] == "2026-01-05"].iloc[0]
    assert other["n_posts"] == 0
    assert other["has_social"] == 0


def test_feature_columns_include_each_source():
    columns = feature_columns(["lihkg", "aastocks", "naver", "blind"])
    for source in ["lihkg", "aastocks", "naver", "blind"]:
        assert f"src_{source}" in columns
