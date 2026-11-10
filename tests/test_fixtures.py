from src import config
from src.ingest import fixtures


def test_fixture_snapshot_is_loadable():
    assert fixtures.available()
    bars = fixtures.load_bars()
    posts = fixtures.load_posts()
    news = fixtures.load_news()

    assert list(bars.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert len(bars) > 300
    assert len(posts) > 1000
    assert len(news) > 0


def test_every_fixture_post_maps_to_a_known_asset():
    tickers = {asset["ticker"] for assets in config.ASSETS.values() for asset in assets}
    posts = fixtures.load_posts()
    assert {post["ticker"] for post in posts} <= tickers
    assert all(post["source"] in config.SOURCE_SPECS for post in posts)


def test_two_sources_per_exchange_and_five_assets_each():
    for exchange, assets in config.ASSETS.items():
        assert len(assets) == 5
        assert len(config.SOURCES[exchange]) == 2


def test_fixture_dates_within_bar_window():
    bars = fixtures.load_bars()
    window = (bars["date"].min(), bars["date"].max())
    for post in fixtures.load_posts():
        assert window[0] <= post["ts"][:10] <= window[1]
