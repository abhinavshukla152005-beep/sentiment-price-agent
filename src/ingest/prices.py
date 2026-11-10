"""Daily bars from Yahoo Finance's public chart endpoint (no key required)."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HEADERS = {"User-Agent": "Mozilla/5.0 (sentiment-agent research script)"}


def fetch_bars(ticker: str, days: int = 40) -> pd.DataFrame:
    """Return a UTC-dated frame with open/high/low/close/volume for the last `days`."""
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - days * 2 * 86400
    query = urllib.parse.urlencode(
        {"period1": start, "period2": now, "interval": "1d", "includePrePost": "false"}
    )
    url = f"{ENDPOINT.format(ticker=urllib.parse.quote(ticker))}?{query}"
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"no chart data for {ticker}: {payload.get('chart', {}).get('error')}")

    series = result[0]
    quotes = series["indicators"]["quote"][0]
    frame = pd.DataFrame({
        "ts": pd.to_datetime(series["timestamp"], unit="s", utc=True),
        "open": quotes["open"],
        "high": quotes["high"],
        "low": quotes["low"],
        "close": quotes["close"],
        "volume": quotes["volume"],
    })
    frame["date"] = frame["ts"].dt.date
    frame = frame.dropna(subset=["close"]).drop(columns=["ts"])
    return frame.tail(days).reset_index(drop=True)


def collect(tickers: list[str], days: int = 40, pause: float = 0.4) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        bars = fetch_bars(ticker, days=days)
        bars["ticker"] = ticker
        frames.append(bars)
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]
