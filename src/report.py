"""Static HTML report: one ranked table per exchange plus the n+1 predictions.

Charts are hand-written inline SVG so the file opens from disk with no network, no CDN and
no JavaScript — it can be attached to an email or committed to a repo as-is.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

from . import config

CSS = """
:root { --ink:#12202e; --muted:#61707f; --line:#e2e8ee; --up:#0f7b52; --down:#b3261e; --flat:#6b7280; }
* { box-sizing: border-box; }
body { margin:0; padding:32px 28px 64px; font:14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
       color:var(--ink); background:#fbfcfd; }
h1 { font-size:22px; margin:0 0 4px; letter-spacing:-0.2px; }
h2 { font-size:15px; margin:34px 0 10px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); }
.meta { color:var(--muted); font-size:12.5px; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin-top:18px; }
.card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:150px; }
.card .k { font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); }
.card .v { font-size:20px; font-weight:600; margin-top:2px; }
table { width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line);
        border-radius:10px; overflow:hidden; font-variant-numeric:tabular-nums; }
th, td { padding:9px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th { font-size:11px; text-transform:uppercase; letter-spacing:0.05em; color:var(--muted);
     background:#f4f7f9; font-weight:600; }
th:first-child, td:first-child { text-align:left; }
tr:last-child td { border-bottom:none; }
.tick { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; }
.name { color:var(--muted); font-size:12px; }
.up { color:var(--up); } .down { color:var(--down); } .flat { color:var(--flat); }
.pill { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11.5px;
        font-weight:600; border:1px solid currentColor; }
.bar { height:6px; background:#eef2f6; border-radius:3px; overflow:hidden; width:72px;
       display:inline-block; vertical-align:middle; margin-right:6px; }
.bar > span { display:block; height:100%; }
.note { background:#fff; border:1px solid var(--line); border-left:3px solid #c8a44b;
        border-radius:8px; padding:12px 14px; font-size:12.5px; color:#4b5563; margin-top:14px; }
footer { margin-top:36px; color:var(--muted); font-size:11.5px; }
"""


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _direction(score: float) -> tuple[str, str]:
    if score > 0.15:
        return "bullish", "up"
    if score < -0.15:
        return "bearish", "down"
    return "neutral", "flat"


def _position(prob_up: float) -> tuple[str, str]:
    if prob_up >= 0.55:
        return "LONG", "up"
    if prob_up <= 0.45:
        return "SHORT", "down"
    return "FLAT", "flat"


def _sparkline(closes: list[float], width: int = 84, height: int = 20) -> str:
    if len(closes) < 2:
        return ""
    low, high = min(closes), max(closes)
    span = (high - low) or 1.0
    step = width / (len(closes) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - (c - low) / span * (height - 3) - 1.5:.1f}"
        for i, c in enumerate(closes)
    )
    rising = closes[-1] >= closes[0]
    color = "#0f7b52" if rising else "#b3261e"
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{points}"/></svg>')


def _bar(value: float, color: str) -> str:
    width = max(2, min(100, abs(value) * 100))
    return f'<span class="bar"><span style="width:{width:.0f}%;background:{color}"></span></span>'


def build(panel: pd.DataFrame, predictions: pd.DataFrame, overall: dict, per_exchange: dict,
          mode: str, days: int) -> str:
    name_of = {a["ticker"]: a["name"] for assets in config.ASSETS.values() for a in assets}
    exchange_of = {a["ticker"]: ex for ex, assets in config.ASSETS.items() for a in assets}

    latest = panel.sort_values("date").groupby("ticker", as_index=False).tail(1)
    history = {ticker: group["close"].tolist()[-10:] for ticker, group in panel.groupby("ticker")}
    pred_index = predictions.set_index("ticker") if not predictions.empty else pd.DataFrame()

    rows_by_exchange: dict[str, list[str]] = {}
    for exchange in config.ASSETS:
        tickers = latest[latest["ticker"].map(exchange_of) == exchange].copy()
        tickers = tickers.sort_values("n_posts", ascending=False)

        body = []
        for _, row in tickers.iterrows():
            ticker = row["ticker"]
            sentiment, sentiment_class = _direction(float(row["sent_weighted"]))
            prob_up = float(pred_index.loc[ticker, "prob_up"]) if ticker in pred_index.index else float("nan")
            expected = float(pred_index.loc[ticker, "expected_ret"]) if ticker in pred_index.index else float("nan")
            position, position_class = _position(prob_up) if prob_up == prob_up else ("FLAT", "flat")
            delta_class = "up" if row["ret_1d"] > 0 else "down" if row["ret_1d"] < 0 else "flat"
            body.append(
                "<tr>"
                f'<td><span class="tick">{html.escape(ticker)}</span><br><span class="name">{html.escape(name_of.get(ticker, ""))}</span></td>'
                f'<td>{_sparkline(history.get(ticker, []))}</td>'
                f"<td>{row['close']:,.2f}</td>"
                f'<td class="{delta_class}">{_pct(float(row["ret_1d"]))}</td>'
                f"<td>{int(row['n_posts'])}</td>"
                f"<td>{_bar(float(row['sent_weighted']), '#0f7b52' if row['sent_weighted'] >= 0 else '#b3261e')}"
                f"{float(row['sent_weighted']):+.2f}</td>"
                f'<td><span class="pill {sentiment_class}">{sentiment}</span></td>'
                f"<td>{row['support']:,.2f}</td><td>{row['resistance']:,.2f}</td>"
                f"<td>{prob_up * 100:.0f}%</td>"
                f'<td class="{position_class}">{_pct(expected) if expected == expected else "n/a"}</td>'
                f'<td><span class="pill {position_class}">{position}</span></td>'
                "</tr>"
            )
        rows_by_exchange[exchange] = body

    tables = []
    for exchange, body in rows_by_exchange.items():
        stats = per_exchange.get(exchange, {})
        summary = (f"walk-forward accuracy {stats.get('accuracy', float('nan')):.3f} · "
                   f"AUC {stats.get('auc', float('nan')):.3f} · n={stats.get('n', 0)}") if stats else "no out-of-sample rows"
        tables.append(
            f"<h2>{exchange} — ranked by social mentions <span class='meta'>· {html.escape(summary)}</span></h2>"
            "<table><thead><tr>"
            "<th>Asset</th><th>10d</th><th>Close</th><th>1d</th><th>Mentions</th>"
            "<th>Sentiment</th><th>Direction</th><th>Support</th><th>Resistance</th>"
            "<th>P(up)</th><th>E[r]</th><th>Position</th>"
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
        )

    def card(label: str, value: str) -> str:
        return f'<div class="card"><div class="k">{label}</div><div class="v">{value}</div></div>'

    fixture_note = ""
    if mode != "live":
        fixture_note = (
            '<div class="note"><b>Fixture mode.</b> Social posts and news in this run come from the bundled '
            'fixtures, not from Apify/Exa. The fixtures contain a weak, deliberately injected signal so the '
            'validation harness has something to find; recover it and the pipeline works, but these numbers '
            'say nothing about real market edge. Re-run with <code>--live</code> and your own tokens for the '
            'real measurement. Price bars are real Yahoo Finance data either way.</div>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HK/KR social sentiment — n+1 price report</title>
<style>{CSS}</style></head><body>
<h1>Social sentiment → n+1 price, HKEX &amp; KRX</h1>
<div class="meta">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ·
window {days} days · mode {html.escape(mode)} · sources:
LIHKG + AAStocks (HKEX) · Naver 종목토론실 + Blind (KRX) · news via Exa</div>

<div class="cards">
  {card("Out-of-sample rows", str(overall.get("n", 0)))}
  {card("Directional accuracy", f"{overall.get('accuracy', float('nan')):.3f}")}
  {card("Balanced accuracy", f"{overall.get('balanced_accuracy', float('nan')):.3f}")}
  {card("AUC", f"{overall.get('auc', float('nan')):.3f}")}
  {card("Majority baseline", f"{overall.get('majority_baseline', float('nan')):.3f}")}
  {card("MAE model / naive", f"{overall.get('mae_model', float('nan')):.4f} / {overall.get('mae_naive', float('nan')):.4f}")}
</div>
{fixture_note}

{"".join(tables)}

<h2>How to read this</h2>
<div class="note">
<b>Sentiment</b> is the engagement-weighted mean polarity of the day's posts (−1 … +1).
<b>P(up)</b> is the walk-forward model's probability that the next close is higher; <b>E[r]</b> is its
predicted next-day return. Position is LONG above 0.55, SHORT below 0.45, FLAT otherwise — a deliberately
wide band, because a 52% edge is not a trade. Support/resistance are trailing 20-day lows/highs, never
recomputed after the fact. Every out-of-sample row was produced by a model that had not seen that day.
P(up) is not a monotone function of sentiment: the model also weighs price state (range position, gap,
volume), so a bullish day at resistance on falling volume can still score below 0.5, and the two columns
disagreeing is information, not a bug.
</div>

<footer>CrowdWisdomTrading / TZURONI assignment — Abhinav Shukla · pipeline: Apify → Exa → features → walk-forward GBM → this report.</footer>
</body></html>"""
