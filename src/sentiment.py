"""Lexicon sentiment scorer tuned for retail trading chatter.

A transformer would score better, but this ships with no model download, runs offline and
is fully inspectable — every number in the report can be traced back to matched tokens,
which is what an auditor of a trading signal actually needs. The scorer handles:

* negation ("not bullish", "冇升", "안 좋아")
* intensifiers ("very bullish", "超級牛")
* emoji and cashtag noise
* Chinese and Korean bullish/bearish vocabulary, because HK and KR retail posts are
  frequently written in the local language or a mix of all three.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

LEXICON: dict[str, float] = {
    # English / financial shorthand
    "bullish": 2.0, "bearish": -2.0, "moon": 1.5, "mooning": 1.5, "dump": -1.6,
    "dumping": -1.6, "pump": 1.2, "pumping": 1.2, "rally": 1.5, "selloff": -1.6,
    "crash": -2.0, "breakout": 1.4, "buy": 1.0, "buying": 1.0, "sell": -1.0,
    "selling": -1.0, "long": 1.0, "short": -1.0, "strong": 1.2, "weak": -1.2,
    "beat": 1.2, "miss": -1.2, "upgrade": 1.3, "downgrade": -1.3, "profit": 1.0,
    "loss": -1.0, "growth": 1.1, "risk": -0.8, "lawsuit": -1.2, "recall": -1.1,
    "guidance": 0.2, "dividend": 0.8, "buyback": 1.2, "oversold": 1.1, "overbought": -0.9,
    "support": 0.4, "resistance": -0.3, "green": 0.8, "red": -0.8, "rip": -1.5,
    # Chinese (traditional + simplified)
    "升": 1.2, "涨": 1.2, "漲": 1.2, "牛": 1.4, "牛市": 1.8, "看好": 1.5, "买入": 1.3,
    "買入": 1.3, "加仓": 1.2, "加倉": 1.2, "突破": 1.3, "反弹": 1.1, "反彈": 1.1,
    "跌": -1.2, "熊": -1.4, "熊市": -1.8, "看淡": -1.4, "卖出": -1.3, "賣出": -1.3,
    "减仓": -1.1, "減倉": -1.1, "暴跌": -2.0, "崩盘": -2.0, "崩盤": -2.0, "利空": -1.4,
    "利好": 1.4, "抄底": 0.9, "套牢": -1.2, "止蚀": -0.9, "止蝕": -0.9,
    # Korean
    "상승": 1.2, "급등": 1.8, "매수": 1.3, "강세": 1.4, "호재": 1.4, "돌파": 1.3,
    "반등": 1.1, "하락": -1.2, "급락": -1.8, "매도": -1.3, "약세": -1.4, "악재": -1.4,
    "손절": -1.2, "물렸": -1.2, "상한가": 2.0, "하한가": -2.0, "존버": 0.6,
    # Emoji that carry directional meaning in this context
    "🚀": 1.5, "📈": 1.3, "📉": -1.3, "💎": 0.8, "🐻": -1.2, "🐂": 1.2, "🔥": 1.0,
}

NEGATORS = (
    "not", "no", "never", "none", "isn't", "aren't", "don't", "doesn't", "won't",
    "cant", "can't", "cannot", "hardly", "without",
    "唔", "不", "沒", "沒冇", "冇", "無", "别", "別", "未",
    "안", "못", "없", "아니",
)

INTENSIFIERS = {
    "very": 1.4, "really": 1.3, "so": 1.2, "super": 1.5, "extremely": 1.6,
    "huge": 1.5, "massive": 1.5, "slightly": 0.6, "a bit": 0.6,
    "超級": 1.5, "非常": 1.4, "勁": 1.3, "완전": 1.4, "매우": 1.3,
}

_TOKEN_RE = re.compile(r"[A-Za-z']+|\$[A-Za-z0-9.]+|[\u4e00-\u9fff]+|[\uac00-\ud7af]+|[\U0001F300-\U0001FAFF]", re.UNICODE)


@dataclass
class Score:
    polarity: float          # -1 .. 1
    raw: float               # sum of matched token weights after negation/intensity
    hits: list[str]
    negated: int

    @property
    def direction(self) -> str:
        if self.polarity > 0.15:
            return "bullish"
        if self.polarity < -0.15:
            return "bearish"
        return "neutral"


CJK_NEGATORS = ("不", "沒", "冇", "無", "唔", "別", "未", "非")


def _cjk_word_hits(token: str) -> list[tuple[str, float]]:
    """Korean keeps spaces, so a multi-syllable lexicon word can match inside a token
    that carries a particle (삼성전자가 -> 삼성전자 ... )."""
    return [(word, weight) for word, weight in LEXICON.items()
            if len(word) > 1 and word != token and "\uac00" <= word[0] <= "\ud7af" and word in token]



def _score_cjk_run(run: str) -> list[tuple[str, float]]:
    """Chinese runs carry no spaces, so scan left to right taking the longest lexicon
    match and looking back for a negator inside the same run."""
    out: list[tuple[str, float]] = []
    i, n = 0, len(run)
    while i < n:
        for length in (3, 2, 1):
            fragment = run[i:i + length]
            if len(fragment) != length or fragment not in LEXICON:
                continue
            weight = LEXICON[fragment]
            before = run[max(0, i - 2):i]
            if any(before.endswith(neg) for neg in CJK_NEGATORS):
                weight = -weight * 0.8
            out.append((fragment, weight))
            i += length
            break
        else:
            i += 1
    return out


def score_text(text: str) -> Score:
    if not text:
        return Score(0.0, 0.0, [], 0)

    tokens = _TOKEN_RE.findall(text.lower())
    raw = 0.0
    magnitude = 0.0
    hits: list[str] = []
    negated = 0
    negation_window = 0
    intensity = 1.0

    for token in tokens:
        if token in NEGATORS or any(tok in NEGATORS for tok in [token]):
            negation_window = 2
            continue
        if token in INTENSIFIERS:
            intensity = INTENSIFIERS[token]
            continue

        matches: list[tuple[str, float]] = []
        weight = LEXICON.get(token)
        if weight is not None:
            matches.append((token, weight))
        if token and any("\u4e00" <= ch <= "\u9fff" for ch in token):
            matches.extend(_score_cjk_run(token))
        elif token and any("\uac00" <= ch <= "\ud7af" for ch in token):
            matches.extend(_cjk_word_hits(token))

        for word, weight in matches:
            applied = weight * intensity
            if negation_window > 0:
                applied = -applied * 0.8
                negated += 1
                negation_window = 0      # "not bullish dump" negates the sentiment word only
            raw += applied
            magnitude += abs(applied)
            hits.append(word)

        intensity = 1.0
        if negation_window > 0:
            negation_window -= 1

    if magnitude == 0:
        return Score(0.0, 0.0, hits, negated)
    polarity = math.tanh(raw / max(magnitude, 1.0))
    return Score(round(polarity, 4), round(raw, 4), hits, negated)


def engagement_weight(likes: int | float, reposts: int | float, views: int | float, followers: int | float) -> float:
    """Posts with reach count more, but sub-linearly — a single viral post should not
    drown out forty ordinary ones."""
    return 1.0 + math.log1p(max(0.0, likes) + 2 * max(0.0, reposts)) + 0.5 * math.log1p(max(0.0, views)) \
        + 0.3 * math.log1p(max(0.0, followers))


def aggregate(posts: list[dict]) -> dict:
    """Summarise a day of posts for one asset. `posts` are dicts with `text`,
    `likes`, `reposts`, `views`, `followers`."""
    if not posts:
        return {"n_posts": 0, "n_authors": 0, "sent_mean": 0.0, "sent_std": 0.0,
                "sent_weighted": 0.0, "bull_share": 0.0, "bear_share": 0.0,
                "engagement": 0.0, "disagreement": 0.0}

    scores, weights, directions = [], [], []
    authors = set()
    total_engagement = 0.0
    for post in posts:
        scored = score_text(post.get("text", ""))
        weight = engagement_weight(
            post.get("likes", 0) or 0,
            post.get("reposts", 0) or 0,
            post.get("views", 0) or 0,
            post.get("followers", 0) or 0,
        )
        scores.append(scored.polarity)
        weights.append(weight)
        directions.append(scored.direction)
        authors.add(post.get("author") or post.get("author_id") or "")
        total_engagement += weight

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    weighted = sum(s * w for s, w in zip(scores, weights)) / sum(weights) if weights else 0.0
    bulls = directions.count("bullish")
    bears = directions.count("bearish")

    return {
        "n_posts": len(posts),
        "n_authors": len(authors),
        "sent_mean": round(mean, 4),
        "sent_std": round(math.sqrt(variance), 4),
        "sent_weighted": round(weighted, 4),
        "bull_share": round(bulls / len(posts), 4),
        "bear_share": round(bears / len(posts), 4),
        "engagement": round(total_engagement, 2),
        "disagreement": round(abs(bulls - bears) / len(posts), 4),
    }
