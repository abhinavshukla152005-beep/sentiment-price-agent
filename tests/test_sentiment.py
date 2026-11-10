import math

from src.sentiment import aggregate, engagement_weight, score_text


def test_english_negation_flips_polarity():
    assert score_text("Tencent breakout, very bullish").polarity > 0.7
    assert score_text("Tencent not bullish, dump incoming").polarity < -0.5


def test_chinese_negation_inside_run():
    assert score_text("騰訊突破，看好").polarity > 0.7
    assert score_text("我不看好，會暴跌").polarity < -0.5


def test_korean_words_with_particles():
    assert score_text("삼성전자가 급등, 강세 지속").polarity > 0.7
    assert score_text("삼성전자 급락, 약세 전환").polarity < -0.7


def test_neutral_text_scores_zero():
    score = score_text("market opens at nine")
    assert score.polarity == 0.0
    assert score.hits == []


def test_polarity_is_bounded():
    for text in ["bullish " * 40, "crash " * 40, "🚀📈🔥" * 20]:
        assert -1.0 <= score_text(text).polarity <= 1.0


def test_engagement_weight_is_monotonic():
    weights = [engagement_weight(likes, reposts, views, followers)
               for likes, reposts, views, followers in
               [(0, 0, 0, 0), (5, 0, 100, 0), (50, 10, 5000, 1000), (500, 100, 50000, 10000)]]
    assert weights == sorted(weights)
    assert weights[0] == 1.0


def test_aggregate_counts_directions():
    posts = [
        {"text": "bullish breakout", "likes": 10, "author": "a"},
        {"text": "bullish rally", "likes": 5, "author": "b"},
        {"text": "bearish dump", "likes": 1, "author": "c"},
        {"text": "range bound", "likes": 0, "author": "d"},
    ]
    stats = aggregate(posts)
    assert stats["n_posts"] == 4
    assert stats["n_authors"] == 4
    assert stats["bull_share"] == 0.5
    assert stats["bear_share"] == 0.25
    assert stats["sent_weighted"] > stats["sent_mean"] > 0
    assert math.isclose(stats["disagreement"], 0.25)


def test_aggregate_empty_day():
    stats = aggregate([])
    assert stats["n_posts"] == 0
    assert stats["sent_weighted"] == 0.0
