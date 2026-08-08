"""
ゴールド関連の最新ニュースを取得し、キーワードベースの簡易センチメント判定を行う。
本格的な精度が欲しい場合は、この関数の中身をFinBERT等のモデルや
Anthropic APIによるテキスト分類に差し替え可能（README参照）。
"""
import time
import yfinance as yf

BULLISH_WORDS = [
    "rally", "surge", "gain", "rise", "rises", "rising", "record high", "safe haven",
    "inflation hedge", "buy", "bullish", "upside", "climb", "soar", "strong demand",
    "rate cut", "dovish", "weaker dollar",
]
BEARISH_WORDS = [
    "fall", "falls", "falling", "drop", "plunge", "decline", "sell-off", "selloff",
    "bearish", "downside", "rate hike", "hawkish", "stronger dollar", "outflow",
    "profit-taking", "correction", "weak demand",
]

TICKER = "GC=F"
MAX_NEWS_AGE_HOURS = 24


def fetch_news(max_items: int = 15):
    t = yf.Ticker(TICKER)
    try:
        news = t.news or []
    except Exception as e:
        print(f"[news] ニュース取得に失敗: {e}")
        return []

    now = time.time()
    fresh = []
    for item in news[:max_items]:
        publish_time = item.get("providerPublishTime", now)
        age_hours = (now - publish_time) / 3600
        if age_hours <= MAX_NEWS_AGE_HOURS:
            fresh.append(item)
    return fresh


def score_headline(title: str) -> int:
    t = title.lower()
    score = 0
    for w in BULLISH_WORDS:
        if w in t:
            score += 1
    for w in BEARISH_WORDS:
        if w in t:
            score -= 1
    return score


def get_news_sentiment():
    """
    戻り値: dict {
      "score": float,  # -1.0(弱気) 〜 +1.0(強気) に正規化
      "headline_count": int,
      "top_headlines": list[str]
    }
    """
    items = fetch_news()
    if not items:
        return {"score": 0.0, "headline_count": 0, "top_headlines": []}

    scores = []
    headlines = []
    for item in items:
        title = item.get("title", "")
        if not title:
            continue
        scores.append(score_headline(title))
        headlines.append(title)

    if not scores:
        return {"score": 0.0, "headline_count": 0, "top_headlines": []}

    raw_avg = sum(scores) / len(scores)
    normalized = max(-1.0, min(1.0, raw_avg / 3.0))  # 目安として±3語相当で振り切れ

    return {
        "score": round(normalized, 2),
        "headline_count": len(headlines),
        "top_headlines": headlines[:5],
    }
