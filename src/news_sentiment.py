"""
Fetches recent news for a given ticker and scores sentiment with a simple keyword approach.
Works for any asset (gold, BTC, forex) since it just takes a ticker string.
Swap score_headline() for a proper NLP/FinBERT model later if you want more accuracy.
"""
import time
import yfinance as yf

BULLISH_WORDS = [
    "rally", "surge", "gain", "rise", "rises", "rising", "record high", "safe haven",
    "inflation hedge", "buy", "bullish", "upside", "climb", "soar", "strong demand",
    "rate cut", "dovish", "weaker dollar", "adoption", "inflow", "breakout", "etf approval",
]
BEARISH_WORDS = [
    "fall", "falls", "falling", "drop", "plunge", "decline", "sell-off", "selloff",
    "bearish", "downside", "rate hike", "hawkish", "stronger dollar", "outflow",
    "profit-taking", "correction", "weak demand", "hack", "ban", "crackdown", "regulation risk",
]

MAX_NEWS_AGE_HOURS = 24


def fetch_news(ticker, max_items=15):
    t = yf.Ticker(ticker)
    try:
        news = t.news or []
    except Exception as e:
        print("news: fetch failed for", ticker, e)
        return []

    now = time.time()
    fresh = []
    for item in news[:max_items]:
        publish_time = item.get("providerPublishTime", now)
        age_hours = (now - publish_time) / 3600
        if age_hours <= MAX_NEWS_AGE_HOURS:
            fresh.append(item)
    return fresh


def score_headline(title):
    t = title.lower()
    score = 0
    for w in BULLISH_WORDS:
        if w in t:
            score += 1
    for w in BEARISH_WORDS:
        if w in t:
            score -= 1
    return score


def get_news_sentiment(ticker):
    """
    Returns dict {
      "score": float,   # -1.0 (bearish) to +1.0 (bullish)
      "headline_count": int,
      "top_headlines": list[str]
    }
    """
    items = fetch_news(ticker)
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
    normalized = max(-1.0, min(1.0, raw_avg / 3.0))

    return {
        "score": round(normalized, 2),
        "headline_count": len(headlines),
        "top_headlines": headlines[:5],
    }
