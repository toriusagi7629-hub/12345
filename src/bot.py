"""
Main script run on a schedule. Loops over every asset in assets_config.ASSETS.
For each asset:
1. Fetch latest 5-minute price data and detect a breakout
2. Confirm the breakout against 1-hour and 1-day trend (multi-timeframe filter)
3. Predict breakout success probability with that asset's trained AI model
4. Fetch news sentiment
5. Estimate a suggested lot size based on account risk and AI confidence
6. Notify Discord if all filters pass
7. Record the last alerted bar per asset in state.json to avoid duplicate notifications
"""
import os
import sys
import json
import joblib
import requests
import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS  # noqa: E402
from news_sentiment import get_news_sentiment  # noqa: E402
from assets_config import ASSETS  # noqa: E402

INTERVAL = "5m"
PERIOD = "5d"

H1_INTERVAL = "60m"
H1_PERIOD = "180d"
D1_INTERVAL = "1d"
D1_PERIOD = "2y"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state.json")

ML_PROB_THRESHOLD = 0.55
NEWS_VETO_THRESHOLD = -0.4
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5
TP2_ATR_MULT = 4.0

ACCOUNT_BALANCE_JPY = 10000.0
RISK_PERCENT = 0.02
USDJPY_RATE = 150.0
MIN_CONFIDENCE_SCALE = 0.3

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def fetch_ohlc(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_trend_only(ticker, interval, period):
    try:
        df = fetch_ohlc(ticker, interval, period)
    except Exception as e:
        print("bot: higher timeframe fetch failed", ticker, interval, e)
        return None
    if df.empty or len(df) < 60:
        return None
    feat = build_features(df)
    return feat.iloc[-1]["trend"]


def send_discord(payload_text, embed):
    if not DISCORD_WEBHOOK_URL:
        print("bot: DISCORD_WEBHOOK_URL not set, printing only")
        print(payload_text)
        return
    resp = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=15)
    if resp.status_code >= 300:
        print("bot: discord send failed", resp.status_code, resp.text)
    else:
        print("bot: discord notification sent")


def confidence_scale(ml_prob):
    if ml_prob is None:
        return 0.5
    raw = (ml_prob - 0.5) / 0.5
    if raw < MIN_CONFIDENCE_SCALE:
        raw = MIN_CONFIDENCE_SCALE
    if raw > 1.0:
        raw = 1.0
    return raw


def estimate_lot_size(sl_points, ml_prob, contract_size, quote_currency):
    scale = confidence_scale(ml_prob)
    risk_amount_jpy = ACCOUNT_BALANCE_JPY * RISK_PERCENT * scale
    if quote_currency == "JPY":
        jpy_loss_per_lot = sl_points * contract_size
    else:
        jpy_loss_per_lot = sl_points * contract_size * USDJPY_RATE
    if jpy_loss_per_lot <= 0:
        return 0.0, round(risk_amount_jpy, 0)
    lot = round(risk_amount_jpy / jpy_loss_per_lot, 4)
    return lot, round(risk_amount_jpy, 0)


def build_embed(asset_label, direction, price, sl, tp, tp2, ml_prob, trend, h1_trend, d1_trend,
                 news, lot_size, risk_amount_jpy, unit_label):
    is_long = direction == "LONG"
    color = 3066993 if is_long else 15158332
    arrow = "LONG" if is_long else "SHORT"
    ml_txt = str(ml_prob) if ml_prob is not None else "no model"

    sl_pts = round(abs(price - sl), 4)
    tp_pts = round(abs(tp - price), 4)
    tp2_pts = round(abs(tp2 - price), 4)

    headlines_txt = ""
    for h in news["top_headlines"][:3]:
        headlines_txt = headlines_txt + "- " + h + "\n"
    if headlines_txt == "":
        headlines_txt = "no news"

    sentiment_txt = str(news["score"]) + " (" + str(news["headline_count"]) + ")"

    return {
        "title": arrow + " AI breakout alert: " + asset_label,
        "color": color,
        "fields": [
            {"name": "price", "value": str(round(price, 4)), "inline": True},
            {"name": "AI probability", "value": ml_txt, "inline": True},
            {"name": "SL", "value": str(round(sl, 4)) + " (-" + str(sl_pts) + ")", "inline": True},
            {"name": "TP", "value": str(round(tp, 4)) + " (+" + str(tp_pts) + ")", "inline": True},
            {"name": "TP2 extended", "value": str(round(tp2, 4)) + " (+" + str(tp2_pts) + ")", "inline": True},
            {"name": "5m trend", "value": trend, "inline": True},
            {"name": "1h trend", "value": h1_trend, "inline": True},
            {"name": "1d trend", "value": d1_trend, "inline": True},
            {"name": "suggested lot", "value": str(lot_size) + " lot (" + unit_label + ")", "inline": True},
            {"name": "risk amount", "value": str(int(risk_amount_jpy)) + " JPY", "inline": True},
            {"name": "news sentiment", "value": sentiment_txt, "inline": True},
            {"name": "recent headlines", "value": headlines_txt, "inline": False},
        ],
    }


def process_asset(asset, state):
    key = asset["key"]
    ticker = asset["ticker"]
    asset_state = state.get(key, {"last_alert_bar": None})

    df = fetch_ohlc(ticker, INTERVAL, PERIOD)
    if df.empty or len(df) < 60:
        print("bot:", key, "not enough 5m data")
        state[key] = asset_state
        return

    feat = build_features(df)
    latest = feat.iloc[-1]
    bar_time = str(feat.index[-1])

    if asset_state.get("last_alert_bar") == bar_time:
        print("bot:", key, "bar already processed")
        state[key] = asset_state
        return

    direction = None
    if bool(latest["long_break"]):
        direction = "LONG"
    elif bool(latest["short_break"]):
        direction = "SHORT"

    if direction is None:
        print("bot:", key, "no breakout")
        state[key] = asset_state
        return

    trend = latest["trend"]
    if direction == "LONG" and trend == "down":
        print("bot:", key, "skip long breakout during 5m downtrend")
        state[key] = asset_state
        return
    if direction == "SHORT" and trend == "up":
        print("bot:", key, "skip short breakout during 5m uptrend")
        state[key] = asset_state
        return

    # ---- multi-timeframe confirmation ----
    h1_trend = get_trend_only(ticker, H1_INTERVAL, H1_PERIOD)
    d1_trend = get_trend_only(ticker, D1_INTERVAL, D1_PERIOD)

    if h1_trend is None or d1_trend is None:
        print("bot:", key, "skip, higher timeframe data unavailable")
        state[key] = asset_state
        return

    if direction == "LONG" and h1_trend != "up":
        print("bot:", key, "skip long, 1h trend not up:", h1_trend)
        state[key] = asset_state
        return
    if direction == "SHORT" and h1_trend != "down":
        print("bot:", key, "skip short, 1h trend not down:", h1_trend)
        state[key] = asset_state
        return
    if direction == "LONG" and d1_trend == "down":
        print("bot:", key, "skip long, 1d trend is down")
        state[key] = asset_state
        return
    if direction == "SHORT" and d1_trend == "up":
        print("bot:", key, "skip short, 1d trend is up")
        state[key] = asset_state
        return

    ml_prob = None
    model_path = os.path.join(MODEL_DIR, "breakout_model_" + key.lower() + ".joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        X = pd.DataFrame([latest[FEATURE_COLUMNS]])
        ml_prob = float(model.predict_proba(X)[0][1])
        if ml_prob < ML_PROB_THRESHOLD:
            print("bot:", key, "skip, ml probability below threshold", ml_prob)
            state[key] = asset_state
            return

    news = get_news_sentiment(asset["news_ticker"])
    if direction == "LONG" and news["score"] < NEWS_VETO_THRESHOLD:
        print("bot:", key, "skip long, news too bearish", news["score"])
        state[key] = asset_state
        return
    if direction == "SHORT" and news["score"] > -NEWS_VETO_THRESHOLD:
        print("bot:", key, "skip short, news too bullish", news["score"])
        state[key] = asset_state
        return

    price = float(latest["Close"])
    atr_val = float(latest["atr14"])
    if direction == "LONG":
        sl = price - atr_val * SL_ATR_MULT
        tp = price + atr_val * TP_ATR_MULT
        tp2 = price + atr_val * TP2_ATR_MULT
    else:
        sl = price + atr_val * SL_ATR_MULT
        tp = price - atr_val * TP_ATR_MULT
        tp2 = price - atr_val * TP2_ATR_MULT

    sl_points = abs(price - sl)
    lot_size, risk_amount_jpy = estimate_lot_size(sl_points, ml_prob, asset["contract_size"], asset["quote_currency"])

    embed = build_embed(
        asset["label"], direction, price, sl, tp, tp2, ml_prob, trend, h1_trend, d1_trend,
        news, lot_size, risk_amount_jpy, asset["contract_unit_label"]
    )
    send_discord(key + " " + direction + " signal at " + str(price), embed)

    asset_state["last_alert_bar"] = bar_time
    state[key] = asset_state


def main():
    state = load_state()
    for asset in ASSETS:
        try:
            process_asset(asset, state)
        except Exception as e:
            print("bot:", asset["key"], "failed:", e)
    save_state(state)


if __name__ == "__main__":
    main()
