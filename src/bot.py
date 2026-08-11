"""
Main script run on a schedule.
1. Fetch latest gold price data
2. Compute breakout / trend / ATR features
3. Predict breakout success probability with the trained AI model
4. Fetch news sentiment
5. Estimate a suggested lot size based on account risk and AI confidence
6. Combine everything and notify Discord
7. Record the last alerted bar in state.json to avoid duplicate notifications
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

TICKER = "GC=F"
INTERVAL = "5m"
PERIOD = "5d"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "breakout_model.joblib")
STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state.json")

ML_PROB_THRESHOLD = 0.55
NEWS_VETO_THRESHOLD = -0.4
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5
TP2_ATR_MULT = 4.0

# ---- Position sizing settings (edit these to match your real account/broker) ----
ACCOUNT_BALANCE_JPY = 10000.0   # account balance in JPY
RISK_PERCENT = 0.02              # max % of balance to risk per trade (2%)
CONTRACT_SIZE_OZ = 1.0           # oz of gold per 1 lot. CHANGE THIS to match your broker's contract spec.
USDJPY_RATE = 150.0              # approximate USD/JPY rate. Update to current rate for accuracy.
MIN_CONFIDENCE_SCALE = 0.3       # lot size floor as a fraction of full risk, used when AI probability is near threshold

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"last_alert_bar": None}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def fetch_latest():
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


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


def estimate_lot_size(sl_points_usd, ml_prob):
    scale = confidence_scale(ml_prob)
    risk_amount_jpy = ACCOUNT_BALANCE_JPY * RISK_PERCENT * scale
    jpy_loss_per_lot = sl_points_usd * CONTRACT_SIZE_OZ * USDJPY_RATE
    if jpy_loss_per_lot <= 0:
        return 0.0, risk_amount_jpy
    lot = risk_amount_jpy / jpy_loss_per_lot
    lot = round(lot, 2)
    return lot, round(risk_amount_jpy, 0)


def build_embed(direction, price, sl, tp, tp2, ml_prob, trend, news, lot_size, risk_amount_jpy):
    is_long = direction == "LONG"
    color = 3066993 if is_long else 15158332
    arrow = "LONG" if is_long else "SHORT"
    ml_txt = str(ml_prob) if ml_prob is not None else "no model"

    sl_pts = round(abs(price - sl), 2)
    tp_pts = round(abs(tp - price), 2)
    tp2_pts = round(abs(tp2 - price), 2)

    headlines_txt = ""
    for h in news["top_headlines"][:3]:
        headlines_txt = headlines_txt + "- " + h + "\n"
    if headlines_txt == "":
        headlines_txt = "no news"

    sentiment_txt = str(news["score"]) + " (" + str(news["headline_count"]) + ")"

    return {
        "title": arrow + " AI breakout alert: GOLD",
        "color": color,
        "fields": [
            {"name": "price", "value": str(round(price, 2)), "inline": True},
            {"name": "AI probability", "value": ml_txt, "inline": True},
            {"name": "SL", "value": str(round(sl, 2)) + " (-" + str(sl_pts) + "pt)", "inline": True},
            {"name": "TP", "value": str(round(tp, 2)) + " (+" + str(tp_pts) + "pt)", "inline": True},
            {"name": "TP2 extended", "value": str(round(tp2, 2)) + " (+" + str(tp2_pts) + "pt)", "inline": True},
            {"name": "trend", "value": trend, "inline": True},
            {"name": "suggested lot", "value": str(lot_size) + " lot", "inline": True},
            {"name": "risk amount", "value": str(int(risk_amount_jpy)) + " JPY", "inline": True},
            {"name": "news sentiment", "value": sentiment_txt, "inline": True},
            {"name": "recent headlines", "value": headlines_txt, "inline": False},
        ],
    }


def main():
    state = load_state()

    df = fetch_latest()
    if df.empty or len(df) < 60:
        print("bot: not enough data")
        return

    feat = build_features(df)
    latest = feat.iloc[-1]
    bar_time = str(feat.index[-1])

    if state.get("last_alert_bar") == bar_time:
        print("bot: bar already processed")
        return

    direction = None
    if bool(latest["long_break"]):
        direction = "LONG"
    elif bool(latest["short_break"]):
        direction = "SHORT"

    if direction is None:
        print("bot: no breakout")
        return

    trend = latest["trend"]
    if direction == "LONG" and trend == "down":
        print("bot: skip long breakout during downtrend")
        return
    if direction == "SHORT" and trend == "up":
        print("bot: skip short breakout during uptrend")
        return

    ml_prob = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        X = pd.DataFrame([latest[FEATURE_COLUMNS]])
        ml_prob = float(model.predict_proba(X)[0][1])
        if ml_prob < ML_PROB_THRESHOLD:
            print("bot: skip, ml probability below threshold", ml_prob)
            return

    news = get_news_sentiment()
    if direction == "LONG" and news["score"] < NEWS_VETO_THRESHOLD:
        print("bot: skip long, news too bearish", news["score"])
        return
    if direction == "SHORT" and news["score"] > -NEWS_VETO_THRESHOLD:
        print("bot: skip short, news too bullish", news["score"])
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

    sl_points_usd = abs(price - sl)
    lot_size, risk_amount_jpy = estimate_lot_size(sl_points_usd, ml_prob)

    embed = build_embed(direction, price, sl, tp, tp2, ml_prob, trend, news, lot_size, risk_amount_jpy)
    send_discord(direction + " signal at " + str(price), embed)

    state["last_alert_bar"] = bar_time
    save_state(state)


if __name__ == "__main__":
    main()
