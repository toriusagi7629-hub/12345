"""
Main script run on a schedule. Loops over every asset and every timeframe
(5m scalp, 1h swing, 4h swing) defined in assets_config.py.

For each (asset, timeframe):
1. Fetch price data for that timeframe and detect a breakout
2. Confirm against the next higher timeframe's trend, and against the daily trend
3. Predict breakout success probability with that combo's trained AI model
4. Fetch news sentiment
5. Estimate a suggested lot size based on account risk and AI confidence
6. Notify Discord if all filters pass
7. Record the last alerted bar per (asset, timeframe) in state.json
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
from assets_config import ASSETS, TIMEFRAMES, DAILY_INTERVAL, DAILY_PERIOD  # noqa: E402

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

TIMEFRAMES_BY_KEY = {tf["key"]: tf for tf in TIMEFRAMES}


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


def resample_ohlc(df, rule):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


def fetch_tf_data(ticker, tf):
    df = fetch_ohlc(ticker, tf["interval"], tf["period"])
    if tf.get("resample"):
        df = resample_ohlc(df, tf["resample"])
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


def build_embed(asset_label, tf_label, direction, price, sl, tp, tp2, ml_prob,
                 confirm_trend, daily_trend, news, lot_size, risk_amount_jpy, unit_label):
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
    confirm_txt = confirm_trend if confirm_trend is not None else "n/a"

    return {
        "title": arrow + " AI breakout alert: " + asset_label + " [" + tf_label + "]",
        "color": color,
        "fields": [
            {"name": "price", "value": str(round(price, 4)), "inline": True},
            {"name": "AI probability", "value": ml_txt, "inline": True},
            {"name": "SL", "value": str(round(sl, 4)) + " (-" + str(sl_pts) + ")", "inline": True},
            {"name": "TP", "value": str(round(tp, 4)) + " (+" + str(tp_pts) + ")", "inline": True},
            {"name": "TP2 extended", "value": str(round(tp2, 4)) + " (+" + str(tp2_pts) + ")", "inline": True},
            {"name": "higher tf trend", "value": confirm_txt, "inline": True},
            {"name": "daily trend", "value": daily_trend, "inline": True},
            {"name": "suggested lot", "value": str(lot_size) + " lot (" + unit_label + ")", "inline": True},
            {"name": "risk amount", "value": str(int(risk_amount_jpy)) + " JPY", "inline": True},
            {"name": "news sentiment", "value": sentiment_txt, "inline": True},
            {"name": "recent headlines", "value": headlines_txt, "inline": False},
        ],
    }


def get_daily_trend(ticker):
    try:
        df = fetch_ohlc(ticker, DAILY_INTERVAL, DAILY_PERIOD)
    except Exception as e:
        print("bot: daily fetch failed", ticker, e)
        return None
    if df.empty or len(df) < 60:
        return None
    feat = build_features(df)
    return feat.iloc[-1]["trend"]


def process_asset(asset, state):
    key = asset["key"]
    ticker = asset["ticker"]

    tf_features = {}
    tf_trend = {}
    for tf in TIMEFRAMES:
        try:
            df = fetch_tf_data(ticker, tf)
        except Exception as e:
            print("bot:", key, tf["key"], "fetch failed:", e)
            tf_features[tf["key"]] = None
            tf_trend[tf["key"]] = None
            continue
        if df.empty or len(df) < 60:
            print("bot:", key, tf["key"], "not enough data")
            tf_features[tf["key"]] = None
            tf_trend[tf["key"]] = None
            continue
        feat = build_features(df)
        tf_features[tf["key"]] = feat
        tf_trend[tf["key"]] = feat.iloc[-1]["trend"]

    daily_trend = get_daily_trend(ticker)

    for tf in TIMEFRAMES:
        combo_key = key + "_" + tf["key"]
        combo_state = state.get(combo_key, {"last_alert_bar": None})

        feat = tf_features.get(tf["key"])
        if feat is None:
            state[combo_key] = combo_state
            continue

        latest = feat.iloc[-1]
        bar_time = str(feat.index[-1])

        if combo_state.get("last_alert_bar") == bar_time:
            print("bot:", key, tf["key"], "bar already processed")
            state[combo_key] = combo_state
            continue

        direction = None
        if bool(latest["long_break"]):
            direction = "LONG"
        elif bool(latest["short_break"]):
            direction = "SHORT"

        if direction is None:
            print("bot:", key, tf["key"], "no breakout")
            state[combo_key] = combo_state
            continue

        own_trend = latest["trend"]
        if direction == "LONG" and own_trend == "down":
            print("bot:", key, tf["key"], "skip long, own trend down")
            state[combo_key] = combo_state
            continue
        if direction == "SHORT" and own_trend == "up":
            print("bot:", key, tf["key"], "skip short, own trend up")
            state[combo_key] = combo_state
            continue

        confirm_key = tf.get("confirm_tf")
        confirm_trend = tf_trend.get(confirm_key) if confirm_key else None
        if confirm_key:
            if confirm_trend is None:
                print("bot:", key, tf["key"], "skip, confirm timeframe unavailable")
                state[combo_key] = combo_state
                continue
            if direction == "LONG" and confirm_trend != "up":
                print("bot:", key, tf["key"], "skip long, higher tf not up:", confirm_trend)
                state[combo_key] = combo_state
                continue
            if direction == "SHORT" and confirm_trend != "down":
                print("bot:", key, tf["key"], "skip short, higher tf not down:", confirm_trend)
                state[combo_key] = combo_state
                continue

        if daily_trend is None:
            print("bot:", key, tf["key"], "skip, daily trend unavailable")
            state[combo_key] = combo_state
            continue
        if direction == "LONG" and daily_trend == "down":
            print("bot:", key, tf["key"], "skip long, daily trend down")
            state[combo_key] = combo_state
            continue
        if direction == "SHORT" and daily_trend == "up":
            print("bot:", key, tf["key"], "skip short, daily trend up")
            state[combo_key] = combo_state
            continue

        ml_prob = None
        model_path = os.path.join(
            MODEL_DIR, "breakout_model_" + key.lower() + "_" + tf["key"].lower() + ".joblib"
        )
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            X = pd.DataFrame([latest[FEATURE_COLUMNS]])
            ml_prob = float(model.predict_proba(X)[0][1])
            if ml_prob < ML_PROB_THRESHOLD:
                print("bot:", key, tf["key"], "skip, ml probability below threshold", ml_prob)
                state[combo_key] = combo_state
                continue

        news = get_news_sentiment(asset["news_ticker"])
        if direction == "LONG" and news["score"] < NEWS_VETO_THRESHOLD:
            print("bot:", key, tf["key"], "skip long, news too bearish", news["score"])
            state[combo_key] = combo_state
            continue
        if direction == "SHORT" and news["score"] > -NEWS_VETO_THRESHOLD:
            print("bot:", key, tf["key"], "skip short, news too bullish", news["score"])
            state[combo_key] = combo_state
            continue

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
        lot_size, risk_amount_jpy = estimate_lot_size(
            sl_points, ml_prob, asset["contract_size"], asset["quote_currency"]
        )

        embed = build_embed(
            asset["label"], tf["label"], direction, price, sl, tp, tp2, ml_prob,
            confirm_trend, daily_trend, news, lot_size, risk_amount_jpy, asset["contract_unit_label"]
        )
        send_discord(key + " " + tf["key"] + " " + direction + " signal at " + str(price), embed)

        combo_state["last_alert_bar"] = bar_time
        state[combo_key] = combo_state


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
