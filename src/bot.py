"""
Main script run on a schedule. For every asset, analyzes ALL timeframes every
single run (not gated by a breakout event) and produces one LONG / SHORT /
NO TRADE decision per asset.

Flow per asset:
1. For each timeframe (5m, 1h, 4h): fetch data, compute trend + AI up-probability
2. Only call LONG if every timeframe leans up AND daily trend is not down
   Only call SHORT if every timeframe leans down AND daily trend is not up
   Otherwise: NO TRADE
3. If a directional call is made, check news sentiment doesn't strongly oppose it
4. Estimate a suggested lot size based on account risk and AI confidence
5. Notify Discord only when the decision changes (avoids repeating the same
   call every 5 minutes while conditions persist)
6. Log every notified decision as a trade in trades.json so results can be
   tracked and fed back into future training
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
from assets_config import (  # noqa: E402
    ASSETS, TIMEFRAMES, DAILY_INTERVAL, DAILY_PERIOD,
    TRADE_SL_ATR_MULT, TRADE_TP_ATR_MULT, TRADE_TP2_ATR_MULT,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state.json")
TRADES_PATH = os.path.join(os.path.dirname(__file__), "..", "trades.json")

AI_UP_THRESHOLD = 0.55
AI_DOWN_THRESHOLD = 0.45
NEWS_VETO_THRESHOLD = -0.4

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


def load_trades():
    if os.path.exists(TRADES_PATH):
        with open(TRADES_PATH, "r") as f:
            return json.load(f)
    return []


def save_trades(trades):
    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2)


def log_trade(asset_key, tf_key, direction, price, sl, tp, tp2, ml_prob, entry_time):
    trades = load_trades()
    trade = {
        "id": asset_key + "_" + tf_key + "_" + entry_time,
        "asset": asset_key,
        "timeframe": tf_key,
        "direction": direction,
        "entry_price": price,
        "sl": sl,
        "tp": tp,
        "tp2": tp2,
        "ml_prob": ml_prob,
        "entry_time": entry_time,
        "status": "open",
        "result": None,
        "close_time": None,
    }
    trades.append(trade)
    save_trades(trades)


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


def analyze_timeframe(asset, tf):
    try:
        df = fetch_tf_data(asset["ticker"], tf)
    except Exception as e:
        print("bot:", asset["key"], tf["key"], "fetch failed:", e)
        return None
    if df.empty or len(df) < 60:
        print("bot:", asset["key"], tf["key"], "not enough data")
        return None

    feat = build_features(df)
    latest = feat.iloc[-1]
    trend = latest["trend"]

    ml_prob_up = None
    model_path = os.path.join(
        MODEL_DIR, "direction_model_" + asset["key"].lower() + "_" + tf["key"].lower() + ".joblib"
    )
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        X = pd.DataFrame([latest[FEATURE_COLUMNS]])
        ml_prob_up = float(model.predict_proba(X)[0][1])

    bias = "neutral"
    if ml_prob_up is not None:
        if trend == "up" and ml_prob_up >= AI_UP_THRESHOLD:
            bias = "up"
        elif trend == "down" and ml_prob_up <= AI_DOWN_THRESHOLD:
            bias = "down"
    else:
        if trend in ("up", "down"):
            bias = trend

    return {
        "trend": trend,
        "ml_prob_up": ml_prob_up,
        "bias": bias,
        "close": float(latest["Close"]),
        "atr": float(latest["atr14"]),
        "bar_time": str(feat.index[-1]),
    }


def build_embed(asset_label, decision, price, sl, tp, tp2, tf_results, daily_trend,
                 news, lot_size, risk_amount_jpy, unit_label):
    is_long = decision == "LONG"
    color = 3066993 if is_long else 15158332
    arrow = "LONG" if is_long else "SHORT"

    sl_pts = round(abs(price - sl), 4)
    tp_pts = round(abs(tp - price), 4)
    tp2_pts = round(abs(tp2 - price), 4)

    tf_lines = ""
    for tf in TIMEFRAMES:
        r = tf_results.get(tf["key"])
        if r is None:
            tf_lines = tf_lines + tf["label"] + ": no data\n"
            continue
        prob_txt = (str(round(r["ml_prob_up"] * 100, 1)) + "% up") if r["ml_prob_up"] is not None else "no model"
        tf_lines = tf_lines + tf["label"] + ": trend=" + r["trend"] + " ai=" + prob_txt + " bias=" + r["bias"] + "\n"

    headlines_txt = ""
    for h in news["top_headlines"][:3]:
        headlines_txt = headlines_txt + "- " + h + "\n"
    if headlines_txt == "":
        headlines_txt = "no news"
    sentiment_txt = str(news["score"]) + " (" + str(news["headline_count"]) + ")"

    return {
        "title": arrow + " AI decision: " + asset_label,
        "color": color,
        "fields": [
            {"name": "price", "value": str(round(price, 4)), "inline": True},
            {"name": "SL", "value": str(round(sl, 4)) + " (-" + str(sl_pts) + ")", "inline": True},
            {"name": "TP", "value": str(round(tp, 4)) + " (+" + str(tp_pts) + ")", "inline": True},
            {"name": "TP2 extended", "value": str(round(tp2, 4)) + " (+" + str(tp2_pts) + ")", "inline": True},
            {"name": "daily trend", "value": str(daily_trend), "inline": True},
            {"name": "suggested lot", "value": str(lot_size) + " lot (" + unit_label + ")", "inline": True},
            {"name": "risk amount", "value": str(int(risk_amount_jpy)) + " JPY", "inline": True},
            {"name": "news sentiment", "value": sentiment_txt, "inline": True},
            {"name": "timeframe breakdown", "value": tf_lines, "inline": False},
            {"name": "recent headlines", "value": headlines_txt, "inline": False},
        ],
    }


def process_asset(asset, state):
    key = asset["key"]

    tf_results = {}
    for tf in TIMEFRAMES:
        tf_results[tf["key"]] = analyze_timeframe(asset, tf)

    if any(tf_results[tf["key"]] is None for tf in TIMEFRAMES):
        print("bot:", key, "skip, missing timeframe data this cycle")
        return

    daily_trend = get_daily_trend(asset["ticker"])

    biases = [tf_results[tf["key"]]["bias"] for tf in TIMEFRAMES]
    all_up = all(b == "up" for b in biases)
    all_down = all(b == "down" for b in biases)

    decision = "NO_TRADE"
    if all_up and daily_trend != "down":
        decision = "LONG"
    elif all_down and daily_trend != "up":
        decision = "SHORT"

    asset_state = state.get(key, {"last_decision": "NO_TRADE"})

    if decision == "NO_TRADE":
        print("bot:", key, "NO TRADE this cycle")
        asset_state["last_decision"] = "NO_TRADE"
        state[key] = asset_state
        return

    news = get_news_sentiment(asset["news_ticker"])
    if decision == "LONG" and news["score"] < NEWS_VETO_THRESHOLD:
        print("bot:", key, "would be LONG but news too bearish", news["score"])
        asset_state["last_decision"] = "NO_TRADE"
        state[key] = asset_state
        return
    if decision == "SHORT" and news["score"] > -NEWS_VETO_THRESHOLD:
        print("bot:", key, "would be SHORT but news too bullish", news["score"])
        asset_state["last_decision"] = "NO_TRADE"
        state[key] = asset_state
        return

    if asset_state.get("last_decision") == decision:
        print("bot:", key, "same decision as last notified, skipping:", decision)
        state[key] = asset_state
        return

    primary_tf = TIMEFRAMES[0]
    primary = tf_results[primary_tf["key"]]
    price = primary["close"]
    atr_val = primary["atr"]
    bar_time = primary["bar_time"]

    if decision == "LONG":
        sl = price - atr_val * TRADE_SL_ATR_MULT
        tp = price + atr_val * TRADE_TP_ATR_MULT
        tp2 = price + atr_val * TRADE_TP2_ATR_MULT
        probs = [tf_results[tf["key"]]["ml_prob_up"] for tf in TIMEFRAMES if tf_results[tf["key"]]["ml_prob_up"] is not None]
    else:
        sl = price + atr_val * TRADE_SL_ATR_MULT
        tp = price - atr_val * TRADE_TP_ATR_MULT
        tp2 = price - atr_val * TRADE_TP2_ATR_MULT
        probs = [1.0 - tf_results[tf["key"]]["ml_prob_up"] for tf in TIMEFRAMES if tf_results[tf["key"]]["ml_prob_up"] is not None]

    avg_prob = (sum(probs) / len(probs)) if probs else None

    sl_points = abs(price - sl)
    lot_size, risk_amount_jpy = estimate_lot_size(sl_points, avg_prob, asset["contract_size"], asset["quote_currency"])

    embed = build_embed(
        asset["label"], decision, price, sl, tp, tp2, tf_results, daily_trend,
        news, lot_size, risk_amount_jpy, asset["contract_unit_label"]
    )
    send_discord(key + " " + decision + " decision at " + str(price), embed)
    log_trade(key, primary_tf["key"], decision, price, sl, tp, tp2, avg_prob, bar_time)

    asset_state["last_decision"] = decision
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
