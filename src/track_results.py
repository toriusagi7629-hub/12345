"""
Checks every open trade recorded in trades.json against actual price action
since it was signaled, and marks it WIN (hit TP), LOSS (hit SL), or TIMEOUT
(neither hit after a while). Run this frequently (piggybacks on the main
scalp_bot.yml schedule) so results get updated soon after they resolve.
"""
import os
import sys
import json
import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(__file__))
from assets_config import ASSETS, TIMEFRAMES  # noqa: E402

TRADES_PATH = os.path.join(os.path.dirname(__file__), "..", "trades.json")
ASSETS_BY_KEY = {a["key"]: a for a in ASSETS}
TIMEFRAMES_BY_KEY = {t["key"]: t for t in TIMEFRAMES}

MAX_OPEN_BARS = 60


def load_trades():
    if os.path.exists(TRADES_PATH):
        with open(TRADES_PATH, "r") as f:
            return json.load(f)
    return []


def save_trades(trades):
    with open(TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2)


def fetch_ohlc(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def resample_ohlc(df, rule):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


def check_trade(trade):
    asset = ASSETS_BY_KEY.get(trade["asset"])
    tf = TIMEFRAMES_BY_KEY.get(trade["timeframe"])
    if asset is None or tf is None:
        return trade

    try:
        df = fetch_ohlc(asset["ticker"], tf["interval"], tf["period"])
        if tf.get("resample"):
            df = resample_ohlc(df, tf["resample"])
    except Exception as e:
        print("track: fetch failed", trade["asset"], trade["timeframe"], e)
        return trade

    if df.empty:
        return trade

    entry_time = pd.Timestamp(trade["entry_time"])
    if df.index.tz is not None and entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    elif df.index.tz is None and entry_time.tzinfo is not None:
        entry_time = entry_time.tz_localize(None)

    bars_after = df[df.index > entry_time]
    if bars_after.empty:
        return trade

    direction = trade["direction"]
    sl = trade["sl"]
    tp = trade["tp"]

    bars_seen = 0
    for idx, row in bars_after.iterrows():
        bars_seen += 1
        high = row["High"]
        low = row["Low"]
        if direction == "LONG":
            hit_tp = high >= tp
            hit_sl = low <= sl
        else:
            hit_tp = low <= tp
            hit_sl = high >= sl

        if hit_sl:
            trade["status"] = "closed"
            trade["result"] = "LOSS"
            trade["close_time"] = str(idx)
            return trade
        if hit_tp:
            trade["status"] = "closed"
            trade["result"] = "WIN"
            trade["close_time"] = str(idx)
            return trade

    if bars_seen >= MAX_OPEN_BARS:
        trade["status"] = "closed"
        trade["result"] = "TIMEOUT"
        trade["close_time"] = str(bars_after.index[-1])

    return trade


def main():
    trades = load_trades()
    updated = 0
    for trade in trades:
        if trade.get("status") == "open":
            before = trade.get("status")
            check_trade(trade)
            if trade.get("status") != before:
                updated += 1
    save_trades(trades)
    print("track: checked open trades, updated", updated)


if __name__ == "__main__":
    main()
