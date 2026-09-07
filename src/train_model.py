"""
Trains one "direction" model per (asset, timeframe): predicts the probability
that price will be meaningfully higher (vs lower) after DIRECTION_LOOKAHEAD_BARS.

Two sources of training data are combined:
1. Historical price data: every bar is labeled up/down based on what actually
   happened next (excludes bars where the move was too small to call).
2. Real trade outcomes from trades.json: every closed signal this bot actually
   sent, labeled by whether it really won or lost. These are weighted more
   heavily since they reflect the bot's own real-world performance, not just
   generic price direction.

Saves each model to model/direction_model_<asset>_<timeframe>.joblib
Run frequency: about once a week (see .github/workflows/train_weekly.yml)
"""
import sys
import os
import json
import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS  # noqa: E402
from assets_config import ASSETS, TIMEFRAMES, DIRECTION_LOOKAHEAD_BARS, DIRECTION_ATR_THRESHOLD  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
TRADES_PATH = os.path.join(os.path.dirname(__file__), "..", "trades.json")
REAL_TRADE_WEIGHT = 5.0
MIN_SAMPLES = 100


def fetch_ohlc(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("no data returned for " + ticker + " " + interval)
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


def load_closed_trades():
    if not os.path.exists(TRADES_PATH):
        return []
    with open(TRADES_PATH, "r") as f:
        trades = json.load(f)
    return [t for t in trades if t.get("status") == "closed" and t.get("result") in ("WIN", "LOSS")]


def build_historical_samples(feat):
    labeled = feat.copy()
    labeled["future_close"] = labeled["Close"].shift(-DIRECTION_LOOKAHEAD_BARS)
    labeled["future_move"] = labeled["future_close"] - labeled["Close"]

    rows = []
    for idx, row in labeled.iterrows():
        if pd.isna(row["future_move"]) or pd.isna(row["atr14"]) or row["atr14"] == 0:
            continue
        threshold = row["atr14"] * DIRECTION_ATR_THRESHOLD
        if row["future_move"] > threshold:
            label = 1
        elif row["future_move"] < -threshold:
            label = 0
        else:
            continue
        rows.append({**{c: row[c] for c in FEATURE_COLUMNS}, "label": label})

    return pd.DataFrame(rows).dropna()


def build_real_trade_samples(asset, feat, closed_trades):
    rows = []
    asset_trades = [t for t in closed_trades if t.get("asset") == asset["key"]]
    if not asset_trades:
        return pd.DataFrame(rows)

    for t in asset_trades:
        try:
            entry_time = pd.Timestamp(t["entry_time"])
        except Exception:
            continue

        idx = feat.index
        if idx.tz is not None and entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        elif idx.tz is None and entry_time.tzinfo is not None:
            entry_time = entry_time.tz_localize(None)

        candidates = feat.index[feat.index <= entry_time]
        if len(candidates) == 0:
            continue
        bar = feat.loc[candidates[-1]]

        direction = t.get("direction")
        result = t.get("result")
        if (direction == "LONG" and result == "WIN") or (direction == "SHORT" and result == "LOSS"):
            label = 1
        else:
            label = 0

        row = {c: bar[c] for c in FEATURE_COLUMNS}
        row["label"] = label
        rows.append(row)

    return pd.DataFrame(rows).dropna()


def train_one(asset, tf, closed_trades):
    combo_name = asset["key"] + " " + tf["key"]
    print("train:", combo_name, "fetching data...")
    df = fetch_tf_data(asset["ticker"], tf)
    print("train:", combo_name, len(df), "bars fetched")

    feat = build_features(df)

    hist_df = build_historical_samples(feat)
    real_df = build_real_trade_samples(asset, feat, closed_trades)
    print("train:", combo_name, "historical samples:", len(hist_df), "| real trade samples:", len(real_df))

    if len(hist_df) < MIN_SAMPLES:
        print("train:", combo_name, "not enough historical samples, skipping this run")
        return

    if len(real_df) > 0:
        X = pd.concat([hist_df[FEATURE_COLUMNS], real_df[FEATURE_COLUMNS]], ignore_index=True)
        y = pd.concat([hist_df["label"], real_df["label"]], ignore_index=True)
        weights = pd.concat([
            pd.Series([1.0] * len(hist_df)),
            pd.Series([REAL_TRADE_WEIGHT] * len(real_df)),
        ], ignore_index=True)
    else:
        X = hist_df[FEATURE_COLUMNS]
        y = hist_df["label"]
        weights = pd.Series([1.0] * len(hist_df))

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    preds = model.predict(X_test)
    print("train:", combo_name, "test accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(
        MODEL_DIR, "direction_model_" + asset["key"].lower() + "_" + tf["key"].lower() + ".joblib"
    )
    joblib.dump(model, model_path)
    print("train:", combo_name, "model saved to", model_path)


def main():
    closed_trades = load_closed_trades()
    print("train: found", len(closed_trades), "closed real trades to learn from")
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                train_one(asset, tf, closed_trades)
            except Exception as e:
                print("train:", asset["key"], tf["key"], "failed:", e)


if __name__ == "__main__":
    main()
