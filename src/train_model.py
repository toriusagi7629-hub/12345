"""
Fetches historical price data for every asset in assets_config.ASSETS and trains
one breakout-success model per asset, saving each to model/breakout_model_<key>.joblib

Run frequency: about once a week (intended for the scheduled GitHub Actions workflow)
"""
import sys
import os
import joblib
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS  # noqa: E402
from assets_config import ASSETS  # noqa: E402

INTERVAL = "5m"
PERIOD = "60d"
LOOKAHEAD_BARS = 6
SUCCESS_ATR_MULT = 0.8
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def fetch_data(ticker):
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty:
        raise RuntimeError("no data returned for " + ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_training_set(df):
    feat = build_features(df)
    feat["future_close"] = feat["Close"].shift(-LOOKAHEAD_BARS)
    feat["future_move"] = feat["future_close"] - feat["Close"]

    rows = []
    for idx, row in feat.iterrows():
        if pd.isna(row["future_move"]) or pd.isna(row["atr14"]) or row["atr14"] == 0:
            continue
        if row["long_break"]:
            success = int(row["future_move"] > row["atr14"] * SUCCESS_ATR_MULT)
            rows.append({**{c: row[c] for c in FEATURE_COLUMNS}, "label": success})
        elif row["short_break"]:
            success = int(row["future_move"] < -row["atr14"] * SUCCESS_ATR_MULT)
            rows.append({**{c: row[c] for c in FEATURE_COLUMNS}, "label": success})

    return pd.DataFrame(rows).dropna()


def train_one(asset):
    key = asset["key"]
    ticker = asset["ticker"]
    print("train:", key, ticker, "fetching data...")
    df = fetch_data(ticker)
    print("train:", key, len(df), "bars fetched")

    train_df = build_training_set(df)
    print("train:", key, "breakout samples:", len(train_df))

    if len(train_df) < 50:
        print("train:", key, "not enough samples, skipping this run")
        return

    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("train:", key, "test accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "breakout_model_" + key.lower() + ".joblib")
    joblib.dump(model, model_path)
    print("train:", key, "model saved to", model_path)


def main():
    for asset in ASSETS:
        try:
            train_one(asset)
        except Exception as e:
            print("train:", asset["key"], "failed:", e)


if __name__ == "__main__":
    main()
