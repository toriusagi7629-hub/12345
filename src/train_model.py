"""
過去のゴールド価格データを取得し、「ブレイクアウトが本物だったか（さらに伸びたか）」を
学習するモデルを訓練して model/breakout_model.joblib に保存する。

実行頻度の目安: 週1回程度（GitHub Actionsのスケジュール実行を想定）
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

TICKER = "GC=F"          # ゴールド先物（現物XAUUSDと高い相関）
INTERVAL = "5m"           # 5分足（yfinanceの制約で5分足は直近60日分まで取得可能）
PERIOD = "60d"
LOOKAHEAD_BARS = 6        # ブレイク後、何本先までの値動きで成否を判定するか（5分足なら30分先）
SUCCESS_ATR_MULT = 0.8    # 成功と判定する値幅の基準（ATRの何倍動いたか）
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "breakout_model.joblib")


def fetch_data() -> pd.DataFrame:
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty:
        raise RuntimeError("価格データを取得できませんでした。ティッカーやyfinanceの状態を確認してください。")
    # yfinanceのMultiIndex列に対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_training_set(df: pd.DataFrame) -> pd.DataFrame:
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


def main():
    print(f"[train] {TICKER} {INTERVAL} データ取得中...")
    df = fetch_data()
    print(f"[train] {len(df)}本のローソク足を取得")

    train_df = build_training_set(df)
    print(f"[train] ブレイクアウトサンプル数: {len(train_df)}")

    if len(train_df) < 50:
        print("[train] サンプル数が少なすぎます。学習をスキップします（次回に持ち越し）。")
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
    print("[train] テスト精度:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[train] モデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
