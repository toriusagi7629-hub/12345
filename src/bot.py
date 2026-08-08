"""
定期実行されるメインスクリプト。
1. 最新のゴールド価格データを取得
2. ブレイクアウト・トレンド・ATRなどの特徴量を計算
3. 学習済みAIモデルでブレイクアウトの成功確率を予測
4. ニュースセンチメントを取得
5. すべてを総合判断してDiscordに通知
6. 二重通知を防ぐため、直近で通知したバーの時刻をstate.jsonに記録
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

ML_PROB_THRESHOLD = 0.55      # このAI予測確率以上のみ通知
NEWS_VETO_THRESHOLD = -0.4    # ロングで、ニュースがこれより弱気ならロング通知を見送る（逆方向も同様）
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5

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


def send_discord(payload_text: str, embed: dict):
    if not DISCORD_WEBHOOK_URL:
        print("[bot] DISCORD_WEBHOOK_URLが未設定のため、通知をスキップして内容のみ表示します。")
        print(payload_text)
        return
    resp = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=15)
    if resp.status_code >= 300:
        print(f"[bot] Discord送信失敗: {resp.status_code} {resp.text}")
    else:
        print("[bot] Discord通知を送信しました。")


def build_embed(direction, price, sl, tp, ml_prob, trend, news):
    is_long = direction == "LONG"
    color = 3066993 if is_long else 15158332
    arrow = "🟢 LONG" if is_long else "🔴 SHORT"
    ml_txt = f"{ml_prob:.0%}" if ml_prob is not None else "モデル未学習"

    headlines_txt = "\n".join(f"・{h}" for h in news["top_headlines"][:3]) or "関連ニュースなし"

    return {
        "title": f"{arrow} AIブレイクアウト通知: GOLD",
        "color": color,
        "fields": [
            {"name": "価格", "value": str(round(price, 2)), "inline": True},
            {"name": "SL目安", "value": str(round(sl, 2)), "inline": True},
            {"name": "TP目安", "value": str(round(tp, 2)), "inline": True},
            {"name": "AI成功確率", "value": ml_txt, "inline": True},
            {"name": "トレンド判定", "value": trend, "inline": True},
            {"name": "ニュースセンチメント", "value": f"{news['score']:+.2f}（{news['headline_count']}件）", "inline": True},
            {"name": "直近ニュース見出し", "value": headlines_txt, "inline": False},
        ],
    }


def main():
    state = load_state()

    df = fetch_latest()
    if df.empty or len(df) < 60:
        print("[bot] データ不足のため終了します。")
        return

    feat = build_features(df)
    latest = feat.iloc[-1]
    bar_time = str(feat.index[-1])

    if state.get("last_alert_bar") == bar_time:
        print("[bot] このバーは既に処理済みです。")
        return

    direction = None
    if bool(latest["long_break"]):
        direction = "LONG"
    elif bool(latest["short_break"]):
        direction = "SHORT"

    if direction is None:
        print("[bot] ブレイクアウトなし。")
        return

    # --- トレンドフィルター ---
    trend = latest["trend"]
    if direction == "LONG" and trend == "down":
        print("[bot] 下降トレンド中のロングブレイクのため見送り。")
        return
    if direction == "SHORT" and trend == "up":
        print("[bot] 上昇トレンド中のショートブレイクのため見送り。")
        return

    # --- AIモデルによる確率予測 ---
    ml_prob = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        X = pd.DataFrame([latest[FEATURE_COLUMNS]])
        ml_prob = float(model.predict_proba(X)[0][1])
        if ml_prob < ML_PROB_THRESHOLD:
            print(f"[bot] AI成功確率が閾値未満（{ml_prob:.2f}）のため見送り。")
            return

    # --- ニュースセンチメントフィルター ---
    news = get_news_sentiment()
    if direction == "LONG" and news["score"] < NEWS_VETO_THRESHOLD:
        print(f"[bot] ニュースが強く弱気（{news['score']}）のためロングを見送り。")
        return
    if direction == "SHORT" and news["score"] > -NEWS_VETO_THRESHOLD:
        print(f"[bot] ニュースが強く強気（{news['score']}）のためショートを
