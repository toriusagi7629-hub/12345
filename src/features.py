"""
テクニカル指標と特徴量を計算するモジュール。
外部のtaライブラリに依存せず、pandas/numpyのみで計算する（環境依存を減らすため）。
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_atr = atr(df, length)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / tr_atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / tr_atr.replace(0, np.nan)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def donchian(df: pd.DataFrame, length: int = 20):
    upper = df["High"].rolling(length).max().shift(1)
    lower = df["Low"].rolling(length).min().shift(1)
    return upper, lower


def build_features(df: pd.DataFrame, donchian_len: int = 20) -> pd.DataFrame:
    """
    OHLCVデータフレーム(列: Open, High, Low, Close, Volume)から
    特徴量とブレイクアウト検出フラグを付加して返す。
    """
    out = df.copy()
    out["atr14"] = atr(out, 14)
    out["atr_avg50"] = out["atr14"].rolling(50).mean()
    out["rsi14"] = rsi(out["Close"], 14)
    out["adx14"] = adx(out, 14)
    out["ema20"] = ema(out["Close"], 20)
    out["ema50"] = ema(out["Close"], 50)
    out["ema200"] = ema(out["Close"], 200)

    upper, lower = donchian(out, donchian_len)
    out["donchian_upper"] = upper
    out["donchian_lower"] = lower

    out["vol_ok"] = out["atr14"] > (out["atr_avg50"] * 0.15)
    out["long_break"] = (out["Close"] > out["donchian_upper"]) & out["vol_ok"]
    out["short_break"] = (out["Close"] < out["donchian_lower"]) & out["vol_ok"]

    # トレンド判定: EMA50がEMA200より上かつADXが一定以上なら上昇トレンド、逆なら下降、弱ければレンジ
    out["trend"] = "range"
    out.loc[(out["ema50"] > out["ema200"]) & (out["adx14"] > 20), "trend"] = "up"
    out.loc[(out["ema50"] < out["ema200"]) & (out["adx14"] > 20), "trend"] = "down"

    # モデル用の特徴量（数値のみ）
    out["dist_from_ema20"] = (out["Close"] - out["ema20"]) / out["atr14"]
    out["ema_slope"] = (out["ema50"] - out["ema50"].shift(5)) / out["atr14"]
    out["atr_ratio"] = out["atr14"] / out["atr_avg50"]

    return out


FEATURE_COLUMNS = ["rsi14", "adx14", "dist_from_ema20", "ema_slope", "atr_ratio"]
