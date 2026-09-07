"""
Shared configuration for all tracked assets and timeframes.
"""

ASSETS = [
    {
        "key": "GOLD",
        "label": "GOLD",
        "ticker": "GC=F",
        "news_ticker": "GC=F",
        "contract_unit_label": "oz",
        "contract_size": 1.0,
        "quote_currency": "USD",
    },
]

TIMEFRAMES = [
    {
        "key": "SCALP_5M",
        "label": "5m",
        "interval": "5m",
        "period": "60d",
        "resample": None,
    },
    {
        "key": "SWING_1H",
        "label": "1h",
        "interval": "60m",
        "period": "730d",
        "resample": None,
    },
    {
        "key": "SWING_4H",
        "label": "4h",
        "interval": "60m",
        "period": "730d",
        "resample": "4h",
    },
]

DAILY_INTERVAL = "1d"
DAILY_PERIOD = "5y"

DIRECTION_LOOKAHEAD_BARS = 12
DIRECTION_ATR_THRESHOLD = 0.5

TRADE_SL_ATR_MULT = 1.5
TRADE_TP_ATR_MULT = 2.5
TRADE_TP2_ATR_MULT = 4.0
TRADE_MAX_BARS = 60
