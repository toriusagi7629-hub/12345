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
    {
        "key": "BTC",
        "label": "BTC",
        "ticker": "BTC-USD",
        "news_ticker": "BTC-USD",
        "contract_unit_label": "BTC",
        "contract_size": 0.01,
        "quote_currency": "USD",
    },
    {
        "key": "USDJPY",
        "label": "USDJPY",
        "ticker": "JPY=X",
        "news_ticker": "JPY=X",
        "contract_unit_label": "USD",
        "contract_size": 1000.0,
        "quote_currency": "JPY",
    },
]

# Each timeframe defines its own entry logic. "confirm_tf" points to another
# timeframe key whose trend must agree before a signal fires. "resample" is
# used when yfinance has no native interval for that timeframe (e.g. 4h).
TIMEFRAMES = [
    {
        "key": "SCALP_5M",
        "label": "5m scalp",
        "interval": "5m",
        "period": "60d",
        "resample": None,
        "confirm_tf": "SWING_1H",
    },
    {
        "key": "SWING_1H",
        "label": "1h swing",
        "interval": "60m",
        "period": "730d",
        "resample": None,
        "confirm_tf": "SWING_4H",
    },
    {
        "key": "SWING_4H",
        "label": "4h swing",
        "interval": "60m",
        "period": "730d",
        "resample": "4h",
        "confirm_tf": None,
    },
]

DAILY_INTERVAL = "1d"
DAILY_PERIOD = "5y"
