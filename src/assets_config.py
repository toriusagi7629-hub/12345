"""
Shared configuration for all tracked assets.
Add or edit entries here to change which instruments the bot watches.
"""

ASSETS = [
    {
        "key": "GOLD",
        "label": "GOLD",
        "ticker": "GC=F",
        "news_ticker": "GC=F",
        "contract_unit_label": "oz",
        "contract_size": 1.0,       # oz per 1 lot. Change to match your broker.
        "quote_currency": "USD",    # price is quoted in USD, needs USDJPY_RATE to convert to JPY
    },
    {
        "key": "BTC",
        "label": "BTC",
        "ticker": "BTC-USD",
        "news_ticker": "BTC-USD",
        "contract_unit_label": "BTC",
        "contract_size": 0.01,      # BTC per 1 lot. Change to match your broker.
        "quote_currency": "USD",
    },
    {
        "key": "USDJPY",
        "label": "USDJPY",
        "ticker": "JPY=X",
        "news_ticker": "JPY=X",
        "contract_unit_label": "USD",
        "contract_size": 1000.0,    # USD notional per 1 lot (e.g. 1000-currency-unit lot). Change to match your broker.
        "quote_currency": "JPY",    # price is already quoted in JPY, no conversion needed
    },
]
