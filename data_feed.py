"""
data_feed.py
------------
Real-time-ish OHLCV data fetching for both stocks and crypto, used by app.py.

- Crypto: ccxt REST polling against a real exchange (Binance by default) —
  genuinely real-time public market data, no API key required.
- Stocks: yfinance polling — delayed ~15 minutes on the free tier. Swap in
  a broker's streaming API (Alpaca, IBKR, Polygon) for true real-time equities.
"""

import pandas as pd


def fetch_crypto_ohlcv(symbol: str, timeframe: str = "1m", limit: int = 300, exchange_id: str = "binance") -> pd.DataFrame:
    import ccxt
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("datetime").drop(columns=["timestamp"])


def fetch_stock_ohlcv(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_ohlcv(asset_type: str, symbol: str, timeframe: str = "5m", limit: int = 300) -> pd.DataFrame:
    """Unified entry point used by the app: asset_type is 'crypto' or 'stock'."""
    if asset_type == "crypto":
        return fetch_crypto_ohlcv(symbol, timeframe=timeframe, limit=limit)
    else:
        # yfinance uses `period` not `limit`; pick a period wide enough for `limit` bars
        period_map = {"1m": "5d", "2m": "5d", "5m": "1mo", "15m": "1mo", "30m": "1mo",
                      "1h": "3mo", "1d": "2y"}
        period = period_map.get(timeframe, "1mo")
        return fetch_stock_ohlcv(symbol, period=period, interval=timeframe)


def fetch_index_quotes() -> list[dict]:
    """Latest price + % change for the major indices, for the top ticker bar."""
    import yfinance as yf
    indices = [("^GSPC", "S&P 500"), ("^IXIC", "NASDAQ Comp"), ("^RUT", "RUSSELL 2000")]
    results = []
    for ticker, label in indices:
        try:
            info = yf.Ticker(ticker).fast_info
            price = info.get("lastPrice") or info.get("last_price")
            prev_close = info.get("previousClose") or info.get("previous_close")
            change_pct = ((price / prev_close) - 1) * 100 if price and prev_close else None
            results.append(dict(label=label, price=price, change_pct=change_pct))
        except Exception:
            results.append(dict(label=label, price=None, change_pct=None))
    return results


def fetch_fundamentals(symbol: str) -> dict:
    """Basic fundamentals (P/E, market cap, 52wk range) for a stock — not available for crypto."""
    import yfinance as yf
    try:
        info = yf.Ticker(symbol).info
        return dict(
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            market_cap=info.get("marketCap"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            sector=info.get("sector"),
            name=info.get("shortName") or info.get("longName"),
        )
    except Exception:
        return dict(pe_ratio=None, forward_pe=None, market_cap=None,
                     fifty_two_week_high=None, fifty_two_week_low=None, sector=None, name=None)
