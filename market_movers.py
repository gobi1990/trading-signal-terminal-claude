"""
market_movers.py
-----------------
"Top active" and "penny stock" screener built from a curated, liquid
universe rather than a true whole-market scanner (no free API provides a
real-time full-market screener). Ranks by relative volume and % change
using real-time quotes — a solid practical approximation, but worth
knowing it's not scanning every ticker on the exchange.
"""

import pandas as pd

# Curated universe of typically high-volume, liquid large/mega-cap names.
ACTIVE_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO",
    "JPM", "V", "WMT", "XOM", "UNH", "ORCL", "COST", "BAC", "PLTR", "SMCI",
    "INTC", "F", "T", "PFE", "DIS", "CSCO", "SOFI", "NIO", "RIVN", "COIN",
]

# Curated universe of commonly-traded sub-$5 "penny stocks". This list can
# go stale — companies move above/below $5 constantly — treat it as a
# reasonable starting watchlist, not an authoritative penny-stock scanner.
PENNY_UNIVERSE = [
    "SIRI", "NOK", "PLUG", "SNDL", "NAKD", "ZOM", "CTRM", "GEVO", "IDEX", "XELA",
    "MULN", "SOS", "BBIG", "TOPS", "SHIP", "GNUS", "NKLA", "WKHS", "CLOV", "OCGN",
]


def _quote_row(symbol: str, fast_info: dict) -> dict:
    price = fast_info.get("lastPrice") or fast_info.get("last_price")
    prev_close = fast_info.get("previousClose") or fast_info.get("previous_close")
    volume = fast_info.get("lastVolume") or fast_info.get("last_volume") or fast_info.get("regularMarketVolume")
    avg_volume = fast_info.get("threeMonthAverageVolume") or fast_info.get("three_month_average_volume")
    change_pct = ((price / prev_close) - 1) * 100 if price and prev_close else None
    rvol = (volume / avg_volume) if volume and avg_volume else None
    return dict(symbol=symbol, price=price, change_pct=change_pct, volume=volume, rvol=rvol)


def get_market_movers(universe: list[str], top_n: int = 20) -> pd.DataFrame:
    """Fetch real-time quotes for the given universe and rank by volume (most active first)."""
    import yfinance as yf
    rows = []
    for symbol in universe:
        try:
            fast_info = yf.Ticker(symbol).fast_info
            rows.append(_quote_row(symbol, dict(fast_info)))
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["price"]).sort_values("volume", ascending=False, na_position="last")
    return df.head(top_n).reset_index(drop=True)


def get_active_stocks(top_n: int = 20) -> pd.DataFrame:
    return get_market_movers(ACTIVE_UNIVERSE, top_n)


def get_penny_stocks(top_n: int = 20) -> pd.DataFrame:
    df = get_market_movers(PENNY_UNIVERSE, top_n)
    if df.empty:
        return df
    return df[df["price"] < 5].reset_index(drop=True)
