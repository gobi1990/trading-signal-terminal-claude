"""
indicators.py
-------------
Thin wrapper around TA-Lib. Falls back to pure-pandas implementations
automatically if TA-Lib's C library isn't installed on the host machine,
so the rest of the codebase never has to care which one is active.
"""

import numpy as np
import pandas as pd

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


def sma(series: pd.Series, period: int) -> pd.Series:
    if TALIB_AVAILABLE:
        return pd.Series(talib.SMA(series.values, timeperiod=period), index=series.index)
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    if TALIB_AVAILABLE:
        return pd.Series(talib.EMA(series.values, timeperiod=period), index=series.index)
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if TALIB_AVAILABLE:
        return pd.Series(talib.RSI(series.values, timeperiod=period), index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    if TALIB_AVAILABLE:
        macd_line, signal_line, hist = talib.MACD(
            series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        idx = series.index
        return pd.Series(macd_line, index=idx), pd.Series(signal_line, index=idx), pd.Series(hist, index=idx)
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(series: pd.Series, period: int = 20, nbdev: float = 2.0):
    if TALIB_AVAILABLE:
        upper, mid, lower = talib.BBANDS(
            series.values, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev
        )
        idx = series.index
        return pd.Series(upper, index=idx), pd.Series(mid, index=idx), pd.Series(lower, index=idx)
    mid = sma(series, period)
    std = series.rolling(period).std()
    return mid + nbdev * std, mid, mid - nbdev * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if TALIB_AVAILABLE:
        return pd.Series(
            talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index
        )
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if TALIB_AVAILABLE:
        return pd.Series(
            talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index
        )
    # simplified fallback (Wilder's smoothing approximated with EMA)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = atr(high, low, close, period)
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/period).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/period).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
         session_reset: bool = True) -> pd.Series:
    """
    Volume Weighted Average Price. By default resets each calendar day
    (the standard intraday VWAP every trading platform shows) — pass
    session_reset=False for a running VWAP over the whole series instead
    (more useful on daily bars, where a "session" reset doesn't apply).
    """
    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume

    if session_reset and isinstance(close.index, pd.DatetimeIndex):
        day = close.index.normalize()
        cum_tp_vol = tp_vol.groupby(day).cumsum()
        cum_vol = volume.groupby(day).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = volume.cumsum()

    return cum_tp_vol / cum_vol.replace(0, np.nan)


def rvol(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Relative volume: current bar's volume vs its own rolling average. >1 means above-average activity."""
    avg_vol = volume.rolling(lookback).mean()
    return volume / avg_vol.replace(0, np.nan)


def support_resistance(df: pd.DataFrame, window: int = 20, lookback: int = 150):
    """
    Simple swing-based support/resistance: a bar is a swing low/high if it's
    the lowest/highest close within `window` bars on each side. Returns the
    nearest confirmed support (below current price) and resistance (above
    current price) found within the last `lookback` bars.
    """
    recent = df.tail(lookback)
    lows, highs = recent["low"], recent["high"]
    current_price = float(df["close"].iloc[-1])

    swing_lows, swing_highs = [], []
    n = len(recent)
    for i in range(window, n - window):
        segment_low = lows.iloc[i - window:i + window + 1]
        segment_high = highs.iloc[i - window:i + window + 1]
        if lows.iloc[i] == segment_low.min():
            swing_lows.append(lows.iloc[i])
        if highs.iloc[i] == segment_high.max():
            swing_highs.append(highs.iloc[i])

    supports_below = [p for p in swing_lows if p < current_price]
    resistances_above = [p for p in swing_highs if p > current_price]

    nearest_support = max(supports_below) if supports_below else None
    nearest_resistance = min(resistances_above) if resistances_above else None
    return nearest_support, nearest_resistance
