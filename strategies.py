"""
strategies.py
-------------
Combines several independent trading strategies into per-bar signals
(-1 = bearish, 0 = neutral, +1 = bullish) and a consensus score.

Strategies included:
  1. RSI mean-reversion       (oversold/overbought)
  2. EMA trend-following      (20/50 EMA cross + price alignment)
  3. SMA trend-following      (price vs SMA50 — a slower, smoother trend read)
  4. MACD momentum            (histogram sign + crossover)
  5. Bollinger Band reversion (price at bands)
  6. VWAP                     (price above/below volume-weighted average price)
  7. FVG reaction             (price retracing into an unfilled gap, in
                                the direction the gap implies continuation)
  8. VWAP + FVG confluence    (bonus high-conviction signal: both agree)

All strategies are long-biased (they look for buy setups on strength/support
and sell/exit setups on weakness/resistance) — there is no short-selling logic.
"""

from dataclasses import dataclass

import pandas as pd

import indicators as ta
from fvg import detect_fair_value_gaps, nearest_unfilled_zone


@dataclass
class StrategySignal:
    name: str
    signal: int          # -1, 0, +1
    reason: str


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ta.ema(out["close"], 20)
    out["ema_slow"] = ta.ema(out["close"], 50)
    out["ema_200"] = ta.ema(out["close"], 200)
    out["sma_50"] = ta.sma(out["close"], 50)
    out["rsi"] = ta.rsi(out["close"], 14)
    macd_line, signal_line, hist = ta.macd(out["close"], 12, 26, 9)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd_line, signal_line, hist
    upper, mid, lower = ta.bollinger_bands(out["close"], 20, 2.0)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = upper, mid, lower
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], 14)
    out["rvol"] = ta.rvol(out["volume"], 20)
    # daily-bar data has no meaningful intraday "session" to reset on, so use a running VWAP there
    session_reset = True
    if isinstance(out.index, pd.DatetimeIndex) and len(out.index) > 1:
        # if bars are >= 1 day apart, treat as daily/weekly and don't reset per calendar day
        if (out.index[1] - out.index[0]) >= pd.Timedelta(days=1):
            session_reset = False
    out["vwap"] = ta.vwap(out["high"], out["low"], out["close"], out["volume"], session_reset=session_reset)
    return out


def rsi_strategy(row, rsi_oversold=30, rsi_overbought=70) -> StrategySignal:
    if row.rsi < rsi_oversold:
        return StrategySignal("RSI Mean-Reversion", 1, f"RSI {row.rsi:.1f} oversold (<{rsi_oversold})")
    if row.rsi > rsi_overbought:
        return StrategySignal("RSI Mean-Reversion", -1, f"RSI {row.rsi:.1f} overbought (>{rsi_overbought})")
    return StrategySignal("RSI Mean-Reversion", 0, f"RSI {row.rsi:.1f} neutral")


def ema_trend_strategy(row) -> StrategySignal:
    if row.ema_fast > row.ema_slow and row.close > row.ema_fast:
        return StrategySignal("EMA Trend (20/50)", 1, "Price > EMA20 > EMA50 (uptrend)")
    if row.ema_fast < row.ema_slow and row.close < row.ema_fast:
        return StrategySignal("EMA Trend (20/50)", -1, "Price < EMA20 < EMA50 (downtrend)")
    return StrategySignal("EMA Trend (20/50)", 0, "No clear EMA trend alignment")


def sma_trend_strategy(row) -> StrategySignal:
    if pd.isna(row.sma_50):
        return StrategySignal("SMA Trend (50)", 0, "Not enough bars for SMA50 yet")
    if row.close > row.sma_50:
        return StrategySignal("SMA Trend (50)", 1, "Price above SMA50 (bullish bias)")
    if row.close < row.sma_50:
        return StrategySignal("SMA Trend (50)", -1, "Price below SMA50 (bearish bias)")
    return StrategySignal("SMA Trend (50)", 0, "Price at SMA50")


def macd_strategy(row) -> StrategySignal:
    if row.macd_hist > 0 and row.macd > row.macd_signal:
        return StrategySignal("MACD Momentum", 1, "MACD above signal, histogram positive")
    if row.macd_hist < 0 and row.macd < row.macd_signal:
        return StrategySignal("MACD Momentum", -1, "MACD below signal, histogram negative")
    return StrategySignal("MACD Momentum", 0, "MACD mixed")


def bollinger_strategy(row) -> StrategySignal:
    if row.close <= row.bb_lower:
        return StrategySignal("Bollinger Reversion", 1, "Price at/below lower band")
    if row.close >= row.bb_upper:
        return StrategySignal("Bollinger Reversion", -1, "Price at/above upper band")
    return StrategySignal("Bollinger Reversion", 0, "Price inside bands")


def vwap_strategy(row) -> StrategySignal:
    if pd.isna(row.vwap):
        return StrategySignal("VWAP", 0, "Not enough bars for VWAP yet")
    if row.close > row.vwap:
        return StrategySignal("VWAP", 1, f"Price ({row.close:.2f}) above VWAP ({row.vwap:.2f}) — bullish")
    if row.close < row.vwap:
        return StrategySignal("VWAP", -1, f"Price ({row.close:.2f}) below VWAP ({row.vwap:.2f}) — bearish")
    return StrategySignal("VWAP", 0, "Price at VWAP")


def fvg_strategy(df_sig: pd.DataFrame, row, price: float) -> StrategySignal:
    zones = detect_fair_value_gaps(df_sig[["open", "high", "low", "close"]])
    bullish_zone = nearest_unfilled_zone(zones, price, direction="bullish")
    bearish_zone = nearest_unfilled_zone(zones, price, direction="bearish")

    if bullish_zone and bullish_zone.bottom <= price <= bullish_zone.top * 1.002:
        return StrategySignal("FVG Reaction", 1,
                               f"Price inside unfilled bullish FVG [{bullish_zone.bottom:.2f}, {bullish_zone.top:.2f}]")
    if bearish_zone and bearish_zone.bottom * 0.998 <= price <= bearish_zone.top:
        return StrategySignal("FVG Reaction", -1,
                               f"Price inside unfilled bearish FVG [{bearish_zone.bottom:.2f}, {bearish_zone.top:.2f}]")
    return StrategySignal("FVG Reaction", 0, "Price not currently inside a relevant unfilled FVG")


def vwap_fvg_confluence_strategy(vwap_signal: StrategySignal, fvg_signal: StrategySignal) -> StrategySignal:
    """
    Bonus confluence signal: VWAP and FVG Reaction only fire together when
    price is on the correct side of VWAP *and* sitting inside an unfilled
    gap in the same direction — a stronger setup than either signal alone.
    """
    if vwap_signal.signal == 1 and fvg_signal.signal == 1:
        return StrategySignal("VWAP + FVG Confluence", 1,
                               "Price above VWAP AND inside an unfilled bullish FVG — high-conviction long setup")
    if vwap_signal.signal == -1 and fvg_signal.signal == -1:
        return StrategySignal("VWAP + FVG Confluence", -1,
                               "Price below VWAP AND inside an unfilled bearish FVG — high-conviction short/exit setup")
    return StrategySignal("VWAP + FVG Confluence", 0, "VWAP and FVG signals don't currently agree")


def evaluate_all_strategies(df: pd.DataFrame) -> tuple[list[StrategySignal], pd.DataFrame]:
    """Run every strategy against the latest bar of df. Returns (signals, indicator_dataframe)."""
    sig_df = compute_indicators(df)
    row = sig_df.iloc[-1]
    price = float(row.close)

    vwap_sig = vwap_strategy(row)
    fvg_sig = fvg_strategy(sig_df, row, price)

    signals = [
        rsi_strategy(row),
        ema_trend_strategy(row),
        sma_trend_strategy(row),
        macd_strategy(row),
        bollinger_strategy(row),
        vwap_sig,
        fvg_sig,
        vwap_fvg_confluence_strategy(vwap_sig, fvg_sig),
    ]
    return signals, sig_df


def consensus(signals: list[StrategySignal]) -> dict:
    """
    Core consensus uses the 7 primary strategies (excludes the VWAP+FVG
    confluence bonus signal, since it's derived from two signals already
    in the sum and would double-count them).
    """
    core = [s for s in signals if s.name != "VWAP + FVG Confluence"]
    total = sum(s.signal for s in core)
    bullish = sum(1 for s in core if s.signal == 1)
    bearish = sum(1 for s in core if s.signal == -1)
    neutral = sum(1 for s in core if s.signal == 0)
    n = len(core)

    buy_threshold = 3   # ~43% of 7 strategies agreeing bullish
    sell_threshold = -3

    if total >= buy_threshold:
        verdict = "BUY"
    elif total <= sell_threshold:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    confluence = next((s for s in signals if s.name == "VWAP + FVG Confluence"), None)
    high_conviction = bool(confluence and confluence.signal != 0)

    return dict(score=total, max_score=n, bullish=bullish, bearish=bearish, neutral=neutral,
                verdict=verdict, high_conviction=high_conviction)


def evaluate_history(df: pd.DataFrame, buy_threshold: int = 3, sell_threshold: int = -3) -> pd.DataFrame:
    """
    Vectorized per-bar scoring across the full price history, used to plot
    buy/sell markers on the chart. Combines 6 fully vectorizable strategies
    (RSI, EMA trend, SMA trend, MACD, Bollinger, VWAP). FVG reaction is
    point-in-time (depends on which gaps were still unfilled at that
    moment) and is shown separately as shaded zones on the chart.
    """
    sig = compute_indicators(df)

    rsi_sig = pd.Series(0, index=sig.index)
    rsi_sig[sig.rsi < 30] = 1
    rsi_sig[sig.rsi > 70] = -1

    ema_sig = pd.Series(0, index=sig.index)
    ema_up = (sig.ema_fast > sig.ema_slow) & (sig.close > sig.ema_fast)
    ema_down = (sig.ema_fast < sig.ema_slow) & (sig.close < sig.ema_fast)
    ema_sig[ema_up] = 1
    ema_sig[ema_down] = -1

    sma_sig = pd.Series(0, index=sig.index)
    sma_sig[sig.close > sig.sma_50] = 1
    sma_sig[sig.close < sig.sma_50] = -1

    macd_sig = pd.Series(0, index=sig.index)
    macd_up = (sig.macd_hist > 0) & (sig.macd > sig.macd_signal)
    macd_down = (sig.macd_hist < 0) & (sig.macd < sig.macd_signal)
    macd_sig[macd_up] = 1
    macd_sig[macd_down] = -1

    bb_sig = pd.Series(0, index=sig.index)
    bb_sig[sig.close <= sig.bb_lower] = 1
    bb_sig[sig.close >= sig.bb_upper] = -1

    vwap_sig = pd.Series(0, index=sig.index)
    vwap_sig[sig.close > sig.vwap] = 1
    vwap_sig[sig.close < sig.vwap] = -1

    sig["score"] = rsi_sig + ema_sig + sma_sig + macd_sig + bb_sig + vwap_sig
    sig["verdict"] = "HOLD"
    sig.loc[sig["score"] >= buy_threshold, "verdict"] = "BUY"
    sig.loc[sig["score"] <= sell_threshold, "verdict"] = "SELL"

    prev_verdict = sig["verdict"].shift(1)
    sig["buy_marker"] = (sig["verdict"] == "BUY") & (prev_verdict != "BUY")
    sig["sell_marker"] = (sig["verdict"] == "SELL") & (prev_verdict != "SELL")

    return sig


def build_trade_plan(signals: list[StrategySignal], cons: dict, price: float, plan) -> dict:
    """
    Plain-language entry/exit plan combining the strategy table and the
    ATR risk plan into "when to buy" / "when to sell" guidance.
    """
    bullish_reasons = [s.reason for s in signals if s.signal == 1 and s.name != "VWAP + FVG Confluence"]
    bearish_reasons = [s.reason for s in signals if s.signal == -1 and s.name != "VWAP + FVG Confluence"]

    if cons["verdict"] == "BUY":
        entry_text = (f"Conditions currently support a BUY: {', '.join(bullish_reasons)}. "
                      f"Suggested entry near the current price of {price:,.2f}.")
    elif cons["verdict"] == "SELL":
        entry_text = (f"Conditions currently support a SELL/avoid-entry: {', '.join(bearish_reasons)}. "
                      f"Not a buy setup right now.")
    else:
        entry_text = ("No clear consensus right now — strategies are mixed. Consider waiting for "
                       "more strategies to align (a BUY needs a majority bullish) before entering.")

    exit_text = (f"Exit / take profit if price reaches the ATR target of {plan.target:,.2f} "
                 f"(+{plan.reward_per_share:,.2f}/share, ~{(plan.reward_per_share/price*100):.1f}% gain). "
                 f"Cut the loss if price falls to the ATR stop of {plan.stop:,.2f} "
                 f"(-{plan.risk_per_share:,.2f}/share, ~{(plan.risk_per_share/price*100):.1f}% loss). "
                 f"Also exit on a trend flip (EMA20 crossing below EMA50) or RSI moving above 70.")

    return dict(entry_text=entry_text, exit_text=exit_text)


# ----------------------------------------------------------------------------
# Systematic gated pipeline (the flowchart):
#
#   MARKET -> TREND FILTER (EMA50/200) -> PRICE STRUCTURE (support/higher low)
#   -> VWAP -> FVG/PULLBACK -> RSI momentum>50 -> VOLUME (RVOL>1)
#   -> ENTRY TRIGGER -> ATR/STRUCTURE STOP LOSS -> POSITION SIZING
#   -> 2R TARGET -> EXIT
#
# Unlike the weighted consensus above (which counts how many strategies
# agree), this is a strict AND-gate: every stage must pass for a BUY.
# That's intentional — it's a *systematic* entry checklist, not a vote.
# ----------------------------------------------------------------------------

@dataclass
class PipelineStage:
    name: str
    passed: bool
    detail: str


def run_systematic_pipeline(sig_df: pd.DataFrame, zones, support, resistance) -> dict:
    row = sig_df.iloc[-1]
    price = float(row.close)
    stages: list[PipelineStage] = []

    # 1. Trend Filter — EMA50 / EMA200
    trend_ok = (not pd.isna(row.ema_200)) and row.ema_slow > row.ema_200 and price > row.ema_slow
    stages.append(PipelineStage(
        "Trend Filter (EMA50/200)", trend_ok,
        f"EMA50 {row.ema_slow:.2f} vs EMA200 {row.ema_200:.2f}, price {price:.2f}"
        if not pd.isna(row.ema_200) else "Not enough bars for EMA200 yet"
    ))

    # 2. Price Structure — price holding above the nearest identified support (a "higher low")
    structure_ok = support is not None and price > support
    stages.append(PipelineStage(
        "Price Structure (Support / Higher Low)", structure_ok,
        f"Nearest support {support:.2f}, price {price:.2f} {'above' if structure_ok else 'below or no support found'}"
        if support is not None else "No clear recent support level identified"
    ))

    # 3. VWAP
    vwap_ok = (not pd.isna(row.vwap)) and price > row.vwap
    stages.append(PipelineStage(
        "VWAP", vwap_ok,
        f"Price {price:.2f} vs VWAP {row.vwap:.2f}" if not pd.isna(row.vwap) else "Not enough bars for VWAP yet"
    ))

    # 4. FVG / Pullback — price retracing into (or near) an unfilled bullish FVG, or near VWAP/EMA20 pullback
    bullish_zone = nearest_unfilled_zone(zones, price, direction="bullish")
    pullback_ok = bool(bullish_zone and bullish_zone.bottom <= price <= bullish_zone.top * 1.01) or \
        (abs(price - row.vwap) / price < 0.005 if not pd.isna(row.vwap) else False) or \
        (abs(price - row.ema_fast) / price < 0.005)
    detail = (f"Inside unfilled bullish FVG [{bullish_zone.bottom:.2f}, {bullish_zone.top:.2f}]" if bullish_zone
              and bullish_zone.bottom <= price <= bullish_zone.top * 1.01
              else "Price near VWAP/EMA20 pullback zone" if pullback_ok else "No pullback/FVG confluence right now")
    stages.append(PipelineStage("FVG / Pullback", pullback_ok, detail))

    # 5. RSI momentum > 50
    rsi_ok = (not pd.isna(row.rsi)) and row.rsi > 50
    stages.append(PipelineStage("RSI Momentum > 50", rsi_ok, f"RSI {row.rsi:.1f}" if not pd.isna(row.rsi) else "n/a"))

    # 6. Volume — RVOL > 1
    rvol_ok = (not pd.isna(row.rvol)) and row.rvol > 1.0
    stages.append(PipelineStage("Volume (RVOL > 1)", rvol_ok,
                                 f"RVOL {row.rvol:.2f}x average" if not pd.isna(row.rvol) else "Not enough bars for RVOL yet"))

    all_passed = all(s.passed for s in stages)
    stages.append(PipelineStage("ENTRY TRIGGER", all_passed,
                                 "All 6 gates passed — systematic long entry confirmed" if all_passed
                                 else f"{sum(s.passed for s in stages)}/6 gates passed — entry NOT confirmed"))

    return dict(stages=stages, entry_confirmed=all_passed, price=price)
