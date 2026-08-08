# Trading Signal Terminal

A real-time, multi-indicator trading dashboard with a systematic gated entry
pipeline (matching the flowchart below), full ATR/structure-based risk
management driven by an investment amount you set, a portfolio analysis tab
for pasted brokerage holdings, and a Market Movers screener with click-to-chart.

**This is a market-data and planning tool only.** It has no connection to
any real brokerage account (Interactive Brokers, Wealthsimple, or otherwise)
— it never places orders and can't see your real account. The top ticker
bar and dashboard layout are inspired by professional trading terminals in
general, not a clone of any specific broker's proprietary UI or branding.

## Files

| File | Purpose |
|---|---|
| `app.py` | The application — run with `streamlit run app.py` |
| `strategies.py` | 8 confluence strategies + the **systematic gated pipeline** (the flowchart) + trade-plan text generator |
| `risk_management.py` | ATR + structure stop-loss, 2R target, investment-amount-based position sizing, volatility regime, chandelier trailing stop |
| `market_movers.py` | Curated liquid-stock and penny-stock screener, ranked by real-time volume/change |
| `fvg.py` | Fair Value Gap detection (bullish/bearish 3-candle imbalance) + fill tracking |
| `portfolio.py` | Parses pasted holdings text (structured or freeform) into symbol/shares/avg_cost |
| `indicators.py` | TA-Lib wrapper: RSI, EMA, SMA, MACD, Bollinger, ATR, VWAP, RVOL, support/resistance |
| `data_feed.py` | Real-time data — `ccxt`/Binance for crypto, `yfinance` for stocks, plus index quotes and fundamentals (P/E, market cap) |
| `requirements.txt` | Dependencies |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` with three tabs.

## The systematic entry pipeline

This is the flowchart, implemented as a literal **AND-gate checklist**
(every stage must pass — this is stricter than a weighted vote):

```
MARKET → TREND FILTER (EMA50/200) → PRICE STRUCTURE (Support/Higher Low)
       → VWAP → FVG/PULLBACK → RSI MOMENTUM>50 → VOLUME (RVOL>1)
       → ENTRY TRIGGER → ATR/STRUCTURE STOP LOSS → POSITION SIZING
       → 2R TARGET → EXIT
```

Each stage renders as a green ✅ or red ❌ card in the **Live Signals** and
**Market Movers** tabs, so you can see exactly which condition is (or
isn't) currently met — not just a final score.

### 📈 Live Signals tab
Pick any stock or crypto symbol:
- Full candlestick chart: EMA20/50/200, SMA50, VWAP, support/resistance
  lines, shaded FVG zones, ATR trailing stop, and ▲/▼ buy/sell markers —
  plus separate RSI, MACD, and ATR panels
- P/E ratio, market cap, 52-week range, sector (stocks only)
- The systematic pipeline checklist
- An 8-strategy confluence table (RSI, EMA trend, SMA trend, MACD,
  Bollinger, VWAP, FVG reaction, VWAP+FVG confluence)
- **Investment-amount-driven risk management**: type in how much you're
  planning to invest, and get — stop-loss (ATR vs structure, whichever is
  safer), 2R (or ATR) target, risk:reward, profit %/loss %, and position
  size computed two ways (risk-based and capital-based, using the more
  conservative of the two)
- A plain-language "when to buy" / "when to sell" trade plan
- Optional auto-refresh

### 💼 My Portfolio tab
Paste your holdings (`SYMBOL, SHARES, AVG_COST` per line, or a best-effort
freeform paste straight from Wealthsimple). For holdings with shares and
avg cost, the investment amount used for risk sizing is your actual cost
basis (shares × avg cost) by default — so stop/target/position-size reflect
your real position, not a hypothetical. Get a summary table, a
portfolio-wide risk summary (total $ at risk, positions below stop), and a
full per-holding breakdown identical to the Live Signals tab.

### 🔥 Market Movers tab
"Most Active" and "Penny Stocks (<$5)" lists, built from a curated,
liquid universe (not a full-market scanner — no free API provides a true
real-time whole-market screener) and ranked by real-time volume/RVOL/%
change. **Click a row** to load that symbol's full chart, pipeline, and
risk plan below the table.

## Risk management details

- **Stop-loss**: the more conservative of an ATR-based stop
  (entry − ATR multiplier × ATR) and a structure-based stop (nearest swing
  support) — whichever is further from entry, so it only triggers on a
  genuine break, not noise
- **Target**: 2R by default (2× the risk distance — the classic
  systematic-trading convention), with an ATR-based target also computed
  for comparison; toggle between them
- **Position sizing**: computed two ways and the smaller (safer) one used —
  risk-based (risk a fixed % of your investment amount, sized off the stop
  distance) and capital-based (simply how many shares your investment
  amount affords at entry)
- **Profit % / Loss %**: the target/stop distance as a % of entry
- **Volatility regime**: current ATR's percentile rank vs its recent history
- **Chandelier Exit**: a ratcheting ATR trailing stop plotted on the chart

## Notes

- Tested end-to-end in this sandbox: all indicators (including the new
  support/resistance, RVOL, and VWAP), the systematic pipeline logic, the
  ATR/structure risk plan (including 2R targets and dual-method sizing),
  the market movers screener's error handling, and a full module-level run
  of `app.py` (all three tabs) plus a live Streamlit server boot — all
  confirmed working with clean HTTP 200 and no exceptions, even when real
  network calls failed (handled gracefully with per-symbol try/except).
  Real Yahoo Finance/Binance endpoints aren't reachable from this sandbox,
  so live data itself needs a first-run check on your own machine.
- The Market Movers universe is a curated list of ~30 liquid large-caps and
  ~20 commonly-traded penny stocks — it will miss real-time movers outside
  that list, and the penny-stock list can go stale as prices cross $5.
- P/E ratio and other fundamentals aren't available for crypto pairs.
- Not financial advice — this is a signal/analysis and planning tool;
  validate against your own judgment and risk tolerance before trading.
