"""
app.py
------
Real-time multi-indicator trading dashboard: ATR, RSI, EMA20/50/200, SMA50,
VWAP, Fair Value Gaps, support/resistance, MACD, P/E, and a systematic
gated entry pipeline (Trend Filter -> Price Structure -> VWAP -> FVG/Pullback
-> RSI Momentum -> Volume -> Entry Trigger -> Stop Loss -> Position Sizing
-> 2R Target -> Exit), plus a Market Movers screener and a Portfolio tab
for pasted brokerage holdings.

Run with:
    streamlit run app.py

This is a market-data and planning tool only — it has no connection to any
real brokerage account (Interactive Brokers, Wealthsimple, or otherwise).
Any "account"-style figures shown are derived purely from the investment
amount you type in, for planning purposes.
"""

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_feed import fetch_ohlcv, fetch_index_quotes, fetch_fundamentals
from fvg import detect_fair_value_gaps, zones_to_dataframe
from strategies import evaluate_all_strategies, consensus, evaluate_history, build_trade_plan, run_systematic_pipeline
from portfolio import parse_holdings
from risk_management import build_risk_plan, volatility_regime, chandelier_exit, stop_breached, target_reached
from market_movers import get_active_stocks, get_penny_stocks
import indicators as ta

st.set_page_config(page_title="Trading Signal Terminal", layout="wide", page_icon="📊")

VERDICT_ICON = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD"}

# ----------------------------------------------------------------------------
# Top ticker bar styling (dashboard look, inspired by pro trading terminals —
# not a clone of any specific broker's proprietary UI/branding)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.ticker-bar {
    background-color: #0b0e1a; padding: 10px 20px; border-radius: 6px;
    display: flex; gap: 32px; margin-bottom: 12px; flex-wrap: wrap;
}
.ticker-item { color: #e8e8e8; font-size: 14px; }
.ticker-item b { color: #fff; }
.ticker-up { color: #2ecc71; }
.ticker-down { color: #e74c3c; }
.stage-pass { background-color: #1e3a2e; color: #4ade80; padding: 8px 14px;
    border-radius: 6px; margin-bottom: 6px; border-left: 4px solid #22c55e; }
.stage-fail { background-color: #3a1e1e; color: #f87171; padding: 8px 14px;
    border-radius: 6px; margin-bottom: 6px; border-left: 4px solid #ef4444; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30, show_spinner=False)
def load_index_quotes():
    try:
        return fetch_index_quotes()
    except Exception:
        return []


def render_ticker_bar():
    quotes = load_index_quotes()
    if not quotes:
        return
    items = []
    for q in quotes:
        if q["price"] is None:
            continue
        cls = "ticker-up" if (q["change_pct"] or 0) >= 0 else "ticker-down"
        arrow = "▲" if (q["change_pct"] or 0) >= 0 else "▼"
        items.append(f'<span class="ticker-item"><b>{q["label"]}</b> {q["price"]:,.2f} '
                      f'<span class="{cls}">{arrow} {q["change_pct"]:+.2f}%</span></span>')
    st.markdown(f'<div class="ticker-bar">{"".join(items)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Shared data + analysis (used by all three tabs)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=15, show_spinner=False)
def load_data(asset_type, symbol, timeframe, limit):
    return fetch_ohlcv(asset_type, symbol, timeframe=timeframe, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def load_fundamentals(symbol):
    return fetch_fundamentals(symbol)


def analyze_symbol(asset_type: str, symbol: str, timeframe: str, limit: int, min_gap_pct: float,
                    investment_amount: float = 10_000, risk_pct: float = 1.0,
                    atr_stop_mult: float = 2.0, r_multiple: float = 2.0,
                    use_structure_stop: bool = True, target_basis: str = "2R",
                    entry_price: float = None):
    """Fetch + fully analyze one symbol: strategies, FVG, systematic pipeline, and risk plan."""
    try:
        df = load_data(asset_type, symbol, timeframe, limit)
    except Exception as e:
        return {"error": str(e)}

    if df is None or df.empty or len(df) < 60:
        return {"error": "not enough bars returned"}

    signals, sig_df = evaluate_all_strategies(df)
    cons = consensus(signals)
    zones = detect_fair_value_gaps(df[["open", "high", "low", "close"]], min_gap_pct=min_gap_pct)
    history = evaluate_history(df)
    support, resistance = ta.support_resistance(df)
    pipeline = run_systematic_pipeline(sig_df, zones, support, resistance)

    last_price = float(sig_df["close"].iloc[-1])
    last_atr = float(sig_df["atr"].iloc[-1])
    ref_price = entry_price if entry_price else last_price

    risk_plan = build_risk_plan(ref_price, last_atr, investment_amount, risk_pct,
                                 atr_stop_mult=atr_stop_mult, r_multiple=r_multiple,
                                 structure_support=support, use_structure_stop=use_structure_stop,
                                 target_basis=target_basis)
    vol_regime = volatility_regime(sig_df["atr"])
    trailing_stop = chandelier_exit(df)
    trade_plan = build_trade_plan(signals, cons, ref_price, risk_plan)

    fundamentals = load_fundamentals(symbol) if asset_type == "stock" else {}

    return dict(
        symbol=symbol, df=df, sig_df=sig_df, signals=signals, consensus=cons,
        zones=zones, history=history, last_price=last_price, last_rsi=float(sig_df["rsi"].iloc[-1]),
        last_atr=last_atr, risk_plan=risk_plan, vol_regime=vol_regime, trailing_stop=trailing_stop,
        trade_plan=trade_plan, support=support, resistance=resistance, pipeline=pipeline,
        fundamentals=fundamentals,
    )


def build_chart(result: dict, timeframe: str, show_filled: bool) -> go.Figure:
    sig_df, zones, history = result["sig_df"], result["zones"], result["history"]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.17, 0.17, 0.16],
                         vertical_spacing=0.025,
                         subplot_titles=(f"{result['symbol']} — {timeframe}", "RSI (14)", "MACD", "ATR (14)"))

    fig.add_trace(go.Candlestick(
        x=sig_df.index, open=sig_df.open, high=sig_df.high, low=sig_df.low, close=sig_df.close, name="Price"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.ema_fast, name="EMA 20",
                              line=dict(width=1, color="dodgerblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.ema_slow, name="EMA 50",
                              line=dict(width=1, color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.ema_200, name="EMA 200",
                              line=dict(width=1.5, color="purple")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.sma_50, name="SMA 50",
                              line=dict(width=1, color="gray", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.vwap, name="VWAP",
                              line=dict(width=1.5, color="teal")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result["trailing_stop"].index, y=result["trailing_stop"], name="ATR Trailing Stop",
                              line=dict(width=1.5, color="crimson", dash="dash")), row=1, col=1)

    if result.get("support"):
        fig.add_hline(y=result["support"], line_dash="dot", line_color="green",
                      annotation_text="Support", row=1, col=1)
    if result.get("resistance"):
        fig.add_hline(y=result["resistance"], line_dash="dot", line_color="red",
                      annotation_text="Resistance", row=1, col=1)

    for z in zones:
        if z.filled and not show_filled:
            continue
        color = "rgba(0,200,0,0.18)" if z.direction == "bullish" else "rgba(220,0,0,0.18)"
        line_color = "green" if z.direction == "bullish" else "red"
        fig.add_shape(
            type="rect", x0=z.timestamp, x1=sig_df.index[-1], y0=z.bottom, y1=z.top,
            fillcolor=color, line=dict(color=line_color, width=1, dash="dot" if z.filled else "solid"),
            row=1, col=1,
        )

    buys = history[history["buy_marker"]]
    sells = history[history["sell_marker"]]
    fig.add_trace(go.Scatter(
        x=buys.index, y=buys.low * 0.995, mode="markers", name="BUY signal",
        marker=dict(symbol="triangle-up", size=13, color="lime", line=dict(width=1, color="darkgreen")),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells.high * 1.005, mode="markers", name="SELL signal",
        marker=dict(symbol="triangle-down", size=13, color="red", line=dict(width=1, color="darkred")),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.rsi, name="RSI", line=dict(color="purple")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Bar(x=sig_df.index, y=sig_df.macd_hist, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.macd, name="MACD", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.macd_signal, name="Signal", line=dict(width=1)), row=3, col=1)

    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df.atr, name="ATR", line=dict(color="brown"),
                              fill="tozeroy"), row=4, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.05))
    return fig


def render_pipeline_checklist(pipeline: dict):
    st.markdown("**Systematic Entry Pipeline** _(every gate must pass for a confirmed long entry)_")
    for stage in pipeline["stages"]:
        css = "stage-pass" if stage.passed else "stage-fail"
        icon = "✅" if stage.passed else "❌"
        st.markdown(f'<div class="{css}">{icon} <b>{stage.name}</b> — {stage.detail}</div>',
                    unsafe_allow_html=True)


def render_fundamentals(fundamentals: dict):
    if not fundamentals or fundamentals.get("pe_ratio") is None and fundamentals.get("market_cap") is None:
        return
    f1, f2, f3, f4 = st.columns(4)
    pe = fundamentals.get("pe_ratio")
    f1.metric("P/E Ratio", f"{pe:.2f}" if pe else "n/a")
    mc = fundamentals.get("market_cap")
    f2.metric("Market Cap", f"${mc/1e9:.2f}B" if mc else "n/a")
    hi, lo = fundamentals.get("fifty_two_week_high"), fundamentals.get("fifty_two_week_low")
    f3.metric("52wk Range", f"{lo:.2f} – {hi:.2f}" if hi and lo else "n/a")
    f4.metric("Sector", fundamentals.get("sector") or "n/a")


def render_symbol_panel(result: dict, timeframe: str, show_filled: bool, holding: dict = None):
    """Renders the full analysis for one already-analyzed symbol."""
    cons = result["consensus"]
    price = result["last_price"]

    name = result["fundamentals"].get("name") if result.get("fundamentals") else None
    st.subheader(f"{result['symbol']}" + (f" — {name}" if name else ""))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"{price:,.4f}" if price < 10 else f"{price:,.2f}")
    c2.metric("RSI (14)", f"{result['last_rsi']:.1f}")
    c3.metric("Consensus", f"{cons['score']:+d} / {cons['max_score']}")
    c4.metric("Verdict", VERDICT_ICON[cons["verdict"]] + (" 🔥" if cons.get("high_conviction") else ""))
    c5.metric("Pipeline", "✅ ENTRY CONFIRMED" if result["pipeline"]["entry_confirmed"] else "⏳ Waiting")

    render_fundamentals(result.get("fundamentals", {}))

    if holding and holding.get("avg_cost"):
        shares = holding.get("shares") or 0
        avg_cost = holding["avg_cost"]
        pnl = (price - avg_cost) * shares
        pnl_pct = (price / avg_cost - 1) * 100
        h1, h2, h3 = st.columns(3)
        h1.metric("Your Avg Cost", f"{avg_cost:,.2f}")
        h2.metric("Shares Held", f"{shares:g}")
        h3.metric("Unrealized P/L", f"{pnl:,.2f}", f"{pnl_pct:+.2f}%")

    st.plotly_chart(build_chart(result, timeframe, show_filled), use_container_width=True)
    st.caption("Purple = EMA200 · gray dotted = SMA50 · teal = VWAP · green/red dotted lines = support/resistance · "
               "🟩/🟥 shaded = unfilled FVG · ▲/▼ = buy/sell signal · dashed red = ATR trailing stop")

    render_pipeline_checklist(result["pipeline"])

    st.markdown("**Strategy Signals (weighted confluence)**")
    icon = {1: "🟢 Bullish", 0: "⚪ Neutral", -1: "🔴 Bearish"}
    rows = [{"Strategy": s.name, "Signal": icon[s.signal], "Reason": s.reason} for s in result["signals"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Risk Management & Position Sizing**")
    plan = result["risk_plan"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ATR (14)", f"{result['last_atr']:.4f}" if result['last_atr'] < 10 else f"{result['last_atr']:.2f}")
    r2.metric("Stop Loss", f"{plan.stop:,.2f}", f"{plan.stop_basis}")
    r3.metric("Target", f"{plan.target:,.2f}", f"{plan.target_basis}")
    r4.metric("Risk:Reward", f"1 : {plan.risk_reward:.2f}" if plan.risk_reward == plan.risk_reward else "n/a")

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("Loss % (if stopped)", f"-{plan.loss_pct:.1f}%")
    r6.metric("Profit % (if target hit)", f"+{plan.profit_pct:.1f}%")
    r7.metric("$ at Risk", f"{plan.dollar_risk:,.2f}")
    r8.metric("$ Potential Reward", f"{plan.dollar_reward:,.2f}")

    r9, r10, r11, r12 = st.columns(4)
    r9.metric("Risk-Based Size", f"{plan.risk_based_size:,.2f} sh")
    r10.metric("Capital-Based Size", f"{plan.capital_based_size:,.2f} sh")
    r11.metric("Suggested Size (min of both)", f"{plan.position_size:,.2f} sh")
    r12.metric("Volatility Regime", result["vol_regime"])
    st.caption(f"ATR stop: {plan.atr_stop:,.2f} · Structure stop: "
               f"{plan.structure_stop:,.2f}" if plan.structure_stop else f"ATR stop: {plan.atr_stop:,.2f} · Structure stop: n/a"
               f" · 2R target: {plan.r_target:,.2f} · ATR target: {plan.atr_target:,.2f}")

    st.markdown("**Trade Plan**")
    tp = result["trade_plan"]
    st.markdown(f"🟢 **When to buy:** {tp['entry_text']}")
    st.markdown(f"🔴 **When to sell:** {tp['exit_text']}")

    if holding and holding.get("avg_cost"):
        if stop_breached(price, plan.stop):
            st.error(f"⚠️ Price ({price:,.2f}) is at/below the stop ({plan.stop:,.2f}) computed from your entry.")
        elif target_reached(price, plan.target):
            st.success(f"🎯 Price ({price:,.2f}) has reached the target ({plan.target:,.2f}) from your entry.")

    with st.expander("Fair Value Gaps"):
        fvg_table = zones_to_dataframe(result["zones"]).sort_values("timestamp", ascending=False)
        if not show_filled:
            fvg_table = fvg_table[~fvg_table["filled"]]
        st.dataframe(fvg_table, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------------
render_ticker_bar()
st.title("📊 Trading Signal Terminal")
st.caption("ATR · RSI · EMA20/50/200 · SMA · VWAP · Fair Value Gaps · MACD · P/E · Support/Resistance — "
           "a systematic entry pipeline, real-time stocks & crypto, and portfolio risk analysis")

tab_live, tab_portfolio, tab_movers = st.tabs(["📈 Live Signals", "💼 My Portfolio", "🔥 Market Movers"])

# ----------------------------------------------------------------------------
# TAB 1 — Live single-symbol signals
# ----------------------------------------------------------------------------
with tab_live:
    with st.sidebar:
        st.header("Market Settings")
        asset_type = st.selectbox("Asset Type", ["crypto", "stock"], key="live_asset_type")
        default_symbol = "BTC/USDT" if asset_type == "crypto" else "AAPL"
        symbol = st.text_input("Symbol", value=default_symbol, key="live_symbol",
                                help="Crypto e.g. BTC/USDT, ETH/USDT · Stock e.g. AAPL, TSLA, MSFT")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=1, key="live_tf")
        limit = st.slider("Bars to load", 100, 500, 300, 50, key="live_limit")

        st.divider()
        st.header("FVG Settings")
        min_gap_pct = st.slider("Min gap size (%)", 0.0, 1.0, 0.05, 0.05, key="live_gap")
        show_filled = st.checkbox("Show filled FVGs too", value=False, key="live_filled")

        st.divider()
        st.header("Investment & Risk")
        investment_amount = st.number_input("Investment Amount ($)", value=10_000, step=500, key="live_inv")
        risk_pct = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1, key="live_risk_pct")
        atr_stop_mult = st.slider("ATR Stop Multiplier", 0.5, 5.0, 2.0, 0.5, key="live_stop_mult")
        r_multiple = st.slider("Target R-Multiple (2R default)", 1.0, 5.0, 2.0, 0.5, key="live_r_mult")
        target_basis = st.radio("Target Basis", ["2R", "ATR"], horizontal=True, key="live_target_basis")
        use_structure_stop = st.checkbox("Use structure (support) stop if safer", value=True, key="live_struct_stop")

        st.divider()
        st.header("Live Updates")
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="live_refresh")
        refresh_secs = st.slider("Refresh every (sec)", 10, 120, 30, disabled=not auto_refresh, key="live_refresh_s")
        st.button("🔄 Refresh Now", use_container_width=True, key="live_refresh_btn")

    result = analyze_symbol(asset_type, symbol, timeframe, limit, min_gap_pct,
                             investment_amount=investment_amount, risk_pct=risk_pct,
                             atr_stop_mult=atr_stop_mult, r_multiple=r_multiple,
                             use_structure_stop=use_structure_stop, target_basis=target_basis)
    if result.get("error"):
        st.warning(f"Could not analyze {symbol}: {result['error']}")
    else:
        render_symbol_panel(result, timeframe, show_filled)
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if auto_refresh:
        time.sleep(refresh_secs)
        st.cache_data.clear()
        st.rerun()

# ----------------------------------------------------------------------------
# TAB 2 — Portfolio analysis
# ----------------------------------------------------------------------------
with tab_portfolio:
    st.subheader("Paste your holdings")
    st.caption(
        "Most reliable format — one line per holding: `SYMBOL, SHARES, AVG_COST` "
        "(shares/avg cost optional), e.g.:\n\n"
        "```\nAAPL, 10, 150.00\nSHOP.TO, 5, 72.50\nBTC/USDT, 0.05\n```\n\n"
        "Pasting directly from the Wealthsimple app also works on a best-effort basis — "
        "always check the parsed table before trusting it."
    )
    holdings_text = st.text_area("Holdings", height=180, placeholder="AAPL, 10, 150.00\nMSFT, 4, 310.00\n...")

    pc1, pc2, pc3 = st.columns(3)
    portfolio_timeframe = pc1.selectbox("Timeframe", ["1d", "1h", "15m", "5m"], index=0, key="port_tf")
    portfolio_limit = pc2.slider("Bars per symbol", 100, 500, 300, 50, key="port_limit")
    portfolio_gap = pc3.slider("Min FVG gap (%)", 0.0, 1.0, 0.05, 0.05, key="port_gap")

    with st.expander("Risk Management Settings", expanded=False):
        rc1, rc2 = st.columns(2)
        default_investment = rc1.number_input(
            "Default Investment Amount ($) — used for holdings with no avg cost, or for sizing a new entry",
            value=10_000, step=500, key="port_default_inv")
        use_cost_basis = rc2.checkbox(
            "For holdings with shares & avg cost, use (shares × avg cost) as the investment amount",
            value=True, key="port_use_cost_basis")
        rc3, rc4, rc5, rc6 = st.columns(4)
        port_risk_pct = rc3.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1, key="port_risk_pct")
        port_stop_mult = rc4.slider("ATR Stop Multiplier", 0.5, 5.0, 2.0, 0.5, key="port_stop_mult")
        port_r_mult = rc5.slider("Target R-Multiple", 1.0, 5.0, 2.0, 0.5, key="port_r_mult")
        port_use_struct = rc6.checkbox("Use structure stop", value=True, key="port_use_struct")

    analyze_clicked = st.button("🔍 Analyze Portfolio", type="primary")

    if analyze_clicked:
        holdings_df = parse_holdings(holdings_text)
        if holdings_df.empty:
            st.error("Couldn't find any recognizable ticker symbols in the pasted text. "
                      "Try the structured `SYMBOL, SHARES, AVG_COST` format instead.")
        else:
            st.session_state["portfolio_holdings"] = holdings_df
            st.session_state["portfolio_results"] = {}
            progress = st.progress(0.0, text="Analyzing holdings...")
            for i, row in holdings_df.iterrows():
                asset_type_guess = "crypto" if "/" in row.symbol else "stock"
                has_cost_basis = use_cost_basis and pd.notna(row.avg_cost) and pd.notna(row.shares) and row.avg_cost and row.shares
                inv_amount = (row.shares * row.avg_cost) if has_cost_basis else default_investment
                res = analyze_symbol(
                    asset_type_guess, row.symbol, portfolio_timeframe, portfolio_limit, portfolio_gap,
                    investment_amount=inv_amount, risk_pct=port_risk_pct,
                    atr_stop_mult=port_stop_mult, r_multiple=port_r_mult,
                    use_structure_stop=port_use_struct, target_basis="2R",
                    entry_price=row.avg_cost if pd.notna(row.avg_cost) else None,
                )
                res["asset_type"] = asset_type_guess
                st.session_state["portfolio_results"][row.symbol] = res
                progress.progress((i + 1) / len(holdings_df), text=f"Analyzed {row.symbol}")
            progress.empty()

    if "portfolio_results" in st.session_state:
        holdings_df = st.session_state["portfolio_holdings"]
        results = st.session_state["portfolio_results"]

        st.markdown("### Parsed Holdings")
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

        st.markdown("### Portfolio Summary")
        summary_rows = []
        total_dollar_risk = 0.0
        total_position_value = 0.0
        stopped_out_symbols = []

        for _, row in holdings_df.iterrows():
            res = results.get(row.symbol, {})
            if res.get("error"):
                summary_rows.append({"Symbol": row.symbol, "Price": "—", "RSI": "—",
                                      "Verdict": "⚠️ " + res["error"], "Unrealized P/L": "—",
                                      "Entry": "—", "Stop": "—", "Target": "—", "R:R": "—",
                                      "Profit %": "—", "Loss %": "—", "Suggested Size": "—"})
                continue
            price = res["last_price"]
            plan = res["risk_plan"]
            pnl_str = "—"
            if pd.notna(row.avg_cost) and row.avg_cost:
                shares = row.shares or 0
                pnl = (price - row.avg_cost) * shares
                pnl_pct = (price / row.avg_cost - 1) * 100
                pnl_str = f"{pnl:+,.2f} ({pnl_pct:+.2f}%)"
                if pd.notna(row.shares) and row.shares:
                    total_position_value += price * shares
                    total_dollar_risk += shares * plan.risk_per_share
                if stop_breached(price, plan.stop):
                    stopped_out_symbols.append(row.symbol)

            summary_rows.append({
                "Symbol": row.symbol,
                "Price": round(price, 2),
                "RSI": round(res["last_rsi"], 1),
                "Verdict": VERDICT_ICON[res["consensus"]["verdict"]] + (" 🔥" if res["consensus"].get("high_conviction") else ""),
                "Pipeline": "✅" if res["pipeline"]["entry_confirmed"] else "⏳",
                "Unrealized P/L": pnl_str,
                "Entry": round(plan.entry, 2),
                "Stop": round(plan.stop, 2),
                "Target": round(plan.target, 2),
                "R:R": f"1:{plan.risk_reward:.2f}" if plan.risk_reward == plan.risk_reward else "n/a",
                "Profit %": f"+{plan.profit_pct:.1f}%",
                "Loss %": f"-{plan.loss_pct:.1f}%",
                "Suggested Size": f"{plan.position_size:,.2f} sh",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.caption("Entry = your avg cost (if provided) or current price · Suggested Size uses the investment "
                   "amount settings above (cost basis or default), not necessarily your existing share count.")

        st.markdown("### Portfolio Risk Summary")
        pr1, pr2, pr3 = st.columns(3)
        pr1.metric("Total Position Value", f"${total_position_value:,.2f}")
        pr2.metric("Total $ at Risk (all stops hit)", f"${total_dollar_risk:,.2f}",
                   f"{(total_dollar_risk/total_position_value*100):.1f}% of positions"
                   if total_position_value else None)
        pr3.metric("Positions Below Stop", len(stopped_out_symbols))
        if stopped_out_symbols:
            st.error(f"⚠️ These holdings are currently trading at/below their stop (from your avg cost): "
                      f"**{', '.join(stopped_out_symbols)}**")

        st.markdown("### Per-Holding Detail")
        for _, row in holdings_df.iterrows():
            res = results.get(row.symbol, {})
            label = row.symbol
            if not res.get("error"):
                label += f" — {VERDICT_ICON[res['consensus']['verdict']]}"
            else:
                label += " — ⚠️ error"
            with st.expander(label):
                if res.get("error"):
                    st.warning(f"Could not analyze {row.symbol}: {res['error']}")
                else:
                    holding_info = {"shares": row.shares, "avg_cost": row.avg_cost}
                    render_symbol_panel(res, portfolio_timeframe, show_filled=False, holding=holding_info)
    else:
        st.info("Paste your holdings above and click **Analyze Portfolio**.")

# ----------------------------------------------------------------------------
# TAB 3 — Market Movers (curated screener, click a row to chart it)
# ----------------------------------------------------------------------------
with tab_movers:
    st.subheader("Market Movers")
    st.caption("Built from a curated, liquid stock/penny-stock universe ranked by real-time volume and "
               "% change — not a full-market scanner (no free API provides one). Click a row to load its "
               "full analysis below.")

    mover_type = st.radio("Universe", ["Most Active", "Penny Stocks (<$5)"], horizontal=True, key="mover_type")
    refresh_movers = st.button("🔄 Refresh Movers", key="movers_refresh")

    @st.cache_data(ttl=30, show_spinner=True)
    def load_movers(kind):
        return get_active_stocks(20) if kind == "Most Active" else get_penny_stocks(20)

    if refresh_movers:
        st.cache_data.clear()

    movers_df = load_movers(mover_type)

    if movers_df.empty:
        st.warning("No mover data available right now.")
    else:
        display_df = movers_df.copy()
        display_df["price"] = display_df["price"].round(2)
        display_df["change_pct"] = display_df["change_pct"].round(2)
        display_df["rvol"] = display_df["rvol"].round(2)
        display_df.columns = ["Symbol", "Price", "Change %", "Volume", "RVOL"]

        event = st.dataframe(
            display_df, use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="movers_table",
        )

        selected_rows = event.selection.rows if hasattr(event, "selection") else []
        if selected_rows:
            selected_symbol = display_df.iloc[selected_rows[0]]["Symbol"]
            st.divider()
            st.markdown(f"### {selected_symbol} — Full Analysis")

            with st.expander("Analysis Settings", expanded=False):
                mv1, mv2, mv3 = st.columns(3)
                mv_timeframe = mv1.selectbox("Timeframe", ["5m", "15m", "1h", "1d"], index=2, key="mv_tf")
                mv_investment = mv2.number_input("Investment Amount ($)", value=10_000, step=500, key="mv_inv")
                mv_risk_pct = mv3.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1, key="mv_risk")

            mv_result = analyze_symbol("stock", selected_symbol, mv_timeframe, 300, 0.05,
                                        investment_amount=mv_investment, risk_pct=mv_risk_pct)
            if mv_result.get("error"):
                st.warning(f"Could not analyze {selected_symbol}: {mv_result['error']}")
            else:
                render_symbol_panel(mv_result, mv_timeframe, show_filled=False)
        else:
            st.info("Select a row above to see its full chart, systematic pipeline, and risk plan.")
