"""
risk_management.py
-------------------
Risk management matching the systematic pipeline's final stages:

    ATR / STRUCTURE STOP LOSS -> POSITION SIZING -> 2R TARGET -> EXIT

- Stop-loss is the more conservative (further from entry, i.e. lower for a
  long) of an ATR-based stop and a structure-based stop (nearest support),
  when a support level is available.
- Target defaults to a 2R (2x the risk distance) reward, the classic
  systematic-trading convention, with an ATR-based target also computed
  for comparison.
- Position sizing is computed two ways and the more conservative (smaller)
  size is used: (1) risk-based — risk a fixed % of your investment amount,
  sized off the stop distance, and (2) capital-based — simply how many
  shares your investment amount can afford at the entry price.

All strategies in this app are long-only (they buy dips/support, they
don't short), so risk levels here assume a long position unless otherwise noted.
"""

from dataclasses import dataclass

import pandas as pd

import indicators as ta


@dataclass
class RiskPlan:
    entry: float
    atr: float
    stop: float
    stop_basis: str            # "ATR" or "Structure (support)"
    atr_stop: float
    structure_stop: float | None
    target: float
    target_basis: str          # "2R" or "ATR"
    atr_target: float
    r_target: float
    risk_per_share: float
    reward_per_share: float
    risk_reward: float
    risk_based_size: float     # shares sized off risk % of investment amount
    capital_based_size: float  # shares your investment amount can simply afford
    position_size: float       # the more conservative (smaller) of the two
    investment_amount: float
    dollar_risk: float         # $ lost if stop is hit, at this size
    dollar_reward: float       # $ gained if target is hit, at this size
    loss_pct: float            # % loss if stop is hit, relative to entry
    profit_pct: float          # % gain if target is hit, relative to entry


def build_risk_plan(entry_price: float, atr_value: float, investment_amount: float, risk_pct: float,
                     atr_stop_mult: float = 2.0, r_multiple: float = 2.0,
                     structure_support: float | None = None, use_structure_stop: bool = True,
                     target_basis: str = "2R") -> RiskPlan:
    """
    Build the full stop/target/sizing plan for a long position.

    entry_price:        planned or actual entry (avg cost for existing holdings)
    atr_value:           current ATR
    investment_amount:   $ you're planning to allocate to (or already have in) this trade
    risk_pct:             % of investment_amount you're willing to risk on this trade
    atr_stop_mult:        stop = entry - atr_stop_mult * ATR
    r_multiple:           target = entry + r_multiple * risk_per_share (2R by default)
    structure_support:    nearest identified support level, if any
    use_structure_stop:   if True and structure_support is available, use whichever
                           of the ATR stop / structure stop is further from entry (safer)
    target_basis:         "2R" (default, risk-multiple target) or "ATR" (ATR-based target)
    """
    atr_stop = entry_price - atr_stop_mult * atr_value
    structure_stop = structure_support if (structure_support is not None and structure_support < entry_price) else None

    if use_structure_stop and structure_stop is not None:
        # the lower (further-away) of the two is the more conservative stop —
        # it only triggers on a genuine structure break, not just ATR noise
        stop = min(atr_stop, structure_stop)
        stop_basis = "Structure (support)" if stop == structure_stop else "ATR"
    else:
        stop = atr_stop
        stop_basis = "ATR"

    risk_per_share = entry_price - stop
    atr_target = entry_price + 3.5 * atr_value
    r_target = entry_price + r_multiple * risk_per_share
    target = r_target if target_basis == "2R" else atr_target
    reward_per_share = target - entry_price
    rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else float("nan")

    # sizing: the more conservative of risk-based and capital-based
    risk_amount = investment_amount * (risk_pct / 100)
    risk_based_size = (risk_amount / risk_per_share) if risk_per_share > 0 else 0.0
    capital_based_size = (investment_amount / entry_price) if entry_price > 0 else 0.0
    size = min(risk_based_size, capital_based_size) if risk_per_share > 0 else capital_based_size

    dollar_risk = size * risk_per_share
    dollar_reward = size * reward_per_share
    loss_pct = (risk_per_share / entry_price * 100) if entry_price > 0 else float("nan")
    profit_pct = (reward_per_share / entry_price * 100) if entry_price > 0 else float("nan")

    return RiskPlan(
        entry=entry_price, atr=atr_value, stop=stop, stop_basis=stop_basis,
        atr_stop=atr_stop, structure_stop=structure_stop,
        target=target, target_basis=target_basis, atr_target=atr_target, r_target=r_target,
        risk_per_share=risk_per_share, reward_per_share=reward_per_share, risk_reward=rr,
        risk_based_size=risk_based_size, capital_based_size=capital_based_size, position_size=size,
        investment_amount=investment_amount, dollar_risk=dollar_risk, dollar_reward=dollar_reward,
        loss_pct=loss_pct, profit_pct=profit_pct,
    )


def volatility_regime(atr_series: pd.Series, lookback: int = 100) -> str:
    """Classify current ATR against its recent history (percentile rank)."""
    clean = atr_series.dropna()
    if len(clean) < 20:
        return "Unknown (not enough data)"
    current = clean.iloc[-1]
    recent = clean.tail(lookback)
    pct = (recent < current).mean() * 100
    if pct >= 80:
        return f"High volatility ({pct:.0f}th percentile)"
    if pct <= 20:
        return f"Low volatility ({pct:.0f}th percentile)"
    return f"Normal volatility ({pct:.0f}th percentile)"


def chandelier_exit(df: pd.DataFrame, atr_period: int = 22, atr_mult: float = 3.0) -> pd.Series:
    """
    Chandelier Exit — a classic ATR trailing stop for a long position:
        stop[t] = highest high over the lookback window - atr_mult * ATR
    The stop only ratchets up as new highs form (never moves down).
    """
    a = ta.atr(df["high"], df["low"], df["close"], atr_period)
    highest_high = df["high"].rolling(atr_period).max()
    raw_stop = highest_high - atr_mult * a
    return raw_stop.cummax()


def stop_breached(current_price: float, stop_price: float) -> bool:
    return current_price <= stop_price


def target_reached(current_price: float, target_price: float) -> bool:
    return current_price >= target_price
