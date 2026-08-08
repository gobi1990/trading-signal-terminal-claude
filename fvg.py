"""
fvg.py
------
Fair Value Gap (FVG) detection — the classic 3-candle ICT/price-action gap:

  Bullish FVG: low of candle[i]  >  high of candle[i-2]
               (an imbalance/gap left behind by a strong up move on candle[i-1])
  Bearish FVG: high of candle[i] <  low  of candle[i-2]
               (an imbalance/gap left behind by a strong down move on candle[i-1])

Each gap is defined by a [bottom, top] price zone. A gap is considered
"filled" once price later trades back through the zone. Many traders treat
an unfilled FVG as a magnet / support-resistance zone and look for price to
retrace into it before continuing in the direction of the original move.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class FVGZone:
    index: int              # bar index where the gap was confirmed (candle i)
    timestamp: pd.Timestamp
    direction: Literal["bullish", "bearish"]
    top: float
    bottom: float
    filled: bool = False
    filled_at: pd.Timestamp = None

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def size(self) -> float:
        return self.top - self.bottom


def detect_fair_value_gaps(df: pd.DataFrame, min_gap_pct: float = 0.0) -> list[FVGZone]:
    """
    Scan a DataFrame with columns [open, high, low, close] (datetime index)
    for bullish and bearish Fair Value Gaps, and mark whether each has since
    been filled by subsequent price action.

    min_gap_pct: minimum gap size as a % of price to filter out noise
                 (e.g. 0.1 = ignore gaps smaller than 0.1% of price).
    """
    zones: list[FVGZone] = []
    highs = df["high"].values
    lows = df["low"].values
    idx = df.index

    for i in range(2, len(df)):
        # Bullish FVG: gap between candle i-2's high and candle i's low
        if lows[i] > highs[i - 2]:
            gap_bottom, gap_top = highs[i - 2], lows[i]
            if (gap_top - gap_bottom) / gap_bottom * 100 >= min_gap_pct:
                zones.append(FVGZone(
                    index=i, timestamp=idx[i], direction="bullish",
                    top=gap_top, bottom=gap_bottom,
                ))

        # Bearish FVG: gap between candle i-2's low and candle i's high
        if highs[i] < lows[i - 2]:
            gap_top, gap_bottom = lows[i - 2], highs[i]
            if (gap_top - gap_bottom) / gap_bottom * 100 >= min_gap_pct:
                zones.append(FVGZone(
                    index=i, timestamp=idx[i], direction="bearish",
                    top=gap_top, bottom=gap_bottom,
                ))

    # mark fill status using all price action after the gap formed
    for zone in zones:
        future = df.iloc[zone.index + 1:]
        if zone.direction == "bullish":
            # filled when price trades back down into/through the zone
            hit = future[future["low"] <= zone.top]
        else:
            # filled when price trades back up into/through the zone
            hit = future[future["high"] >= zone.bottom]
        if not hit.empty:
            zone.filled = True
            zone.filled_at = hit.index[0]

    return zones


def zones_to_dataframe(zones: list[FVGZone]) -> pd.DataFrame:
    if not zones:
        return pd.DataFrame(columns=["timestamp", "direction", "top", "bottom", "size", "filled", "filled_at"])
    return pd.DataFrame([{
        "timestamp": z.timestamp, "direction": z.direction, "top": z.top,
        "bottom": z.bottom, "size": z.size, "filled": z.filled, "filled_at": z.filled_at,
    } for z in zones])


def nearest_unfilled_zone(zones: list[FVGZone], price: float, direction: str = None):
    """Return the closest unfilled FVG to the current price, optionally filtered by direction."""
    candidates = [z for z in zones if not z.filled and (direction is None or z.direction == direction)]
    if not candidates:
        return None
    return min(candidates, key=lambda z: abs(z.mid - price))
