"""
portfolio.py
------------
Parses pasted brokerage holdings (e.g. copied from Wealthsimple) into a
clean DataFrame of symbol / shares / avg_cost, so each position can be run
through the same signal engine as the live single-symbol view.

Two input styles are supported:

  1. Structured (most reliable) — one holding per line:
         AAPL, 10, 150.00
         SHOP.TO, 5, 72.50
     (symbol, shares, avg cost — shares and avg cost are optional)

  2. Freeform paste — whatever you copy directly out of the Wealthsimple
     app/site. The parser scans each line for a recognizable ticker symbol
     and any nearby numbers, on a best-effort basis. Always double check
     the parsed table before relying on it — freeform parsing can miscount
     shares/cost or miss symbols with unusual formatting.
"""

import re

import pandas as pd

TICKER_RE = re.compile(r"^[A-Z]{1,10}(\.[A-Z]{1,3})?(/[A-Z]{2,10})?$")  # stocks, .TO/.V suffixes, and CRYPTO/PAIR
NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")

# common words that look like tickers but aren't, to reduce false positives
# in freeform parsing
BLACKLIST = {
    "USD", "CAD", "ETF", "INC", "LTD", "THE", "FOR", "AND", "NEW", "LLC",
    "CO", "CORP", "AVG", "QTY", "TOTAL", "VALUE", "COST", "SHARES", "SHARE",
    "GAIN", "LOSS", "TODAY", "ALL", "TIME", "BUY", "SELL", "CASH", "FUND",
}


def _clean_num(s: str) -> float:
    return float(s.replace("$", "").replace(",", ""))


def parse_holdings(text: str) -> pd.DataFrame:
    rows = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # --- Attempt 1: structured CSV/TSV "SYMBOL, SHARES, AVG_COST" ---
        parts = [p.strip() for p in re.split(r"[,\t]+", line) if p.strip()]
        symbol, shares, avg_cost = None, None, None

        if parts and TICKER_RE.match(parts[0].upper()) and parts[0].upper() not in BLACKLIST:
            symbol = parts[0].upper()
            nums = []
            for p in parts[1:]:
                try:
                    nums.append(_clean_num(p))
                except ValueError:
                    continue
            if len(nums) >= 1:
                shares = nums[0]
            if len(nums) >= 2:
                avg_cost = nums[1]

        # --- Attempt 2: freeform line, e.g. pasted straight from the app ---
        if symbol is None:
            tokens = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?(?:/[A-Z]{2,10})?\b", line)
            candidates = [t for t in tokens if t not in BLACKLIST]
            if candidates:
                symbol = candidates[0]
                nums = [_clean_num(n) for n in NUM_RE.findall(line)]
                if len(nums) >= 1:
                    shares = nums[0]
                if len(nums) >= 2:
                    avg_cost = nums[1]

        if symbol:
            rows.append(dict(symbol=symbol, shares=shares, avg_cost=avg_cost, raw_line=raw_line.strip()))

    if not rows:
        return pd.DataFrame(columns=["symbol", "shares", "avg_cost", "raw_line"])

    df = pd.DataFrame(rows).drop_duplicates(subset="symbol").reset_index(drop=True)
    return df
