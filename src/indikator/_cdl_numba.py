"""Numba-optimized Candle Pattern Recognition.

This module provides JIT-compiled kernels for detecting candlestick patterns.
Optimized with Branchless Logic where possible to maximize CPU pipeline throughput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Pattern Return Codes
BULLISH = 100
BEARISH = -100
NO_PATTERN = 0


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Doji pattern (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]

    # Branchless logic:
    # (rng > 0) & (body <= rng * 0.1)
    mask = (rng > 0) & (body <= (rng * 0.1))
    out[i] = mask * 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hammer_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hammer pattern (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(10, n):
    o = open_[i]
    c = close[i]
    h = high[i]
    l = low[i]
    
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l
    rng = h - l
    
    body_val = real_body if real_body > 0 else 1e-9
    
    # Hammer: Long lower shadow, small upper shadow, small body
    c1 = lower_shadow >= (2.0 * body_val)
    c2 = upper_shadow <= (0.1 * rng)
    c3 = real_body < (0.3 * rng) # Liberal body check
    c4 = rng > 0
    
    is_hammer = c1 & c2 & c3 & c4
    out[i] = is_hammer * 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_engulfing_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Engulfing pattern (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    pc = close[i - 1]
    po = open_[i - 1]
    cc = close[i]
    co = open_[i]

    # Bullish: Prev Red, Curr Green, Engulfs
    is_bull = (po > pc) & (cc > co) & (co <= pc) & (cc >= po)
    
    # Bearish: Prev Green, Curr Red, Engulfs
    is_bear = (pc > po) & (co > cc) & (co >= pc) & (cc <= po)
    
    out[i] = (is_bull * 100) - (is_bear * 100)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_harami_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Harami pattern (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    pc = close[i - 1]
    po = open_[i - 1]
    cc = close[i]
    co = open_[i]

    prev_top = po if po > pc else pc
    prev_bot = po if po < pc else pc
    curr_top = co if co > cc else cc
    curr_bot = co if co < cc else cc
    
    is_inside = (curr_top < prev_top) & (curr_bot > prev_bot)
    
    # Bullish Harami: Prev is Bearish (Red)
    is_prev_bear = pc < po
    # Bearish Harami: Prev is Bullish (Green)
    is_prev_bull = pc >= po 
    
    val = (is_prev_bear * 100) - (is_prev_bull * 100)
    out[i] = is_inside * val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_shooting_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Shooting Star (Branchless).
  
  Bearish reversal. Inverse of Hammer.
  - Small body near bottom.
  - Long upper shadow (>= 2 * body).
  - Short lower shadow.
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l
    rng = h - l
    
    body_val = real_body if real_body > 0 else 1e-9
    
    # Criteria
    c1 = upper_shadow >= (2.0 * body_val)
    c2 = lower_shadow <= (0.1 * rng) # Small lower shadow
    c3 = real_body < (0.3 * rng)
    c4 = rng > 0
    
    # Gap up validation? TA-Lib Shooting Star often checks if body gaps up from previous.
    # Simplified here: just shape.
    
    is_star = c1 & c2 & c3 & c4
    out[i] = is_star * -100 # Bearish

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_inverted_hammer_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Inverted Hammer (Branchless).
  
  Bullish reversal. Same shape as Shooting Star, but found in downtrend.
  For pure pattern recognition, we return 100 if shape matches.
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l
    rng = h - l
    
    body_val = real_body if real_body > 0 else 1e-9
    
    c1 = upper_shadow >= (2.0 * body_val)
    c2 = lower_shadow <= (0.1 * rng)
    c3 = real_body < (0.3 * rng)
    c4 = rng > 0
    
    is_inv_hammer = c1 & c2 & c3 & c4
    out[i] = is_inv_hammer * 100 # Bullish

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hanging_man_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hanging Man (Branchless).
  
  Bearish reversal. Same shape as Hammer.
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l
    rng = h - l
    
    body_val = real_body if real_body > 0 else 1e-9
    
    c1 = lower_shadow >= (2.0 * body_val)
    c2 = upper_shadow <= (0.1 * rng)
    c3 = real_body < (0.3 * rng)
    c4 = rng > 0
    
    is_hanging = c1 & c2 & c3 & c4
    out[i] = is_hanging * -100 # Bearish

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_marubozu_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Marubozu (Branchless).
  
  Long body, very small shadows (<= 5% of body or range).
  Bullish (White) Marubozu: Close > Open (100)
  Bearish (Black) Marubozu: Open > Close (-100)
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    
    body = abs(c - o)
    rng = h - l
    
    # Avoid zero body or range issues
    if body < 1e-9:
        continue
    
    body_top = o if o > c else c
    body_bot = o if o < c else c
    
    upper_shadow = h - body_top
    lower_shadow = body_bot - l
    
    # Shadows must be tiny (e.g. < 5% of body)
    c1 = upper_shadow < (0.05 * body)
    c2 = lower_shadow < (0.05 * body)
    c3 = body > (0.5 * rng) # Body dominates range
    
    is_marubozu = c1 & c2 & c3
    
    # Direction
    is_bull = c > o
    val = (is_bull * 100) - ((1 - is_bull) * 100) # 100 if bull, -100 if bear
    
    out[i] = is_marubozu * val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_morning_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Morning Star (3-candle Bullish Reversal).
  
  1. Long Bearish candle
  2. Small candle (gap down) - Star
  3. Long Bullish candle (closes well inside first body)
  
  Returns: 100 (Bullish)
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    # Candle 1 (i-2): Long Bearish
    c1 = close[i-2]
    o1 = open_[i-2]
    body1 = abs(c1 - o1)
    is_bear1 = c1 < o1
    is_long1 = body1 > (high[i-2] - low[i-2]) * 0.6 # Body > 60% range
    
    # Candle 2 (i-1): Small Body, Gap Down
    c2 = close[i-1]
    o2 = open_[i-1]
    body2 = abs(c2 - o2)
    is_small2 = body2 < body1 * 0.3 # Significantly smaller
    # Gap Down: Body 2 below Body 1
    body1_bot = c1 # Bearish, so Close is bottom
    body2_top = max(o2, c2)
    is_gap_down = body2_top < body1_bot
    
    # Candle 3 (i): Long Bullish
    c3 = close[i]
    o3 = open_[i]
    is_bull3 = c3 > o3
    # Closes inside body of Candle 1 (above midpoint usually)
    midpoint1 = (o1 + c1) * 0.5
    closes_inside = c3 > midpoint1
    
    if is_bear1 & is_long1 & is_small2 & is_gap_down & is_bull3 & closes_inside:
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_evening_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Evening Star (3-candle Bearish Reversal).
  
  1. Long Bullish candle
  2. Small candle (gap up)
  3. Long Bearish candle (closes well inside first body)
  
  Returns: -100 (Bearish)
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    # Candle 1 (i-2): Long Bullish
    c1 = close[i-2]
    o1 = open_[i-2]
    body1 = abs(c1 - o1)
    is_bull1 = c1 > o1
    is_long1 = body1 > (high[i-2] - low[i-2]) * 0.6
    
    # Candle 2 (i-1): Small Body, Gap Up
    c2 = close[i-1]
    o2 = open_[i-1]
    body2 = abs(c2 - o2)
    is_small2 = body2 < body1 * 0.3
    # Gap Up: Body 2 above Body 1
    body1_top = c1
    body2_bot = min(o2, c2)
    is_gap_up = body2_bot > body1_top
    
    # Candle 3 (i): Long Bearish
    c3 = close[i]
    o3 = open_[i]
    is_bear3 = c3 < o3
    # Closes inside body of Candle 1 (below midpoint)
    midpoint1 = (o1 + c1) * 0.5
    closes_inside = c3 < midpoint1
    
    if is_bull1 & is_long1 & is_small2 & is_gap_up & is_bear3 & closes_inside:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_black_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Black Crows (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(3, n):
    is_bear1 = close[i-2] < open_[i-2]
    is_bear2 = close[i-1] < open_[i-1]
    is_bear3 = close[i] < open_[i]

    lower_close1 = close[i-1] < close[i-2]
    lower_close2 = close[i] < close[i-1]

    open_in_body1 = (open_[i-1] < open_[i-2]) & (open_[i-1] > close[i-2])
    open_in_body2 = (open_[i] < open_[i-1]) & (open_[i] > close[i-1])

    if is_bear1 & is_bear2 & is_bear3 & lower_close1 & lower_close2 & open_in_body1 & open_in_body2:
      out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_white_soldiers_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three White Soldiers (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(3, n):
    is_bull1 = close[i-2] > open_[i-2]
    is_bull2 = close[i-1] > open_[i-1]
    is_bull3 = close[i] > open_[i]

    higher_close1 = close[i-1] > close[i-2]
    higher_close2 = close[i] > close[i-1]

    open_in_body1 = (open_[i-1] > open_[i-2]) & (open_[i-1] < close[i-2])
    open_in_body2 = (open_[i] > open_[i-1]) & (open_[i] < close[i-1])

    if is_bull1 & is_bull2 & is_bull3 & higher_close1 & higher_close2 & open_in_body1 & open_in_body2:
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_inside_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Inside Up/Down (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    o1, c1 = open_[i-2], close[i-2]
    o2, c2 = open_[i-1], close[i-1]
    o3, c3 = open_[i], close[i]
    
    top1, bot1 = max(o1, c1), min(o1, c1)
    top2, bot2 = max(o2, c2), min(o2, c2)
    
    is_harami = (top2 < top1) & (bot2 > bot1)
    
    if not is_harami: continue
        
    is_bull1 = c1 > o1
    is_bear1 = c1 < o1
    is_bull3 = c3 > o3
    is_bear3 = c3 < o3
    
    is_inside_up = is_bear1 & (c2 > o2) & is_bull3 & (c3 > c2)
    is_inside_down = is_bull1 & (c2 < o2) & is_bear3 & (c3 < c2)
    
    out[i] = (is_inside_up * 100) - (is_inside_down * 100)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_outside_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Outside Up/Down."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    o1, c1 = open_[i-2], close[i-2]
    o2, c2 = open_[i-1], close[i-1]
    c3 = close[i]
    
    is_bear1 = c1 < o1
    is_bull2 = c2 > o2
    engulfs_up = (c2 > o1) & (o2 < c1)
    confirm_up = c3 > c2
    
    is_bull1 = c1 > o1
    is_bear2 = c2 < o2
    engulfs_down = (c2 < o1) & (o2 > c1)
    confirm_down = c3 < c2
    
    is_up = is_bear1 & is_bull2 & engulfs_up & confirm_up
    is_down = is_bull1 & is_bear2 & engulfs_down & confirm_down
    
    out[i] = (is_up * 100) - (is_down * 100)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_line_strike_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Line Strike (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(4, n):
    c1, o1 = close[i-3], open_[i-3]
    c2, o2 = close[i-2], open_[i-2]
    c3, o3 = close[i-1], open_[i-1]
    c4, o4 = close[i], open_[i]
    
    is_3_bears = (c1 < o1) & (c2 < o2) & (c3 < o3)
    is_stairs_down = (c2 < c1) & (c3 < c2)
    is_bull_strike = (c4 > o4) & (c4 > o1)
    
    strike_up = is_3_bears & is_stairs_down & is_bull_strike
    
    is_3_bulls = (c1 > o1) & (c2 > o2) & (c3 > o3)
    is_stairs_up = (c2 > c1) & (c3 > c2)
    is_bear_strike = (c4 < o4) & (c4 < o1)
    
    strike_down = is_3_bulls & is_stairs_up & is_bear_strike
    
    out[i] = (strike_up * 100) - (strike_down * 100)

  return out
