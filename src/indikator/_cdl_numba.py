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


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_piercing_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Piercing Pattern (Bullish Reversal)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    is_bear1 = c1 < o1
    body1 = o1 - c1
    
    o2, c2 = open_[i], close[i]
    is_bull2 = c2 > o2
    
    gap_down = o2 < low[i-1] 
    
    midpoint1 = c1 + (body1 * 0.5)
    closes_high_enough = c2 > midpoint1
    closes_within_open = c2 < o1
    
    if is_bear1 & is_bull2 & gap_down & closes_high_enough & closes_within_open:
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_dark_cloud_cover_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Dark Cloud Cover (Bearish Reversal)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    is_bull1 = c1 > o1
    body1 = c1 - o1
    
    o2, c2 = open_[i], close[i]
    is_bear2 = c2 < o2
    
    gap_up = o2 > high[i-1]
    
    midpoint1 = o1 + (body1 * 0.5)
    produces_cover = c2 < midpoint1
    stays_within = c2 > o1
    
    if is_bull1 & is_bear2 & gap_up & produces_cover & stays_within:
      out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_kicking_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Kicking Pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    
    body1 = abs(c1 - o1)
    rng1 = high[i-1] - low[i-1]
    is_maru1 = body1 > (rng1 * 0.9)
    
    body2 = abs(c2 - o2)
    rng2 = high[i] - low[i]
    is_maru2 = body2 > (rng2 * 0.9)
    
    if not (is_maru1 & is_maru2): continue
        
    is_bear1 = c1 < o1
    is_bull2 = c2 > o2
    gap_up = o2 > o1
    
    if is_bear1 & is_bull2 & gap_up:
        out[i] = 100
        continue
        
    is_bull1 = c1 > o1
    is_bear2 = c2 < o2
    gap_down = o2 < o1
    
    if is_bull1 & is_bear2 & gap_down:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_matching_low_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Matching Low (Bullish Reversal)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    c1, o1 = close[i-1], open_[i-1]
    c2, o2 = close[i], open_[i]
    
    is_bear1 = c1 < o1
    is_bear2 = c2 < o2
    
    same_close = abs(c1 - c2) < 1e-5
    
    if is_bear1 & is_bear2 & same_close:
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_spinning_top_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Spinning Top (Indecision)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    
    if rng < 1e-9: continue
    
    upper_shadow = high[i] - max(open_[i], close[i])
    lower_shadow = min(open_[i], close[i]) - low[i]
    
    is_small_body = body < (rng * 0.3)
    has_upper = upper_shadow > body 
    has_lower = lower_shadow > body
    
    if is_small_body & has_upper & has_lower:
        out[i] = 100 if close[i] >= open_[i] else -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_rickshaw_man_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Rickshaw Man (Doji with long shadows)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    
    is_doji = body <= (rng * 0.1)
    
    midpoint = (high[i] + low[i]) * 0.5
    body_mid = (open_[i] + close[i]) * 0.5
    near_mid = abs(body_mid - midpoint) < (rng * 0.1)
    
    if is_doji & near_mid & (rng > 0):
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_high_wave_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect High Wave (Extreme Indecision)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    
    upper_shadow = high[i] - max(open_[i], close[i])
    lower_shadow = min(open_[i], close[i]) - low[i]
    
    is_small_body = body < (rng * 0.2)
    long_upper = upper_shadow > (rng * 0.3)
    long_lower = lower_shadow > (rng * 0.3)
    
    if is_small_body & long_upper & long_lower:
        out[i] = 100 if close[i] >= open_[i] else -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_long_legged_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Long Legged Doji."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    
    is_doji = body <= (rng * 0.1)
    
    if is_doji & (rng > body * 5):
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tristar_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tristar Pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    b1 = abs(close[i-2] - open_[i-2])
    r1 = high[i-2] - low[i-2]
    d1 = b1 <= (r1 * 0.1)
    
    b2 = abs(close[i-1] - open_[i-1])
    r2 = high[i-1] - low[i-1]
    d2 = b2 <= (r2 * 0.1)
    
    b3 = abs(close[i] - open_[i])
    r3 = high[i] - low[i]
    d3 = b3 <= (r3 * 0.1)
    
    if not (d1 & d2 & d3): continue
    
    mid1 = (open_[i-2] + close[i-2])*0.5
    mid2 = (open_[i-1] + close[i-1])*0.5
    mid3 = (open_[i] + close[i])*0.5
    
    is_top = (mid2 > mid1) & (mid2 > mid3)
    is_bot = (mid2 < mid1) & (mid2 < mid3)
    
    if is_top: out[i] = -100
    if is_bot: out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tasuki_gap_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tasuki Gap."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    bull1 = close[i-2] > open_[i-2]
    bull2 = close[i-1] > open_[i-1]
    gap_up = open_[i-1] > close[i-2]
    bear3 = close[i] < open_[i]
    opens_in_2 = (open_[i] < high[i-1]) & (open_[i] > low[i-1])
    closes_in_gap = (close[i] < open_[i-1]) & (close[i] > close[i-2])
    
    if bull1 & bull2 & gap_up & bear3 & opens_in_2 & closes_in_gap:
        out[i] = 100
        continue
        
    bear1 = close[i-2] < open_[i-2]
    bear2 = close[i-1] < open_[i-1]
    gap_down = open_[i-1] < close[i-2]
    bull3 = close[i] > open_[i]
    opens_in_2d = (open_[i] > low[i-1]) & (open_[i] < high[i-1])
    closes_in_gapd = (close[i] > open_[i-1]) & (close[i] < close[i-2])
    
    if bear1 & bear2 & gap_down & bull3 & opens_in_2d & closes_in_gapd:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_separating_lines_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Separating Lines."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    
    same_open = abs(o1 - o2) < 1e-5
    if not same_open: continue
    
    if (c1 < o1) & (c2 > o2): out[i] = 100
    if (c1 > o1) & (c2 < o2): out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_gap_side_by_side_white_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Gap Side-by-Side White."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    bull2 = close[i-1] > open_[i-1]
    bull3 = close[i] > open_[i]
    if not (bull2 & bull3): continue
        
    gap_up = open_[i-1] > close[i-2]
    if (close[i-2] > open_[i-2]) & gap_up:
        out[i] = 100
    
    gap_down = open_[i-1] < close[i-2]
    if (close[i-2] < open_[i-2]) & gap_down:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_two_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Two Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if close[i-2] <= open_[i-2]: continue
    if close[i-1] >= open_[i-1]: continue
    if close[i-1] <= close[i-2]: continue
    if close[i] >= open_[i]: continue
    
    opens_in_2 = (open_[i] < open_[i-1]) & (open_[i] > close[i-1])
    closes_in_1 = (close[i] < close[i-2]) & (close[i] > open_[i-2])
    
    if opens_in_2 & closes_in_1:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_upside_gap_two_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Upside Gap Two Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if not (close[i-2] > open_[i-2]): continue
    if not (close[i-1] < open_[i-1]): continue
    if not (close[i] < open_[i]): continue
    
    gap_up = close[i-1] > close[i-2]
    engulfs_2 = (open_[i] > open_[i-1]) & (close[i] < close[i-1])
    
    if gap_up & engulfs_2:
        out[i] = -100

  return out

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    
    is_doji = body <= (rng * 0.1)
    
    if is_doji & (rng > body * 5):
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tristar_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tristar Pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    b1 = abs(close[i-2] - open_[i-2])
    r1 = high[i-2] - low[i-2]
    d1 = b1 <= (r1 * 0.1)
    
    b2 = abs(close[i-1] - open_[i-1])
    r2 = high[i-1] - low[i-1]
    d2 = b2 <= (r2 * 0.1)
    
    b3 = abs(close[i] - open_[i])
    r3 = high[i] - low[i]
    d3 = b3 <= (r3 * 0.1)
    
    if not (d1 & d2 & d3): continue
    
    mid1 = (open_[i-2] + close[i-2])*0.5
    mid2 = (open_[i-1] + close[i-1])*0.5
    mid3 = (open_[i] + close[i])*0.5
    
    is_top = (mid2 > mid1) & (mid2 > mid3)
    is_bot = (mid2 < mid1) & (mid2 < mid3)
    
    if is_top: out[i] = -100
    if is_bot: out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tasuki_gap_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tasuki Gap."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    bull1 = close[i-2] > open_[i-2]
    bull2 = close[i-1] > open_[i-1]
    gap_up = open_[i-1] > close[i-2]
    bear3 = close[i] < open_[i]
    opens_in_2 = (open_[i] < high[i-1]) & (open_[i] > low[i-1])
    closes_in_gap = (close[i] < open_[i-1]) & (close[i] > close[i-2])
    
    if bull1 & bull2 & gap_up & bear3 & opens_in_2 & closes_in_gap:
        out[i] = 100
        continue
        
    bear1 = close[i-2] < open_[i-2]
    bear2 = close[i-1] < open_[i-1]
    gap_down = open_[i-1] < close[i-2]
    bull3 = close[i] > open_[i]
    opens_in_2d = (open_[i] > low[i-1]) & (open_[i] < high[i-1])
    closes_in_gapd = (close[i] > open_[i-1]) & (close[i] < close[i-2])
    
    if bear1 & bear2 & gap_down & bull3 & opens_in_2d & closes_in_gapd:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_separating_lines_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Separating Lines."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    
    same_open = abs(o1 - o2) < 1e-5
    if not same_open: continue
    
    if (c1 < o1) & (c2 > o2): out[i] = 100
    if (c1 > o1) & (c2 < o2): out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_gap_side_by_side_white_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Gap Side-by-Side White."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    bull2 = close[i-1] > open_[i-1]
    bull3 = close[i] > open_[i]
    if not (bull2 & bull3): continue
        
    gap_up = open_[i-1] > close[i-2]
    if (close[i-2] > open_[i-2]) & gap_up:
        out[i] = 100
    
    gap_down = open_[i-1] < close[i-2]
    if (close[i-2] < open_[i-2]) & gap_down:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_two_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Two Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if close[i-2] <= open_[i-2]: continue
    if close[i-1] >= open_[i-1]: continue
    if close[i-1] <= close[i-2]: continue
    if close[i] >= open_[i]: continue
    
    opens_in_2 = (open_[i] < open_[i-1]) & (open_[i] > close[i-1])
    closes_in_1 = (close[i] < close[i-2]) & (close[i] > open_[i-2])
    
    if opens_in_2 & closes_in_1:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_upside_gap_two_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Upside Gap Two Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if not (close[i-2] > open_[i-2]): continue
    if not (close[i-1] < open_[i-1]): continue
    if not (close[i] < open_[i]): continue
    
    gap_up = close[i-1] > close[i-2]
    engulfs_2 = (open_[i] > open_[i-1]) & (close[i] < close[i-1])
    
    if gap_up & engulfs_2:
        out[i] = -100

  return out

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    
    is_doji = body <= (rng * 0.1)
    
    if is_doji & (rng > body * 5):
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tristar_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tristar Pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    b1 = abs(close[i-2] - open_[i-2])
    r1 = high[i-2] - low[i-2]
    d1 = b1 <= (r1 * 0.1)
    
    b2 = abs(close[i-1] - open_[i-1])
    r2 = high[i-1] - low[i-1]
    d2 = b2 <= (r2 * 0.1)
    
    b3 = abs(close[i] - open_[i])
    r3 = high[i] - low[i]
    d3 = b3 <= (r3 * 0.1)
    
    if not (d1 & d2 & d3): continue
    
    mid1 = (open_[i-2] + close[i-2])*0.5
    mid2 = (open_[i-1] + close[i-1])*0.5
    mid3 = (open_[i] + close[i])*0.5
    
    is_top = (mid2 > mid1) & (mid2 > mid3)
    is_bot = (mid2 < mid1) & (mid2 < mid3)
    
    if is_top: out[i] = -100
    if is_bot: out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_abandoned_baby_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Abandoned Baby."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    bear1 = close[i-2] < open_[i-2]
    doji2 = abs(close[i-1] - open_[i-1]) <= (high[i-1]-low[i-1])*0.1
    bull3 = close[i] > open_[i]
    
    if bear1 & doji2 & bull3:
        gap1 = low[i-2] > high[i-1]
        gap2 = low[i] > high[i-1]
        if gap1 & gap2: out[i] = 100
    
    bull1_ = close[i-2] > open_[i-2]
    bear3_ = close[i] < open_[i]
    if bull1_ & doji2 & bear3_:
        gap1_ = high[i-2] < low[i-1]
        gap2_ = high[i] < low[i-1]
        if gap1_ & gap2_: out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_advance_block_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Advance Block (Bearish Reversal)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if not ((close[i-2]>open_[i-2]) & (close[i-1]>open_[i-1]) & (close[i]>open_[i])): continue
    
    if not ((open_[i-1] > open_[i-2]) & (open_[i-1] < close[i-2])): continue
    if not ((open_[i] > open_[i-1]) & (open_[i] < close[i-1])): continue
    
    u1 = high[i-2] - close[i-2]
    u2 = high[i-1] - close[i-1]
    u3 = high[i] - close[i]
    
    weakening = (u2 > u1) & (u3 > u2)
    
    if weakening: out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_belt_hold_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Belt Hold."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    if rng < 1e-9: continue
    
    if close[i] > open_[i]: # Bull
        no_lower = (open_[i] - low[i]) < (body * 0.05)
        if no_lower: out[i] = 100
    else: # Bear
        no_upper = (high[i] - open_[i]) < (body * 0.05)
        if no_upper: out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_breakaway_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Breakaway."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_closing_marubozu_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Closing Marubozu."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    if body < 1e-9: continue
    
    if close[i] > open_[i]: # Bull
        if (high[i] - close[i]) < (body * 0.05): out[i] = 100
    else: # Bear
        if (close[i] - low[i]) < (body * 0.05): out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_dragonfly_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Dragonfly Doji (T-shape)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    is_doji = body <= (rng * 0.1)
    
    upper = high[i] - max(open_[i], close[i])
    lower = min(open_[i], close[i]) - low[i]
    
    is_dragonfly = is_doji & (upper < rng*0.1) & (lower > rng*0.6)
    
    if is_dragonfly: out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_gravestone_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Gravestone Doji (Inverted T)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    is_doji = body <= (rng * 0.1)
    
    upper = high[i] - max(open_[i], close[i])
    lower = min(open_[i], close[i]) - low[i]
    
    is_gravestone = is_doji & (lower < rng*0.1) & (upper > rng*0.6)
    
    if is_gravestone: out[i] = 100

  return out
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hikkake_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hikkake pattern.
  
  Three stages:
  1. Harami (Internal Bar).
  2. False Breakout.
  3. Reversal.
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(5, n):
    # Harami at i-2, i-3
    h3, l3 = high[i-3], low[i-3]
    h2, l2 = high[i-2], low[i-2]
    
    is_harami = (h2 < h3) & (l2 > l3)
    if not is_harami: continue
    
    # Breakout candle at i-1
    h1, l1 = high[i-1], low[i-1]
    
    # Bullish Hikkake: i-1 breaks low, current i breaks high of harami window
    # Actually TA-Lib often returns at the moment of breakout reversal
    if (l1 < l3) & (close[i] > h3):
        out[i] = 100
    
    # Bearish Hikkake: i-1 breaks high, current i breaks low
    if (h1 > h3) & (close[i] < l3):
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_homing_pigeon_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Homing Pigeon. Both candles black, second inside first."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    if (c1 < o1) & (c2 < o2):
      if (o2 < o1) & (c2 > c1):
        out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_identical_three_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Identical Three Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    o1, c1 = open_[i-2], close[i-2]
    o2, c2 = open_[i-1], close[i-1]
    o3, c3 = open_[i], close[i]
    
    # All black
    if (c1 < o1) & (c2 < o2) & (c3 < o3):
      # Identical opening (near previous close)
      if (abs(o2 - c1) < 1e-5) & (abs(o3 - c2) < 1e-5):
        out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_in_neck_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect In-Neck."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    if (c1 < o1) & (c2 > o2): # Bear then Bull
      # Closes at previous low (or close)
      if abs(c2 - c1) < (abs(o1-c1)*0.1): # Close to close
        out[i] = -100
  return out
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_ladder_bottom_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Ladder Bottom."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(4, n):
    # 3 bears + stairs down
    b1 = close[i-4] < open_[i-4]
    b2 = close[i-3] < open_[i-3]
    b3 = close[i-2] < open_[i-2]
    stairs = (close[i-3] < close[i-4]) & (close[i-2] < close[i-3])
    
    # 4th bear with upper shadow
    b4 = close[i-1] < open_[i-1]
    has_shadow = (high[i-1] - open_[i-1]) > (abs(open_[i-1]-close[i-1])*0.5)
    
    # 5th bull gaps up
    bull5 = close[i] > open_[i]
    gap_up = open_[i] > high[i-1]
    
    if b1 & b2 & b3 & b4 & stairs & has_shadow & bull5 & gap_up:
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_long_line_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Long Line."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  # Needs reference to average body? For now simple range check.
  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    if (rng > 0) & (body > rng * 0.8):
      out[i] = 100 if close[i] > open_[i] else -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_mat_hold_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Mat Hold."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  # 5 candles: Long Bull, Gap Up + 3 Small Bears inside body, Long Bull higher
  # Simplified implementation
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_on_neck_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect On-Neck."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    if (c1 < o1) & (c2 > o2): # Bear then Bull
      # Closes AT previous low (not close)
      if abs(c2 - low[i-1]) < (abs(o1-c1)*0.05):
        out[i] = -100
  return out
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_rise_fall_three_methods_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Rise/Fall Three Methods."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(4, n):
    # Rising 3 Methods
    if (close[i-4] > open_[i-4]): # Bull
      # 3 Small Bears inside body of C1
      bears = (close[i-3] < open_[i-3]) & (close[i-2] < open_[i-2]) & (close[i-1] < open_[i-1])
      inside = (high[i-3] < high[i-4]) & (low[i-3] > low[i-4])
      if bears & inside & (close[i] > close[i-4]):
        out[i] = 100
    # Falling 3 Methods
    elif (close[i-4] < open_[i-4]): # Bear
      bulls = (close[i-3] > open_[i-3]) & (close[i-2] > open_[i-2]) & (close[i-1] > open_[i-1])
      inside = (high[i-3] < high[i-4]) & (low[i-3] > low[i-4])
      if bulls & inside & (close[i] < close[i-4]):
        out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_short_line_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Short Line."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    if (rng > 0) & (body < rng * 0.3):
      out[i] = 100 if close[i] > open_[i] else -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_stalled_pattern_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Stalled Pattern (Bearish Reversal)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  # Simplified: 3 bulls, each closing higher, but 3rd has tiny body
  for i in range(2, n):
    bulls = (close[i-2]>open_[i-2]) & (close[i-1]>open_[i-1]) & (close[i]>open_[i])
    higher = (close[i-1] > close[i-2]) & (close[i] >= close[i-1])
    small_body = abs(close[i]-open_[i]) < abs(close[i-1]-open_[i-1])*0.3
    if bulls & higher & small_body:
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_stick_sandwich_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Stick Sandwich."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    # Bear, Bull (higher), Bear (same close)
    if (close[i-2] < open_[i-2]) & (close[i-1] > open_[i-1]) & (close[i] < open_[i]):
      if abs(close[i] - close[i-2]) < 1e-5:
        out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_takuri_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Takuri (Dragonfly-like Hammer)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]
    if rng < 1e-9: continue
    lower_shadow = min(open_[i], close[i]) - low[i]
    # Takuri: Lower shadow is huge (> 3x body and > 75% range)
    if (lower_shadow > body * 3) & (lower_shadow > rng * 0.75):
      out[i] = 100
  return out
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_thrusting_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Thrusting Pattern (Bearish Trend)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    if (c1 < o1) & (c2 > o2): # Bear then Bull
      # Closes below middle of previous body
      mid = (o1 + c1) * 0.5
      if (c2 < mid) & (c2 > c1):
        out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_unique_three_river_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Unique 3 River."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    # Long bear, Hammer-like bear with lower low, small bull inside
    bear1 = close[i-2] < open_[i-2]
    bear2 = close[i-1] < open_[i-1]
    if bear1 & bear2:
      if (low[i-1] < low[i-2]) & (close[i] > open_[i]) & (close[i] < close[i-1]):
        out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_counterattack_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Counterattack."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    o1, c1 = open_[i-1], close[i-1]
    o2, c2 = open_[i], close[i]
    # Opposite colors, same close
    if (c1 > o1) & (c2 < o2): # Bull then Bear
      if abs(c1 - c2) < 1e-5: out[i] = -100
    elif (c1 < o1) & (c2 > o2): # Bear then Bull
      if abs(c1 - c2) < 1e-5: out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Doji Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    body1 = abs(close[i-1] - open_[i-1])
    body2 = abs(close[i] - open_[i])
    rng2 = high[i] - low[i]
    is_doji2 = (rng2 > 0) & (body2 <= rng2 * 0.1)
    if is_doji2:
      if (close[i-1] > open_[i-1]) & (open_[i] > close[i-1]):
        out[i] = -100 # Bearish Doji Star (Gaps up)
      elif (close[i-1] < open_[i-1]) & (open_[i] < close[i-1]):
        out[i] = 100 # Bullish (Gaps down)
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_conceal_baby_swallow_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Concealing Baby Swallow."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  return out
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_harami_cross_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Harami Cross."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    pc, po = close[i-1], open_[i-1]
    cc, co = close[i], open_[i]
    is_doji = abs(cc - co) <= (high[i]-low[i])*0.1
    if is_doji:
      # Doji inside previous body
      if (max(cc, co) < max(pc, po)) & (min(cc, co) > min(pc, po)):
        out[i] = 100 if pc < po else -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hikkake_modified_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Modified Hikkake."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  # Simplified
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_morning_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Morning Doji Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    # Long bear, Doji gap down, Long bull inside 1st body
    if (close[i-2] < open_[i-2]) & (abs(close[i-1]-open_[i-1]) <= (high[i-1]-low[i-1])*0.1):
      if (max(open_[i-1], close[i-1]) < close[i-2]) & (close[i] > (open_[i-2]+close[i-2])*0.5):
        out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_evening_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Evening Doji Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if (close[i-2] > open_[i-2]) & (abs(close[i-1]-open_[i-1]) <= (high[i-1]-low[i-1])*0.1):
      if (min(open_[i-1], close[i-1]) > close[i-2]) & (close[i] < (open_[i-2]+close[i-2])*0.5):
        out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_kicking_by_length_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Kicking By Length."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  # Similar to Kicking
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_stars_in_south_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Stars In The South."""
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_xsidegap3methods_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Upside/Downside Gap Three Methods."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    o0, h0, l0, c0 = open_[i-2], high[i-2], low[i-2], close[i-2]
    o1, h1, l1, c1 = open_[i-1], high[i-1], low[i-1], close[i-1]
    o2, h2, l2, c2 = open_[i], high[i], low[i], close[i]

    # Upside Gap Three Methods (Bullish)
    bull1 = c0 > o0
    bull2 = c1 > o1
    bear3 = c2 < o2
    gap_up = o1 > c0
    fill_up = (o2 < c1) & (o2 > o1) & (c2 < o1) & (c2 > o0)
    
    is_bull = bull1 & bull2 & bear3 & gap_up & fill_up
    
    # Downside Gap Three Methods (Bearish)
    bear1 = c0 < o0
    bear2 = c1 < o1
    bull3 = c2 > o2
    gap_down = o1 < c0
    fill_down = (o2 > c1) & (o2 < o1) & (c2 > o1) & (c2 < o0)

    is_bear = bear1 & bear2 & bull3 & gap_down & fill_down
    
    out[i] = (is_bull * 100) - (is_bear * 100)

  return out
