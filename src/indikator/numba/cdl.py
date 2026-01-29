"""Numba-optimized Candle Pattern Recognition.

This module provides JIT-compiled kernels for detecting candlestick patterns.
Optimized with Branchless Logic where possible to maximize CPU pipeline throughput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
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
  avg_body_doji: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Doji pattern (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body_doji[i]):
      continue
    body = abs(close[i] - open_[i])
    if body < avg_body_doji[i]:
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hammer_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  avg_shadow_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hammer pattern (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    if np.isnan(avg_body_short[i]) or np.isnan(avg_near[i - 1]):
      continue

    o, c, h, l = open_[i], close[i], high[i], low[i]
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # TA-Lib logic:
    # 1. small real body
    # 2. long lower shadow
    # 3. no, or very short, upper shadow
    # 4. body below or near the lows of the previous candle
    if (
      real_body < avg_body_short[i]
      and lower_shadow > avg_shadow_long[i]
      and upper_shadow < avg_shadow_very_short[i]
      and body_bot <= low[i - 1] + avg_near[i - 1]
    ):
      out[i] = 100
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

    # Bullish: Prev Red or Doji, Curr Green, Engulfs
    # Full Engulfment: (co < pc) & (cc > po) -> 100
    # One side equal: ((co == pc) & (cc > po)) | ((co < pc) & (cc == po)) -> 80

    is_bull_full = (po >= pc) & (cc > co) & (co < pc) & (cc > po)
    is_bull_partial = (
      (po >= pc) & (cc > co) & (((co == pc) & (cc > po)) | ((co < pc) & (cc == po)))
    )

    # Bearish: Prev Green or Doji, Curr Red, Engulfs
    is_bear_full = (pc >= po) & (co > cc) & (co > pc) & (cc < po)
    is_bear_partial = (
      (pc >= po) & (co > cc) & (((co == pc) & (cc < po)) | ((co > pc) & (cc == po)))
    )

    val = 0
    if is_bull_full:
      val = 100
    elif is_bull_partial:
      val = 80
    elif is_bear_full:
      val = -100
    elif is_bear_partial:
      val = -80

    out[i] = val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_harami_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Harami pattern (Branchless)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    pc = close[i - 1]
    po = open_[i - 1]
    cc = close[i]
    co = open_[i]

    # 1. Body size checks
    prev_body = abs(pc - po)
    curr_body = abs(cc - co)
    if not (prev_body >= avg_body_long[i - 1] and curr_body <= avg_body_short[i]):
      continue

    # Signal based on FIRST candle color
    # Prev is Bullish (Green) -> Bearish Harami (result -100/-80)
    # Prev is Bearish (Red) -> Bullish Harami (result 100/80)
    is_bull = pc < po
    is_bear = pc > po

    prev_top = po if po > pc else pc
    prev_bot = po if po < pc else pc
    curr_top = co if co > cc else cc
    curr_bot = co if co < cc else cc

    # 3. Containment
    # Grade 100: Strictly inside
    # Grade 80: Semi-strict (allow matching edges)
    if (curr_top < prev_top) and (curr_bot > prev_bot):
      out[i] = 100 if is_bull else -100
    elif (curr_top <= prev_top) and (curr_bot >= prev_bot):
      out[i] = 80 if is_bull else -80

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_shooting_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  avg_shadow_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Shooting Star pattern (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    if np.isnan(avg_body_short[i]):
      continue

    o, c, h, l = open_[i], close[i], high[i], low[i]
    po, pc = open_[i - 1], close[i - 1]

    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # TA-Lib logic:
    # 1. small real body
    # 2. long upper shadow
    # 3. no, or very short, lower shadow
    # 4. gap up from prior real body
    if (
      real_body < avg_body_short[i]
      and upper_shadow > avg_shadow_long[i]
      and lower_shadow < avg_shadow_very_short[i]
      and body_bot > (po if po > pc else pc)
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_inverted_hammer_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  avg_shadow_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Inverted Hammer pattern (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    if np.isnan(avg_body_short[i]):
      continue

    o, c, h, l = open_[i], close[i], high[i], low[i]
    po, pc = open_[i - 1], close[i - 1]

    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # TA-Lib logic:
    # 1. small real body
    # 2. long upper shadow
    # 3. no, or very short, lower shadow
    # 4. gap down from prior real body
    if (
      real_body < avg_body_short[i]
      and upper_shadow > avg_shadow_long[i]
      and lower_shadow < avg_shadow_very_short[i]
      and body_top < (po if po < pc else pc)
    ):
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hanging_man_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  avg_shadow_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hanging Man pattern (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    if np.isnan(avg_body_short[i]) or np.isnan(avg_near[i - 1]):
      continue

    o, c, h, l = open_[i], close[i], high[i], low[i]
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # TA-Lib logic:
    # 1. small real body
    # 2. long lower shadow
    # 3. no, or very short, upper shadow
    # 4. body above or near the highs of the previous candle
    if (
      real_body < avg_body_short[i]
      and lower_shadow > avg_shadow_long[i]
      and upper_shadow < avg_shadow_very_short[i]
      and body_bot >= high[i - 1] - avg_near[i - 1]
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_marubozu_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Marubozu."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body = abs(c - o)

    body_top = o if o > c else c
    body_bot = o if o < c else c

    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # 1. Body must be Long
    c1 = body > avg_body_long[i]
    # 2. Shadows must be Very Short
    c2 = upper_shadow < avg_shadow_very_short[i]
    c3 = lower_shadow < avg_shadow_very_short[i]

    if c1 & c2 & c3:
      if c > o:
        out[i] = 100
      else:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_morning_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Morning Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if (
      np.isnan(avg_body_long[i - 2])
      or np.isnan(avg_body_short[i - 1])
      or np.isnan(avg_body_short[i])
    ):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # TA-Lib logic:
    # 1st: Long Black
    # 2nd: Short, Gap Down (Real Body)
    # 3rd: Longer than Short, White
    # 3rd closes well within 1st RB (at least penetration %)

    body2 = abs(c2 - o2)
    body1 = abs(c1 - o1)
    body0 = abs(c0 - o0)

    if (
      body2 > avg_body_long[i - 2]
      and c2 < o2
      and body1 <= avg_body_short[i - 1]
      and max(o1, c1) < min(o2, c2)
      and body0 > avg_body_short[i]
      and c0 > o0
      and c0 > c2 + body2 * penetration
    ):
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_evening_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Evening Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if (
      np.isnan(avg_body_long[i - 2])
      or np.isnan(avg_body_short[i - 1])
      or np.isnan(avg_body_short[i])
    ):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # TA-Lib logic:
    # 1st: Long White
    # 2nd: Short, Gap Up (Real Body)
    # 3rd: Longer than Short, Black
    # 3rd closes well within 1st RB (at least penetration %)

    body2 = abs(c2 - o2)
    body1 = abs(c1 - o1)
    body0 = abs(c0 - o0)

    if (
      body2 > avg_body_long[i - 2]
      and c2 > o2
      and body1 <= avg_body_short[i - 1]
      and min(o1, c1) > max(o2, c2)
      and body0 > avg_body_short[i]
      and c0 < o0
      and c0 < c2 - body2 * penetration
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_black_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Black Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(3, n):
    if np.isnan(avg_shadow_very_short[i]):
      continue

    c1, o1 = close[i - 2], open_[i - 2]  # 1st Black
    c2, o2 = close[i - 1], open_[i - 1]  # 2nd Black
    c3, o3 = close[i], open_[i]  # 3rd Black

    c0, o0 = close[i - 3], open_[i - 3]  # Prior

    # 1. Prior White
    if not (c0 > o0):
      continue

    # 2. 3 Blacks
    if not ((c1 < o1) & (c2 < o2) & (c3 < o3)):
      continue

    # 3. Very Short Lower Shadows
    ls1 = c1 - low[i - 2]
    ls2 = c2 - low[i - 1]
    ls3 = c3 - low[i]

    # TA-Lib alignment? i=3rd.
    if not (ls1 < avg_shadow_very_short[i - 2]):
      continue
    if not (ls2 < avg_shadow_very_short[i - 1]):
      continue
    if not (ls3 < avg_shadow_very_short[i]):
      continue

    # 4. Opens within previous Real Body
    if not ((o2 < o1) & (o2 > c1)):
      continue
    if not ((o3 < o2) & (o3 > c2)):
      continue

    # 5. Declining Closes
    if not ((c1 > c2) & (c2 > c3)):
      continue

    # 6. 1st Black closes under prior High
    if not (c1 < high[i - 3]):
      continue

    out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_white_soldiers_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  avg_near: NDArray[np.float64],
  avg_far: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three White Soldiers."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if (
      np.isnan(avg_shadow_very_short[i])
      or np.isnan(avg_near[i])
      or np.isnan(avg_far[i])
      or np.isnan(avg_body_short[i])
    ):
      continue

    c1, o1 = close[i - 2], open_[i - 2]
    c2, o2 = close[i - 1], open_[i - 1]
    c3, o3 = close[i], open_[i]

    if not ((c1 > o1) & (c2 > o2) & (c3 > o3)):
      continue

    us1 = high[i - 2] - c1
    us2 = high[i - 1] - c2
    us3 = high[i] - c3

    if not (us1 < avg_shadow_very_short[i - 2]):
      continue
    if not (us2 < avg_shadow_very_short[i - 1]):
      continue
    if not (us3 < avg_shadow_very_short[i]):
      continue

    if not ((c2 > c1) & (c3 > c2)):
      continue

    if not ((o2 > o1) & (o2 <= c1 + avg_near[i - 2])):
      continue
    if not ((o3 > o2) & (o3 <= c2 + avg_near[i - 1])):
      continue

    body1 = c1 - o1
    body2 = c2 - o2
    body3 = c3 - o3

    if not (body2 > body1 - avg_far[i - 2]):
      continue
    if not (body3 > body2 - avg_far[i - 1]):
      continue
    if not (body3 > avg_body_short[i]):
      continue

    out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_inside_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Inside Up/Down."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if np.isnan(avg_body_long[i - 2]) or np.isnan(avg_body_short[i - 1]):
      continue

    o1, c1 = open_[i - 2], close[i - 2]
    o2, c2 = open_[i - 1], close[i - 1]
    c3 = close[i]

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    top1, bot1 = (o1, c1) if o1 > c1 else (c1, o1)
    top2, bot2 = (o2, c2) if o2 > c2 else (c2, o2)

    # 1st candle: long
    # 2nd candle: short and strictly engulfed by 1st
    if (
      body1 > avg_body_long[i - 2]
      and body2 <= avg_body_short[i - 1]
      and top2 < top1
      and bot2 > bot1
    ):
      # 3rd candle color is opposite to 1st and closes outside 1st candle's OPEN
      # Bullish case (Three Inside Up): 1st is Black, 3rd is White and Close > 1st Open
      if c1 < o1 and c3 > open_[i] and c3 > o1:
        out[i] = 100
      # Bearish case (Three Inside Down): 1st is White, 3rd is Black and Close < 1st Open
      elif c1 > o1 and c3 < open_[i] and c3 < o1:
        out[i] = -100
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
    o1, c1 = open_[i - 2], close[i - 2]
    o2, c2 = open_[i - 1], close[i - 1]
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

    val = 0
    if is_up:
      val = 100
    elif is_down:
      val = -100
    out[i] = val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_line_strike_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Line Strike."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(4, n):
    if np.isnan(avg_near[i]):
      continue

    c1, o1 = close[i - 3], open_[i - 3]
    c2, o2 = close[i - 2], open_[i - 2]
    c3, o3 = close[i - 1], open_[i - 1]
    c4, o4 = close[i], open_[i]

    # Check colors: 3 same, 4th opposite
    is_white1 = c1 > o1
    is_white2 = c2 > o2
    is_white3 = c3 > o3
    is_white4 = c4 > o4

    # 3 Must be same color
    if not ((is_white1 == is_white2) & (is_white2 == is_white3)):
      continue

    # 4th must be opposite
    if is_white3 == is_white4:
      continue

    # Check 2nd opens near 1st body
    # min(o1, c1) - near <= o2 <= max(o1, c1) + near
    bot1 = min(o1, c1)
    top1 = max(o1, c1)
    if not ((o2 >= bot1 - avg_near[i]) & (o2 <= top1 + avg_near[i])):
      continue

    # Check 3rd opens near 2nd body
    bot2 = min(o2, c2)
    top2 = max(o2, c2)
    if not ((o3 >= bot2 - avg_near[i]) & (o3 <= top2 + avg_near[i])):
      continue

    if is_white1:
      # Bullish Strike logic (3 White, 1 Black)
      # TA-Lib returns 100 for Bullish Strike?
      # "3 Line Strike": If 3 Green then 1 Red engulfing -> Signal is usually continuation of trend?
      # TA-Lib returns `TA_CANDLECOLOR(i-1) * 100`. So if 3 Whites, returns 100.

      # Consecutive higher closes
      stairs = (c2 > c1) & (c3 > c2)
      # 4th opens above prior close (gap up)
      gap_up_start = o4 > c3
      # 4th closes below 1st open (engulfs all)
      engulfs = c4 < o1

      if stairs & gap_up_start & engulfs:
        out[i] = 100

    else:  # 3 Black
      # Consecutive lower closes
      stairs = (c2 < c1) & (c3 < c2)
      # 4th opens below prior close (gap down)
      gap_down_start = o4 < c3
      # 4th closes above 1st open (engulfs all)
      engulfs = c4 > o1

      if stairs & gap_down_start & engulfs:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_piercing_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Piercing Pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    if np.isnan(avg_body_long[i - 1]) or np.isnan(avg_body_long[i]):
      continue

    o1, c1, l1 = open_[i - 1], close[i - 1], low[i - 1]
    o0, c0 = open_[i], close[i]

    # 1st: long black
    # 2nd: long white
    # opens below prior low
    # closes within prior body, above midpoint
    if (
      c1 < o1
      and (o1 - c1) > avg_body_long[i - 1]
      and c0 > o0
      and (c0 - o0) > avg_body_long[i]
      and o0 < l1
      and c0 < o1
      and c0 > c1 + (o1 - c1) * 0.5
    ):
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_dark_cloud_cover_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  penetration: float,
) -> NDArray[np.int32]:
  """Detect Dark Cloud Cover."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    if np.isnan(avg_body_long[i - 1]):
      continue

    o1, h1, c1 = open_[i - 1], high[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # 1st: long white
    # 2nd: black
    # opens above prior high
    # closes within prior body, deeply enough
    if (
      c1 > o1
      and (c1 - o1) > avg_body_long[i - 1]
      and c0 < o0
      and o0 > h1
      and c0 > o1
      and c0 < c1 - (c1 - o1) * penetration
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_kicking_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  by_length: bool = False,
) -> NDArray[np.int32]:
  """Detect Kicking Pattern (stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    if np.isnan(avg_body_long[i]):
      continue

    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    body1 = abs(c1 - o1)
    body0 = abs(c0 - o0)

    # 1st marubozu
    is_maru1 = (
      body1 > avg_body_long[i - 1]
      and (high[i - 1] - max(o1, c1)) < avg_shadow_very_short[i - 1]
      and (min(o1, c1) - low[i - 1]) < avg_shadow_very_short[i - 1]
    )

    # 2nd marubozu
    is_maru2 = (
      body0 > avg_body_long[i]
      and (high[i] - max(o0, c0)) < avg_shadow_very_short[i]
      and (min(o0, c0) - low[i]) < avg_shadow_very_short[i]
    )

    if is_maru1 and is_maru2 and (c1 > o1) != (c0 > o0):
      # Gap check per TA-Lib:
      # If Black then White: low(i) > high(i-1) (Gap Up)
      # If White then Black: high(i) < low(i-1) (Gap Down)
      if (c1 < o1 and low[i] > high[i - 1]) or (c1 > o1 and high[i] < low[i - 1]):
        if by_length:
          # TA_CANDLECOLOR( ( body0 >= body1 ? i : i-1 ) ) * 100
          if body0 >= body1:
            out[i] = 100 if c0 > o0 else -100
          else:
            out[i] = 100 if c1 > o1 else -100
        else:
          out[i] = 100 if c0 > o0 else -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_matching_low_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Matching Low (Bullish Reversal)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    if np.isnan(avg_equal[i - 1]):
      continue

    c1, o1 = close[i - 1], open_[i - 1]
    c2, o2 = close[i], open_[i]

    # TA-Lib logic:
    # 1. 1st Black
    # 2. 2nd Black
    # 3. Same Close (within Equal tolerance)
    if (
      c1 < o1
      and c2 < o2
      and c2 <= c1 + avg_equal[i - 2]
      and c2 >= c1 - avg_equal[i - 2]
    ):
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_spinning_top_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Spinning Top (Indecision)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body[i]):
      continue

    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]

    if rng < 1e-9:
      continue

    upper_shadow = high[i] - max(open_[i], close[i])
    lower_shadow = min(open_[i], close[i]) - low[i]

    is_small_body = body < avg_body[i]
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
  avg_body: NDArray[np.float64],
  avg_shadow: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Rickshaw Man (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body[i]):
      continue

    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]

    upper_shadow = high[i] - max(open_[i], close[i])
    lower_shadow = min(open_[i], close[i]) - low[i]

    is_doji = body <= avg_body[i]
    long_upper = upper_shadow > avg_shadow[i]
    long_lower = lower_shadow > avg_shadow[i]

    # TA-Lib Rickshaw Man midpoint check:
    # min(o, c) <= low + range/2 + near_avg AND max(o, c) >= low + range/2 - near_avg
    midpoint = low[i] + rng * 0.5
    near_mid = (
      min(open_[i], close[i]) <= midpoint + avg_near[i]
      and max(open_[i], close[i]) >= midpoint - avg_near[i]
    )

    if is_doji & long_upper & long_lower & near_mid:
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_high_wave_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body: NDArray[np.float64],
  avg_shadow: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect High Wave (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body[i]):
      continue

    body = abs(close[i] - open_[i])
    upper_shadow = high[i] - max(open_[i], close[i])
    lower_shadow = min(open_[i], close[i]) - low[i]

    is_small_body = body < avg_body[i]
    long_upper = upper_shadow > avg_shadow[i]
    long_lower = lower_shadow > avg_shadow[i]

    if is_small_body & long_upper & long_lower:
      out[i] = 100 if close[i] >= open_[i] else -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_long_legged_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Long Legged Doji."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(n):
    if np.isnan(avg_body_doji[i]):
      continue
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body = abs(c - o)

    # Doji body
    if body <= avg_body_doji[i]:
      # Long shadows: TA-Lib compares either shadow to RealBody * 1.0 (ShadowLong)
      upper = h - (o if o > c else c)
      lower = (o if o < c else c) - l
      if upper > body or lower > body:
        out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tristar_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tristar pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_body_doji[i]):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # TA-Lib uses same average (calculated at starting candle-1) for all three dojis
    # Loop i refers to 3rd candle. startIdx-2 in C corresponds to i-2 here.
    # Total summed in C up to (i-2)-1 = i-3.
    avg_ref = avg_body_doji[i - 2]

    if abs(c2 - o2) <= avg_ref and abs(c1 - o1) <= avg_ref and abs(c0 - o0) <= avg_ref:
      # Bearish: 2nd gaps up (Bodies), 3rd top not higher than 2nd top
      if min(o1, c1) > max(o2, c2) and max(o0, c0) < max(o1, c1):
        out[i] = -100
      # Bullish: 2nd gaps down (Bodies), 3rd bot not lower than 2nd bot
      elif max(o1, c1) < min(o2, c2) and min(o0, c0) > min(o1, c1):
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tasuki_gap_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tasuki Gap."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_near[i - 1]):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # Upside Gap
    # 1. Gap Up (Body Gap)
    # 2. 2nd White, 3rd Black
    if (
      (min(o1, c1) > max(o2, c2))
      and (c1 > o1)
      and (c0 < o0)
      and (o0 < c1)
      and (o0 > o1)
      and (c0 < o1)
      and (c0 > max(c2, o2))
      and (abs(abs(c1 - o1) - abs(c0 - o0)) < avg_near[i - 1])
    ):
      out[i] = 100
      continue

    # Downside Gap
    if (
      (max(o1, c1) < min(o2, c2))
      and (c1 < o1)
      and (c0 > o0)
      and (o0 < o1)
      and (o0 > c1)
      and (c0 > o1)
      and (c0 < min(c2, o2))
      and (abs(abs(o1 - c1) - abs(c0 - o0)) < avg_near[i - 1])
    ):
      out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_gap_side_by_side_white_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_near: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Gap Side-by-Side White."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if np.isnan(avg_near[i]) or np.isnan(avg_equal[i]):
      continue

    # Gap Logic
    # Upside gap: open(i-1) > high(i-2) (Strict) or close(i-2)?
    # TA-Lib uses RealBodyGap. RealBodyGapUp(i-1, i-2) => BodyBottom(i-1) > BodyTop(i-2)

    c1, o1 = close[i - 2], open_[i - 2]
    c2, o2 = close[i - 1], open_[i - 1]
    c3, o3 = close[i], open_[i]

    top1 = max(o1, c1)
    bot1 = min(o1, c1)
    top2 = max(o2, c2)
    bot2 = min(o2, c2)
    top3 = max(o3, c3)
    bot3 = min(o3, c3)

    # 2nd and 3rd must be White (Bullish)
    is_white2 = c2 > o2
    is_white3 = c3 > o3

    if not (is_white2 & is_white3):
      continue

    # Real Body Gap Up: Bottom of 2 and 3 > Top of 1
    # Note: TA-Lib checks separate gaps for both candles against i-2
    gap_up = (bot2 > top1) & (bot3 > top1)
    gap_down = (top2 < bot1) & (top3 < bot1)

    if not (gap_up or gap_down):
      continue

    # Size check: Body3 within "Near" of Body2
    # TA-Lib: TA_CANDLEAVERAGE( Near, ..., i-1 ) where total summed up to i-2.
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)
    size_ok = abs(body3 - body2) < avg_near[i - 1]

    # Open check: Open3 within "Equal" of Open2
    open_ok = abs(o3 - o2) < avg_equal[i - 1]

    if size_ok & open_ok:
      out[i] = 100 if gap_up else -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_two_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Two Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_body_long[i]):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # white, black, black
    if (
      c2 > o2
      and (c2 - o2) > avg_body_long[i - 2]
      and c1 < o1
      and min(o1, c1) > max(o2, c2)
      and c0 < o0
      and o0 < o1
      and o0 > c1
      and c0 < c2
      and c0 > o2
    ):
      out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_upside_gap_two_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Upside Gap Two Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if np.isnan(avg_body_long[i - 2]) or np.isnan(avg_body_short[i - 1]):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # TA-Lib logic:
    # 1st: White Long
    # 2nd: Black Short, RealBodyGapUp
    # 3rd: Black Engulfing 2nd's Real Body, closing above 1st's Close

    body2 = abs(c2 - o2)
    body1 = abs(c1 - o1)

    if (
      c2 > o2
      and body2 > avg_body_long[i - 2]
      and c1 < o1
      and body1 <= avg_body_short[i - 1]
      and min(o1, c1) > c2
      and c0 < o0
      and o0 > o1
      and c0 < c1
      and c0 > c2
    ):
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
def detect_abandoned_baby_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Abandoned Baby."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if (
      np.isnan(avg_body_long[i])
      or np.isnan(avg_body_doji[i])
      or np.isnan(avg_body_short[i])
    ):
      continue

    # 1. 1st Candle Long
    body1 = abs(close[i - 2] - open_[i - 2])
    if not (body1 > avg_body_long[i - 2]):
      continue

    # 2. 2nd Candle Doji
    body2 = abs(close[i - 1] - open_[i - 1])
    if not (body2 <= avg_body_doji[i - 1]):
      continue

    # 3. 3rd Candle Longer than Short
    body3 = abs(close[i] - open_[i])
    if not (body3 > avg_body_short[i]):
      continue

    c1, o1 = close[i - 2], open_[i - 2]
    c2, o2 = close[i - 1], open_[i - 1]
    c3, o3 = close[i], open_[i]
    h1, l1 = high[i - 2], low[i - 2]
    h2, l2 = high[i - 1], low[i - 1]
    h3, l3 = high[i], low[i]

    is_white1 = c1 > o1
    is_white3 = c3 > o3

    # Bearish Abandoned Baby (Top)
    if is_white1 and (not is_white3):
      gap_up = l2 > h1
      gap_down = h3 < l2
      penetrate = c3 < (c1 - body1 * penetration)
      if gap_up and gap_down and penetrate:
        out[i] = -100

    # Bullish Abandoned Baby (Bottom)
    elif (not is_white1) and is_white3:
      gap_down = h2 < l1
      gap_up = l3 > h2
      penetrate = c3 > (c1 + body1 * penetration)
      if gap_down and gap_up and penetrate:
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_advance_block_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_short: NDArray[np.float64],
  avg_shadow_long: NDArray[np.float64],
  avg_far: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Advance Block."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_body_long[i]):
      continue

    o2, h2, c2 = open_[i - 2], high[i - 2], close[i - 2]
    o1, h1, c1 = open_[i - 1], high[i - 1], close[i - 1]
    o0, h0, c0 = open_[i], high[i], close[i]

    body2 = c2 - o2
    body1 = c1 - o1
    body0 = c0 - o0

    if (
      c2 > o2
      and c1 > o1
      and c0 > o0
      and c0 > c1
      and c1 > c2
      and o1 > o2
      and o1 <= c2 + avg_near[i - 2]
      and o0 > o1
      and o0 <= c1 + avg_near[i - 1]
      and body2 > avg_body_long[i - 2]
      and (h2 - c2) < avg_shadow_short[i - 2]
    ):
      blocked = False
      # ( 2 far smaller than 1 && 3 not longer than 2 )
      if body1 < body2 - avg_far[i - 2] and body0 < body1 + avg_near[i - 1]:
        blocked = True
      # 3 far smaller than 2
      elif body0 < body1 - avg_far[i - 1]:
        blocked = True
      # ( 3 smaller than 2 && 2 smaller than 1 && (3 or 2 not short upper shadow) )
      elif (
        body0 < body1
        and body1 < body2
        and ((h1 - c1) > avg_shadow_short[i - 1] or (h0 - c0) > avg_shadow_short[i])
      ):
        blocked = True
      # ( 3 smaller than 2 && 3 long upper shadow )
      elif body0 < body1 and (h0 - c0) > body0:  # TA-Lib ShadowLong P=0 uses RealBody
        blocked = True

      if blocked:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_belt_hold_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Belt Hold."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body_long[i]):
      continue

    body = abs(close[i] - open_[i])
    # Must be Long Body
    if body <= avg_body_long[i]:
      continue

    if close[i] > open_[i]:  # Bull (White Opening Marubozu)
      # No lower shadow
      if (open_[i] - low[i]) < avg_shadow_very_short[i]:
        out[i] = 100
    else:  # Bear (Black Opening Marubozu)
      # No upper shadow
      if (high[i] - open_[i]) < avg_shadow_very_short[i]:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_breakaway_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Breakaway."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(4, n):
    if np.isnan(avg_body_long[i]):
      continue

    c1, o1 = close[i - 4], open_[i - 4]

    # 1. First Candle Long
    body1 = abs(c1 - o1)
    if not (body1 > avg_body_long[i - 4]):
      continue

    c2, o2 = close[i - 3], open_[i - 3]
    c3, o3 = close[i - 2], open_[i - 2]
    c4, o4 = close[i - 1], open_[i - 1]
    c5, o5 = close[i], open_[i]

    h2, l2 = high[i - 3], low[i - 3]
    h3, l3 = high[i - 2], low[i - 2]
    h4, l4 = high[i - 1], low[i - 1]

    is_white1 = c1 > o1
    is_white2 = c2 > o2
    is_white4 = c4 > o4
    is_white5 = c5 > o5

    # Check Colors: 1, 2, 4 same. 5 opposite.
    if (is_white1 != is_white2) or (is_white2 != is_white4):
      continue

    # Bearish Breakaway (1st White) -> Expects 5th Black
    if is_white1:
      if is_white5:
        continue

      # 2nd gaps up (Real Body Gap)
      gap_up = min(o2, c2) > max(o1, c1)
      if not gap_up:
        continue

      # 3rd higher High/Low than 2nd
      if not ((h3 > h2) & (l3 > l2)):
        continue

      # 4th higher High/Low than 3rd
      if not ((h4 > h3) & (l4 > l3)):
        continue

      # 5th closes inside gap (Close < Open2, Close > Close1)
      if (c5 < o2) & (c5 > c1):
        out[i] = -100

    # Bullish Breakaway (1st Black) -> Expects 5th White
    else:
      if not is_white5:
        continue

      # 2nd gaps down (Body Gap)
      gap_down = max(o2, c2) < min(o1, c1)
      if not gap_down:
        continue

      # 3rd lower High/Low than 2nd
      if not ((h3 < h2) & (l3 < l2)):
        continue

      # 4th lower High/Low than 3rd
      if not ((h4 < h3) & (l4 < l3)):
        continue

      # 5th closes inside gap (Close > Open2, Close < Close1)
      if (c5 > o2) & (c5 < c1):
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_closing_marubozu_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Closing Marubozu.

  Definition:
  - Body is Long ( > avg_body_long)
  - No shadow at the closing end ( < avg_shadow_very_short)
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body_long[i]):
      continue

    body = abs(close[i] - open_[i])

    # Check 1: Long Body
    if body <= avg_body_long[i]:
      continue

    # Check 2: No shadow at closing end
    if close[i] > open_[i]:  # Bull
      upper_shadow = high[i] - close[i]
      if upper_shadow < avg_shadow_very_short[i]:
        out[i] = 100
    else:  # Bear
      lower_shadow = close[i] - low[i]
      if lower_shadow < avg_shadow_very_short[i]:
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_dragonfly_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Dragonfly Doji (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body_doji[i]):
      continue

    body = abs(close[i] - open_[i])
    upper_shadow = high[i] - max(open_[i], close[i])
    lower_shadow = min(open_[i], close[i]) - low[i]

    if (
      (body < avg_body_doji[i])
      and (upper_shadow < avg_shadow_very_short[i])
      and (lower_shadow > avg_shadow_very_short[i])
    ):
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_gravestone_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Gravestone Doji (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body_doji[i]):
      continue

    body = abs(close[i] - open_[i])
    upper_shadow = high[i] - max(open_[i], close[i])
    lower_shadow = min(open_[i], close[i]) - low[i]

    if (
      (body < avg_body_doji[i])
      and (lower_shadow < avg_shadow_very_short[i])
      and (upper_shadow > avg_shadow_very_short[i])
    ):
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hikkake_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hikkake pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  pattern_idx = 0
  pattern_result = 0

  for i in range(2, n):
    # New breakout check
    if (
      high[i - 1] < high[i - 2]
      and low[i - 1] > low[i - 2]
      and (
        (high[i] < high[i - 1] and low[i] < low[i - 1])
        or (high[i] > high[i - 1] and low[i] > low[i - 1])
      )
    ):
      pattern_result = 100 if high[i] < high[i - 1] else -100
      pattern_idx = i
      out[i] = pattern_result
    else:
      # Confirmation check
      if pattern_idx != 0 and i <= pattern_idx + 3:
        if (pattern_result > 0 and close[i] > high[pattern_idx - 1]) or (
          pattern_result < 0 and close[i] < low[pattern_idx - 1]
        ):
          # Confirmed: return +/- 200
          out[i] = pattern_result + (100 if pattern_result > 0 else -100)
          pattern_idx = 0
        else:
          out[i] = 0
      else:
        out[i] = 0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_homing_pigeon_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Homing Pigeon. Both candles black, second inside first."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]

    # 1. Body check
    b1 = o1 - c1
    b2 = o2 - c2
    if not (b1 > avg_body_long[i - 1] and b2 > 0 and b2 < avg_body_short[i]):
      continue

    # 2. Containment (strict for Homing Pigeon)
    if (o2 < o1) and (c2 > c1):
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_identical_three_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Identical Three Crows."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_shadow_very_short[i]) or np.isnan(avg_equal[i - 1]):
      continue

    o2, c2, l2 = open_[i - 2], close[i - 2], low[i - 2]
    o1, c1, l1 = open_[i - 1], close[i - 1], low[i - 1]
    o0, c0, l0 = open_[i], close[i], low[i]

    # All black, declining closes
    if c2 < o2 and c1 < o1 and c0 < o0 and c2 > c1 and c1 > c0:
      # Very short lower shadows
      if (
        (c2 - l2) < avg_shadow_very_short[i - 2]
        and (c1 - l1) < avg_shadow_very_short[i - 1]
        and (c0 - l0) < avg_shadow_very_short[i]
      ):
        # Identical opening (2nd and 3rd open near previous close)
        # TA-Lib uses Equal sum ending at i-3 (sum(j-2) for j up to i-1)
        if (
          o1 <= c2 + avg_equal[i - 2]
          and o1 >= c2 - avg_equal[i - 2]
          and o0 <= c1 + avg_equal[i - 1]
          and o0 >= c1 - avg_equal[i - 1]
        ):
          out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_in_neck_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect In-Neck."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    if np.isnan(avg_body_long[i - 1]) or np.isnan(avg_equal[i - 1]):
      continue

    o1, c1, l1 = open_[i - 1], close[i - 1], low[i - 1]
    o0, c0 = open_[i], close[i]

    if (
      c1 < o1
      and (o1 - c1) > avg_body_long[i - 1]
      and c0 > o0
      and o0 < l1
      and c0 <= c1 + avg_equal[i - 1]
      and c0 >= c1
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_ladder_bottom_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Ladder Bottom."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(4, n):
    if np.isnan(avg_shadow_very_short[i]):
      continue

    c1, o1 = close[i - 4], open_[i - 4]
    c2, o2 = close[i - 3], open_[i - 3]
    c3, o3 = close[i - 2], open_[i - 2]
    c4, o4 = close[i - 1], open_[i - 1]
    c5, o5 = close[i], open_[i]

    # 1. 3 Black Candles
    bears3 = (c1 < o1) & (c2 < o2) & (c3 < o3)
    if not bears3:
      continue

    # 2. Lower Opens and Closes
    lower_opens = (o2 < o1) & (o3 < o2)
    lower_closes = (c2 < c1) & (c3 < c2)
    if not (lower_opens & lower_closes):
      continue

    # 3. 4th Candle: Black with Upper Shadow > VeryShort
    if not (c4 < o4):
      continue

    # Check upper shadow
    upper_shadow4 = high[i - 1] - o4
    if not (upper_shadow4 > avg_shadow_very_short[i - 1]):
      continue

    # 4. 5th Candle: White
    if not (c5 > o5):
      continue

    # 5. Gap Up Logic?
    # Open > Prior Open (Open5 > Open4)
    # Close > Prior High (Close5 > High4)
    if (o5 > o4) & (c5 > high[i - 1]):
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_long_line_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Long Line."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(n):
    if np.isnan(avg_body_long[i]):
      continue

    o, h, l, c = open_[i], high[i], low[i], close[i]
    body = abs(c - o)
    body_top = o if o > c else c
    body_bot = o if o < c else c

    # 1. Body must be Long
    # 2. Both shadows must be Short
    if (
      body > avg_body_long[i]
      and (h - body_top) < avg_shadow_short[i]
      and (body_bot - l) < avg_shadow_short[i]
    ):
      out[i] = 100 if c > o else -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_mat_hold_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  penetration: float = 0.5,
) -> NDArray[np.int32]:
  """Detect Mat Hold."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(4, n):
    if (
      np.isnan(avg_body_long[i - 4])
      or np.isnan(avg_body_short[i - 3])
      or np.isnan(avg_body_short[i - 2])
      or np.isnan(avg_body_short[i - 1])
    ):
      continue

    o4, h4, l4, c4 = open_[i - 4], high[i - 4], low[i - 4], close[i - 4]
    o3, h3, c3 = open_[i - 3], high[i - 3], close[i - 3]
    o2, h2, c2 = open_[i - 2], high[i - 2], close[i - 2]
    o1, h1, c1 = open_[i - 1], high[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    body4 = abs(c4 - o4)
    body3 = abs(c3 - o3)
    body2 = abs(c2 - o2)
    body1 = abs(c1 - o1)

    # TA-Lib logic:
    # 1. 1st: Long white
    # 2. 2nd: Small black, Gap UP (Real Body)
    # 3. 3rd-4th: Small reaction days
    # 4. 2nd-4th hold within 1st body (allow penetration %)
    # 5. 2nd-4th are falling
    # 6. 5th: white confirms
    h3, h2, h1 = high[i - 3], high[i - 2], high[i - 1]
    if (
      c4 > o4
      and body4 > avg_body_long[i - 4]
      and c3 < o3
      and body3 < avg_body_short[i - 3]
      and body2 < avg_body_short[i - 2]
      and body1 < avg_body_short[i - 1]
      and c0 > o0
      and min(o3, c3) > max(o4, c4)
      and min(o2, c2) < c4
      and min(o2, c2) > c4 - body4 * penetration
      and min(o1, c1) < c4
      and min(o1, c1) > c4 - body4 * penetration
      and max(o2, c2) < o3
      and max(o1, c1) < max(o2, c2)
      and o0 > c1
      and c0 > max(h3, max(h2, h1))
    ):
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_on_neck_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect On-Neck."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    if np.isnan(avg_body_long[i - 1]) or np.isnan(avg_equal[i - 1]):
      continue

    o1, c1, l1 = open_[i - 1], close[i - 1], low[i - 1]
    o0, c0 = open_[i], close[i]

    if (
      c1 < o1
      and (o1 - c1) > avg_body_long[i - 1]
      and c0 > o0
      and o0 < l1
      and c0 <= l1 + avg_equal[i - 1]
      and c0 >= l1 - avg_equal[i - 1]
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_rise_fall_three_methods_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Rise/Fall Three Methods."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(4, n):
    if (
      np.isnan(avg_body_long[i - 4])
      or np.isnan(avg_body_long[i])
      or np.isnan(avg_body_short[i - 3])
      or np.isnan(avg_body_short[i - 2])
      or np.isnan(avg_body_short[i - 1])
    ):
      continue

    o4, h4, l4, c4 = open_[i - 4], high[i - 4], low[i - 4], close[i - 4]
    o3, c3 = open_[i - 3], close[i - 3]
    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    body4 = abs(c4 - o4)
    body3 = abs(c3 - o3)
    body2 = abs(c2 - o2)
    body1 = abs(c1 - o1)
    body0 = abs(c0 - o0)

    # Rising Three Methods
    if (
      c4 > o4
      and body4 > avg_body_long[i - 4]
      and c3 < o3
      and body3 < avg_body_short[i - 3]
      and c2 < o2
      and body2 < avg_body_short[i - 2]
      and c1 < o1
      and body1 < avg_body_short[i - 1]
      and c0 > o0
      and body0 > avg_body_long[i]
      and min(o3, c3) < h4
      and max(o3, c3) > l4
      and min(o2, c2) < h4
      and max(o2, c2) > l4
      and min(o1, c1) < h4
      and max(o1, c1) > l4
      and c3 > c2
      and c2 > c1
      and o0 > c1
      and c0 > c4
    ):
      out[i] = 100

    # Falling Three Methods
    elif (
      c4 < o4
      and body4 > avg_body_long[i - 4]
      and c3 > o3
      and body3 < avg_body_short[i - 3]
      and c2 > o2
      and body2 < avg_body_short[i - 2]
      and c1 > o1
      and body1 < avg_body_short[i - 1]
      and c0 < o0
      and body0 > avg_body_long[i]
      and min(o3, c3) < h4
      and max(o3, c3) > l4
      and min(o2, c2) < h4
      and max(o2, c2) > l4
      and min(o1, c1) < h4
      and max(o1, c1) > l4
      and c3 < c2
      and c2 < c1
      and o0 < c1
      and c0 < c4
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_short_line_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  avg_shadow_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Short Line (Stateful)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(n):
    if np.isnan(avg_body_short[i]):
      continue
    body = abs(close[i] - open_[i])
    upper = high[i] - max(open_[i], close[i])
    lower = min(open_[i], close[i]) - low[i]
    if (
      (body < avg_body_short[i])
      and (upper < avg_shadow_short[i])
      and (lower < avg_shadow_short[i])
    ):
      out[i] = 100 if close[i] >= open_[i] else -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_stalled_pattern_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Stalled Pattern."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(2, n):
    if (
      np.isnan(avg_body_long[i - 2])
      or np.isnan(avg_body_long[i - 1])
      or np.isnan(avg_body_short[i])
      or np.isnan(avg_shadow_very_short[i - 1])
      or np.isnan(avg_near[i - 1])
      or np.isnan(avg_near[i])
    ):
      continue

    o2, h2, l2, c2 = open_[i - 2], high[i - 2], low[i - 2], close[i - 2]
    o1, h1, l1, c1 = open_[i - 1], high[i - 1], low[i - 1], close[i - 1]
    o0, h0, l0, c0 = open_[i], high[i], low[i], close[i]

    body2 = c2 - o2
    body1 = c1 - o1
    body0 = c0 - o0

    if (
      c2 > o2
      and c1 > o1
      and c0 > o0
      and c0 > c1
      and c1 > c2
      and o1 > o2
      and o1 <= c2 + avg_near[i - 2]
      and body2 > avg_body_long[i - 2]
      and body1 > avg_body_long[i - 1]
      and (h1 - c1) < avg_shadow_very_short[i - 1]
      and body0 < avg_body_short[i]
      and o0 >= c1 - body0 - avg_near[i - 1]
    ):
      out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_stick_sandwich_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Stick Sandwich."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_equal[i - 2]) or np.isnan(avg_body_short[i]):
      continue

    c1, o1 = close[i - 2], open_[i - 2]
    c2, o2 = close[i - 1], open_[i - 1]
    c3, o3 = close[i], open_[i]
    l2 = low[i - 1]

    is_bear1 = c1 < o1
    is_bull2 = c2 > o2
    is_bear3 = c3 < o3

    if is_bear1 & is_bull2 & is_bear3:
      # 2nd: long (BodyShort) relative to i-1
      # TA-Lib behavior suggests this check is effectively absent or 0.
      # if not (abs(c2-o2) > avg_body_short[i-1]):
      #   continue

      # 2nd low > 1st close (Strict per TA-Lib)
      if not (l2 > c1):
        continue

      if abs(c3 - c1) <= avg_equal[i - 2]:
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_takuri_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  avg_shadow_very_long: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Takuri Doji (Dragonfly with very long lower shadow)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    if np.isnan(avg_body_doji[i]):
      continue

    o = open_[i]
    c = close[i]
    h = high[i]
    l = low[i]

    body_top = o if o > c else c
    real_body = abs(c - o)
    upper_shadow = h - body_top
    lower_shadow = min(o, c) - l

    if (
      (real_body < avg_body_doji[i])
      and (upper_shadow < avg_shadow_very_short[i])
      and (lower_shadow >= avg_shadow_very_long[i])
    ):
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_thrusting_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Thrusting Pattern (Bearish Trend)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    if np.isnan(avg_body_long[i - 1]) or np.isnan(avg_equal[i - 1]):
      continue

    o1, c1, l1 = open_[i - 1], close[i - 1], low[i - 1]
    o0, c0 = open_[i], close[i]

    if (
      c1 < o1
      and (o1 - c1) > avg_body_long[i - 1]
      and c0 > o0
      and o0 < l1
      and c0 >= c1 + avg_equal[i - 1]
      and c0 <= c1 + (o1 - c1) * 0.5
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_unique_three_river_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Unique 3 River."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_body_long[i - 2]) or np.isnan(avg_body_short[i]):
      continue

    o2, l2, c2 = open_[i - 2], low[i - 2], close[i - 2]
    o1, l1, c1 = open_[i - 1], low[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # TA-Lib logic:
    # 1st: Long Black
    # 2nd: Black Harami (Close > 1st Close, Open <= 1st Open) with Lower Low
    # 3rd: Short White with Open > 2nd Low

    body2 = abs(c2 - o2)
    body0 = abs(c0 - o0)

    if (
      body2 > avg_body_long[i - 2]
      and c2 < o2
      and c1 < o1
      and c1 > c2
      and o1 <= o2
      and l1 < l2
      and body0 < avg_body_short[i]
      and c0 > o0
      and o0 > l1
    ):
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_separating_lines_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Separating Lines."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    if (
      np.isnan(avg_equal[i - 2])
      or np.isnan(avg_body_long[i])
      or np.isnan(avg_shadow_very_short[i])
    ):
      continue

    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]
    h2, l2 = high[i], low[i]

    # TA-Lib Logic:
    # 1. Opposite Colors
    is_bull1 = c1 > o1
    is_bull2 = c2 > o2
    if is_bull1 == is_bull2:
      continue

    # 2. Same Open
    # Use i-2 (Immediately preceding the pattern)
    if not (abs(o1 - o2) <= avg_equal[i - 2]):
      continue

    # 3. 2nd is Long Body (vs Average at i)
    if not (abs(c2 - o2) > avg_body_long[i]):
      continue

    # 4. Shadow Very Short (vs Average at i)
    # Bearish: Bull then Bear. Open2=High, so Upper Shadow 2.
    if is_bull1 and (not is_bull2):
      if (h2 - o2) < avg_shadow_very_short[i]:
        out[i] = -100

    # Bullish: Bear then Bull. Open2=Low, so Lower Shadow 2.
    elif (not is_bull1) and is_bull2:
      if (o2 - l2) < avg_shadow_very_short[i]:
        out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_counterattack_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_equal: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Counterattack."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    if np.isnan(avg_equal[i]) or np.isnan(avg_body_long[i]):
      continue

    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]

    # 1. Opposite Colors
    is_bull1 = c1 > o1
    is_bull2 = c2 > o2
    if is_bull1 == is_bull2:
      continue

    # 2. Both Long bodies (Strict TA-Lib check)
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    # TA-Lib uses BodyLong average from previous bars but array index logic:
    # TA_CANDLEAVERAGE(BodyLong, ..., i-1) for 1st
    # TA_CANDLEAVERAGE(BodyLong, ..., i) for 2nd
    if not (body1 > avg_body_long[i - 1]):
      continue
    if not (body2 > avg_body_long[i]):
      continue

    # 3. Equal Closes
    same_close = abs(c1 - c2) <= avg_equal[i - 1]
    if not same_close:
      continue

    if is_bull2:  # Bear then Bull (Bullish Counter)
      out[i] = 100
    else:  # Bull then Bear (Bearish Counter)
      out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Doji Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    if np.isnan(avg_body_long[i - 1]) or np.isnan(avg_body_doji[i]):
      continue

    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    # 1st candle: body long
    # 2nd candle: doji
    if body1 > avg_body_long[i - 1] and body2 <= avg_body_doji[i]:
      # Gap check (RealBodyGapUp or Down)
      top1 = o1 if o1 > c1 else c1
      bot1 = o1 if o1 < c1 else c1
      top2 = o2 if o2 > c2 else c2
      bot2 = o2 if o2 < c2 else c2

      # 1st Bullish & 2nd gaps up
      if c1 > o1 and bot2 > top1:
        out[i] = -100
      # 1st Bearish & 2nd gaps down
      elif c1 < o1 and top2 < bot1:
        out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_conceal_baby_swallow_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Concealing Baby Swallow."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(3, n):
    if np.isnan(avg_shadow_very_short[i]):
      continue

    c1, o1 = close[i - 3], open_[i - 3]
    c2, o2 = close[i - 2], open_[i - 2]
    c3, o3 = close[i - 1], open_[i - 1]
    c4, o4 = close[i], open_[i]

    # 1. Black Marubozu
    if not (c1 < o1):
      continue
    l1, h1 = low[i - 3], high[i - 3]
    shad1_u = h1 - o1
    shad1_l = c1 - l1
    if not (
      (shad1_u < avg_shadow_very_short[i - 3])
      & (shad1_l < avg_shadow_very_short[i - 3])
    ):
      continue

    # 2. Black Marubozu
    if not (c2 < o2):
      continue
    l2, h2 = low[i - 2], high[i - 2]
    shad2_u = h2 - o2
    shad2_l = c2 - l2
    if not (
      (shad2_u < avg_shadow_very_short[i - 2])
      & (shad2_l < avg_shadow_very_short[i - 2])
    ):
      continue

    # 3. Black, Inverted Hammer-like (long upper shadow)
    if not (c3 < o3):
      continue
    l3, h3 = low[i - 1], high[i - 1]
    # Gap Down (Open < Prior Close)
    if not (o3 < c2):
      continue
    # High invades prior body (High3 > Close2)
    if not (h3 > c2):
      continue

    # 4. Black, Engulfing
    if not (c4 < o4):
      continue
    l4, h4 = low[i], high[i]
    # Engulfs 3rd (High4 > High3, Low4 < Low3) strict?
    # TA-Lib: Open > High3, Close < Low3
    if not ((o4 > h3) & (c4 < l3)):
      continue

    out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_harami_cross_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Harami Cross."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(1, n):
    pc, po = close[i - 1], open_[i - 1]
    cc, co = close[i], open_[i]

    # 1. First candle is BodyLong
    p_body = abs(pc - po)
    if p_body < avg_body_long[i - 1]:
      continue

    # 2. Second candle is Doji
    c_body = abs(cc - co)
    if c_body >= avg_body_doji[i]:
      continue

    # Signal based on FIRST candle color
    is_bull = pc < po
    is_bear = pc > po

    prev_top = po if po > pc else pc
    prev_bot = po if po < pc else pc
    curr_top = co if co > cc else cc
    curr_bot = co if co < cc else cc

    # Doji inside previous body
    # Grade 100: Strictly inside
    # Grade 80: Semi-strict (allow matching edges)
    if (curr_top < prev_top) and (curr_bot > prev_bot):
      out[i] = 100 if pc < po else -100
    elif (curr_top <= prev_top) and (curr_bot >= prev_bot):
      out[i] = 80 if pc < po else -80
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hikkake_modified_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_near: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Modified Hikkake."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  pattern_idx = 0
  pattern_result = 0

  for i in range(3, n):
    # New breakout check: double inside + breakout
    near = avg_near[i - 2]
    if (
      high[i - 2] < high[i - 3]
      and low[i - 2] > low[i - 3]
      and high[i - 1] < high[i - 2]
      and low[i - 1] > low[i - 2]
      and (
        (
          high[i] < high[i - 1]
          and low[i] < low[i - 1]
          and close[i - 2] <= low[i - 2] + near
        )
        or (
          high[i] > high[i - 1]
          and low[i] > low[i - 1]
          and close[i - 2] >= high[i - 2] - near
        )
      )
    ):
      pattern_result = 100 if high[i] < high[i - 1] else -100
      pattern_idx = i
      out[i] = pattern_result
    else:
      # Confirmation check
      if pattern_idx != 0 and i <= pattern_idx + 3:
        if (pattern_result > 0 and close[i] > high[pattern_idx - 1]) or (
          pattern_result < 0 and close[i] < low[pattern_idx - 1]
        ):
          # Confirmed
          out[i] = pattern_result + (100 if pattern_result > 0 else -100)
          pattern_idx = 0
        else:
          out[i] = 0
      else:
        out[i] = 0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_morning_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Morning Doji Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if (
      np.isnan(avg_body_long[i - 2])
      or np.isnan(avg_body_doji[i - 1])
      or np.isnan(avg_body_short[i])
    ):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # TA-Lib logic:
    # 1st: Long Black
    # 2nd: Doji, Gap Down (Real Body)
    # 3rd: Longer than Short, White
    # 3rd closes well within 1st RB (at least penetration %)

    body2 = abs(c2 - o2)
    body1 = abs(c1 - o1)
    body0 = abs(c0 - o0)

    if (
      body2 > avg_body_long[i - 2]
      and c2 < o2
      and body1 <= avg_body_doji[i - 1]
      and max(o1, c1) < min(o2, c2)
      and body0 > avg_body_short[i]
      and c0 > o0
      and c0 > c2 + body2 * penetration
    ):
      out[i] = 100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_evening_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_body_doji: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Evening Doji Star."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if (
      np.isnan(avg_body_long[i - 2])
      or np.isnan(avg_body_doji[i - 1])
      or np.isnan(avg_body_short[i])
    ):
      continue

    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # TA-Lib logic:
    # 1st: Long White
    # 2nd: Doji, Gap Up (Real Body)
    # 3rd: Longer than Short, Black
    # 3rd closes well within 1st RB (at least penetration %)

    body2 = abs(c2 - o2)
    body1 = abs(c1 - o1)
    body0 = abs(c0 - o0)

    if (
      body2 > avg_body_long[i - 2]
      and c2 > o2
      and body1 <= avg_body_doji[i - 1]
      and min(o1, c1) > max(o2, c2)
      and body0 > avg_body_short[i]
      and c0 < o0
      and c0 < c2 - body2 * penetration
    ):
      out[i] = -100
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_stars_in_south_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  avg_body_long: NDArray[np.float64],
  avg_shadow_long: NDArray[np.float64],
  avg_shadow_very_short: NDArray[np.float64],
  avg_body_short: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Stars In The South."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  for i in range(2, n):
    if np.isnan(avg_body_long[i]):
      continue

    o2, l2, c2 = open_[i - 2], low[i - 2], close[i - 2]
    o1, h1, l1, c1 = open_[i - 1], high[i - 1], low[i - 1], close[i - 1]
    o0, h0, l0, c0 = open_[i], high[i], low[i], close[i]

    # all black
    if (
      c2 < o2
      and c1 < o1
      and c0 < o0
      and (o2 - c2) > avg_body_long[i - 2]
      and (c2 - l2) > avg_shadow_long[i - 2]
      and (o1 - c1) < (o2 - c2)
      and o1 > c2
      and o1 <= high[i - 2]
      and l1 < c2
      and l1 >= l2
      and (c1 - l1) > avg_shadow_very_short[i - 1]
      and abs(c0 - o0) < avg_body_short[i]
      and (c0 - l0) < avg_shadow_very_short[i]
      and (h0 - o0) < avg_shadow_very_short[i]
      and l0 > l1
      and h0 < h1
    ):
      out[i] = 100

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
    o0, c0 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]

    # Upside Gap Three Methods (Bullish)
    # TA-Lib: 1st/2nd same color, 3rd opposite. 3rd opens in 2nd, closes in 1st.
    if (
      c0 > o0
      and c1 > o1
      and c2 < o2
      and min(o1, c1) > max(o0, c0)
      and o2 < max(o1, c1)
      and o2 > min(o1, c1)
      and c2 < max(o0, c0)
      and c2 > min(o0, c0)
    ):
      out[i] = 100
      continue

    # Downside Gap Three Methods (Bearish)
    if (
      c0 < o0
      and c1 < o1
      and c2 > o2
      and max(o1, c1) < min(o0, c0)
      and o2 < max(o1, c1)
      and o2 > min(o1, c1)
      and c2 < max(o0, c0)
      and c2 > min(o0, c0)
    ):
      out[i] = -100

  return out
