"""Numba-optimized Candle Pattern Recognition.

This module provides JIT-compiled kernels for detecting candlestick patterns.
Optimized with Branchless Logic where possible to maximize CPU pipeline throughput.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
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
  """Detect Doji pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:  # Need 10 for SMA + 1 for current
    return out

  # Doji pattern: Body <= Average Range * 0.1
  # Initial sum of total range of previous 10 candles (0..9)
  prev_range_sum = 0.0
  for i in range(10):
    prev_range_sum += high[i] - low[i]

  for i in range(10, n):
    body = abs(close[i] - open_[i])
    # Comparison using avg of previous 10
    if body < (prev_range_sum / 10.0) * 0.1:
      out[i] = 100

    # Update sum for next bar: Add current range (i), subtract oldest range (i-10)
    prev_range_sum += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hammer_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hammer pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Initialize rolling sums for i=10
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for j in range(10):
    body_sum_10 += abs(close[j] - open_[j])
    range_sum_10 += high[j] - low[j]

  # range_sum_5_prev covers [i-6, i-1], i.e., [4, 9) for i=10
  range_sum_5_prev = 0.0
  for j in range(4, 9):
    range_sum_5_prev += high[j] - low[j]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    if (
      real_body < (body_sum_10 / 10.0)
      and lower_shadow > real_body  # ShadowLong filter
      and upper_shadow < (range_sum_10 / 10.0) * 0.1
      and body_bot <= low[i - 1] + (range_sum_5_prev / 5.0) * 0.2
    ):
      out[i] = 100

    # Update sums incrementally (O(1) per iteration)
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
    range_sum_5_prev += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

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
) -> NDArray[np.int32]:
  """Detect Harami pattern (Optimized Fused)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # rb_sum tracks sum(rb, i-10..i-1)
  rb_sum = 0.0
  for i in range(1, 11):
    rb_sum += abs(close[i] - open_[i])

  # Initial state for the loop (for i=11, prev is i=10)
  # We need to know if body[10] was Long relative to avg(0..9)
  body_10 = abs(close[10] - open_[10])
  body_0 = abs(close[0] - open_[0])
  sum_0_9 = rb_sum - body_10 + body_0
  is_prev_long = (body_10 * 10.0) >= sum_0_9

  for i in range(11, n):
    rb = abs(close[i] - open_[i])

    # Check 1: Current body is Short (<= avg)
    is_short = (rb * 10.0) <= rb_sum

    # Check 2: Previous body was Long (already computed)
    if is_prev_long and is_short:
      o1, c1 = open_[i - 1], close[i - 1]
      o2, c2 = open_[i], close[i]

      # Harami Logic: body2 inside body1
      p_t = o1 if o1 > c1 else c1
      p_b = c1 if o1 > c1 else o1
      c_t = o2 if o2 > c2 else c2
      c_b = c2 if o2 > c2 else o2

      # Branchless logic for containment
      # is_contained = (c_t <= p_t) & (c_b >= p_b)
      # is_strict = (c_t < p_t) & (c_b > p_b)
      # val = 80 + 20 * is_strict
      # sign = 1 if c1 < o1 else -1    =>  (c1 < o1) * 2 - 1 ? No, -100/100 or -80/80
      # sign = 1 - 2 * (c1 >= o1)

      if c_t <= p_t and c_b >= p_b:
        if c_t < p_t and c_b > p_b:
          out[i] = 100 if c1 < o1 else -100
        else:
          out[i] = 80 if c1 < o1 else -80

    # Prepare for next iteration
    # Current become previous
    is_prev_long = (rb * 10.0) >= rb_sum

    # Update rolling sum
    # new sum will be sum(i-9..i)
    # current sum is sum(i-10..i-1)
    # diff is body[i] - body[i-10]
    body_trailing = abs(close[i - 10] - open_[i - 10])
    rb_sum += rb - body_trailing

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_shooting_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Shooting Star pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0

  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    po, pc = open_[i - 1], close[i - 1]

    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # TA-Lib logic: Gap up from prior real body
    prior_rb_top = po if po > pc else pc

    if (
      real_body < (body_sum_10 / 10.0)
      and upper_shadow > real_body
      and lower_shadow < (range_sum_10 / 10.0) * 0.1
      and body_bot > prior_rb_top
    ):
      out[i] = -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_inverted_hammer_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Inverted Hammer pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0

  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    po, pc = open_[i - 1], close[i - 1]

    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # TA-Lib logic: Gap down from prior real body
    prior_rb_bot = po if po < pc else pc

    if (
      real_body < (body_sum_10 / 10.0)
      and upper_shadow > real_body
      and lower_shadow < (range_sum_10 / 10.0) * 0.1
      and body_top < prior_rb_bot
    ):
      out[i] = 100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hanging_man_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hanging Man pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10, range_5_prev
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for j in range(10):
    body_sum_10 += abs(close[j] - open_[j])
    range_sum_10 += high[j] - low[j]

  range_sum_5_prev = 0.0
  for j in range(4, 9):
    range_sum_5_prev += high[j] - low[j]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body_top = o if o > c else c
    body_bot = o if o < c else c
    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # TA-Lib logic: body above or near the highs of the previous candle
    if (
      real_body < (body_sum_10 / 10.0)
      and lower_shadow > real_body
      and upper_shadow < (range_sum_10 / 10.0) * 0.1
      and body_bot >= high[i - 1] - (range_sum_5_prev / 5.0) * 0.2
    ):
      out[i] = -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
    range_sum_5_prev += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_marubozu_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Marubozu (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body = abs(c - o)
    body_top = o if o > c else c
    body_bot = o if o < c else c
    upper_shadow = h - body_top
    lower_shadow = body_bot - l

    # 1. Body must be Long
    # 2. Shadows must be Very Short (0.1 * avg_range_10)
    if (
      body > (body_sum_10 / 10.0)
      and upper_shadow < (range_sum_10 / 10.0) * 0.1
      and lower_shadow < (range_sum_10 / 10.0) * 0.1
    ):
      out[i] = 100 if c > o else -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_morning_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Morning Star (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  # We need avg[i-2], avg[i-1], avg[i].
  # First i such that avg[i-2] is valid is i=12 (sum 0..9).
  # We'll maintain body_sum_10 ending at i-1.
  body_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    if i >= 12:
      # avg[i] = mean(i-10..i-1)
      avg_i = body_sum_10 / 10.0

      # avg[i-1] = mean(i-11..i-2)
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      avg_i1 = (body_sum_10 - body_i_1 + body_i_11) / 10.0

      # avg[i-2] = mean(i-12..i-3)
      body_i_2 = abs(close[i - 2] - open_[i - 2])
      body_i_12 = abs(close[i - 12] - open_[i - 12])
      avg_i2 = (body_sum_10 - body_i_1 - body_i_2 + body_i_11 + body_i_12) / 10.0

      o2, c2 = open_[i - 2], close[i - 2]
      o1, c1 = open_[i - 1], close[i - 1]
      o0, c0 = open_[i], close[i]

      body2 = abs(c2 - o2)
      body1 = abs(c1 - o1)
      body0 = abs(c0 - o0)

      if (
        body2 > avg_i2
        and c2 < o2
        and body1 <= avg_i1
        and max(o1, c1) < min(o2, c2)
        and body0 > avg_i
        and c0 > o0
        and c0 > c2 + body2 * penetration
      ):
        out[i] = 100

    # Update sum to cover i-9..i for next iteration (which will call it i-1)
    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_evening_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Evening Star (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  body_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    if i >= 12:
      avg_i = body_sum_10 / 10.0
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      avg_i1 = (body_sum_10 - body_i_1 + body_i_11) / 10.0
      body_i_2 = abs(close[i - 2] - open_[i - 2])
      body_i_12 = abs(close[i - 12] - open_[i - 12])
      avg_i2 = (body_sum_10 - body_i_1 - body_i_2 + body_i_11 + body_i_12) / 10.0

      o2, c2 = open_[i - 2], close[i - 2]
      o1, c1 = open_[i - 1], close[i - 1]
      o0, c0 = open_[i], close[i]

      body2 = abs(c2 - o2)
      body1 = abs(c1 - o1)
      body0 = abs(c0 - o0)

      if (
        body2 > avg_i2
        and c2 > o2
        and body1 <= avg_i1
        and min(o1, c1) > max(o2, c2)
        and body0 > avg_i
        and c0 < o0
        and c0 < c2 - body2 * penetration
      ):
        out[i] = -100

    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_3black_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Black Crows (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 14:
    return out

  rng_sum_10 = 0.0
  # Initial sums for i=13. Need avg ending at 12, 11, 10.
  # avg_10[12] = sum(3..12).
  for j in range(3, 13):
    rng_sum_10 += high[j] - low[j]

  for i in range(13, n):
    o0, c0, h0 = open_[i - 3], close[i - 3], high[i - 3]
    o1, c1, l1 = open_[i - 2], close[i - 2], low[i - 2]
    o2, c2, l2 = open_[i - 1], close[i - 1], low[i - 1]
    o3, c3, l3 = open_[i], close[i], low[i]

    # Shadows
    rr1 = high[i - 1] - low[i - 1]
    rr2 = high[i - 2] - low[i - 2]
    rr11 = high[i - 11] - low[i - 11]
    rr12 = high[i - 12] - low[i - 12]

    # 10-period SMA ending at i, i-1, i-2
    # TA-Lib: TA_CANDLEAVERAGE( ShadowVeryShort, ShadowVeryShortPeriodTotal[2], i-2 )
    # ShadowVeryShortPeriodTotal[2] is sum ending at i-3.
    # rng_sum_10 is sum(i-10..i-1).
    sum_r10_i2 = rng_sum_10 - rr1 - rr2 + rr11 + rr12
    sum_r10_i1 = rng_sum_10 - rr1 + rr11
    sum_r10_i0 = rng_sum_10

    if c0 > o0:  # Prior White
      if c1 < o1 and c2 < o2 and c3 < o3:  # 3 Blacks
        if (
          c1 - l1 < sum_r10_i2 / 10.0 * 0.1  # Very Short Lower Shadows
          and c2 - l2 < sum_r10_i1 / 10.0 * 0.1
          and c3 - l3 < sum_r10_i0 / 10.0 * 0.1
        ):
          if (
            o2 < o1
            and o2 > c1  # Open in Body
            and o3 < o2
            and o3 > c2
          ):
            if c1 > c2 and c2 > c3:  # Declining Closes
              if h0 > c1:  # Closes under prior high
                out[i] = -100

    # Update sum
    rng_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_3white_soldiers_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three White Soldiers (Fused Multi-SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  body_sum_10 = 0.0
  rng_sum_10 = 0.0
  rng_sum_5 = 0.0

  for j in range(2, 12):
    rb = abs(close[j] - open_[j])
    rr = high[j] - low[j]
    body_sum_10 += rb
    rng_sum_10 += rr
    if j >= 7:
      rng_sum_5 += rr

  for i in range(12, n):
    o1, c1, h1 = open_[i - 2], close[i - 2], high[i - 2]
    o2, c2, h2 = open_[i - 1], close[i - 1], high[i - 1]
    o3, c3, h3 = open_[i], close[i], high[i]
    rb1, rb2, rb3 = c1 - o1, c2 - o2, c3 - o3

    rr1 = high[i - 1] - low[i - 1]
    rr2 = high[i - 2] - low[i - 2]
    rr6 = high[i - 6] - low[i - 6]
    rr7 = high[i - 7] - low[i - 7]
    rr11 = high[i - 11] - low[i - 11]

    # Shadows (shifted for each candle)
    sum_r10_i2 = rng_sum_10 - rr1 - rr2 + rr11 + (high[i - 12] - low[i - 12])
    sum_r10_i1 = rng_sum_10 - rr1 + rr11
    sum_r10_i0 = rng_sum_10

    # Near/Far
    sum_r5_i2 = rng_sum_5 - rr1 - rr2 + rr6 + rr7
    sum_r5_i1 = rng_sum_5 - rr1 + rr6

    # Body sums shifted for each candle position
    rb_i10 = abs(close[i - 10] - open_[i - 10])
    sum_b10_i0 = body_sum_10

    if c1 > o1 and c2 > o2 and c3 > o3:  # 3 Whites
      if (
        h1 - c1 < sum_r10_i2 / 10.0 * 0.1
        and h2 - c2 < sum_r10_i1 / 10.0 * 0.1
        and h3 - c3 < sum_r10_i0 / 10.0 * 0.1
      ):
        if c3 > c2 and c2 > c1:  # Higher Closes
          if o2 > o1 and o2 <= c1 + sum_r5_i2 / 5.0 * 0.2:
            if o3 > o2 and o3 <= c2 + sum_r5_i1 / 5.0 * 0.2:
              if (
                rb2 > rb1 - sum_r5_i2 / 5.0 * 0.6 and rb3 > rb2 - sum_r5_i1 / 5.0 * 0.6
              ):
                # 3rd body must be long (not short)
                if rb3 > sum_b10_i0 / 10.0:
                  out[i] = 100

    body_sum_10 += abs(rb3) - rb_i10
    rng_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
    rng_sum_5 += (high[i] - low[i]) - (high[i - 5] - low[i - 5])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_inside_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Inside Up/Down (Fused History)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  # Rolling sum for real body (Period 10)
  sb = 0.0
  for k in range(10):
    sb += abs(close[k] - open_[k])

  # Buffer for sums at i, i-1, i-2 (Circular)
  sb_hist = np.zeros(3)
  sb_hist[10 % 3] = sb

  for i in range(11, n):
    # Update sum for index i (sum over [i-10, i-1])
    sb += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
    sb_hist[i % 3] = sb

    if i < 12:
      continue

    # 3Inside Logic
    # avg_body_long[i-2] = sb_hist[(i-2)%3] / 10.0
    # avg_body_short[i-1] = sb_hist[(i-1)%3] / 10.0

    o1, c1 = open_[i - 2], close[i - 2]
    o2, c2 = open_[i - 1], close[i - 1]
    c3 = close[i]

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    top1, bot1 = (o1, c1) if o1 > c1 else (c1, o1)
    top2, bot2 = (o2, c2) if o2 > c2 else (c2, o2)

    # 1st candle: long (avg_body_long[i-2])
    # 2nd candle: short (avg_body_short[i-1]) and engulfed by 1st
    if (
      body1 > (sb_hist[(i - 2) % 3] * 0.1)
      and body2 <= (sb_hist[(i - 1) % 3] * 0.1)
      and top2 < top1
      and bot2 > bot1
    ):
      # Bullish case: 1st Black, 3rd White, 3rd Close > 1st Open
      if c1 < o1 and c3 > open_[i] and c3 > o1:
        out[i] = 100
      # Bearish case: 1st White, 3rd Black, 3rd Close < 1st Open
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
) -> NDArray[np.int32]:
  """Detect Three Line Strike (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 9:
    return out

  # avg_near[i] = SMA(rng, 5, shifted, scale=0.2)
  #   = sum(rng, i-5..i-1) / 5.0 * 0.2
  # We maintain s_rng = sum(rng, k-5..k-1)
  s_rng = 0.0
  for k in range(5):
    s_rng += high[k] - low[k]

  for i in range(5, n):
    if i >= 4:
      avg_near_i = s_rng * 0.04  # / 5.0 * 0.2

      c1, o1 = close[i - 3], open_[i - 3]
      c2, o2 = close[i - 2], open_[i - 2]
      c3, o3 = close[i - 1], open_[i - 1]
      c4, o4 = close[i], open_[i]

      # Check colors: 3 same, 4th opposite
      is_white1 = c1 > o1
      is_white2 = c2 > o2
      is_white3 = c3 > o3
      is_white4 = c4 > o4

      if (
        (is_white1 == is_white2)
        and (is_white2 == is_white3)
        and (is_white3 != is_white4)
      ):
        # Check 2nd opens near 1st body
        bot1 = min(o1, c1)
        top1 = max(o1, c1)
        if (o2 >= bot1 - avg_near_i) and (o2 <= top1 + avg_near_i):
          # Check 3rd opens near 2nd body
          bot2 = min(o2, c2)
          top2 = max(o2, c2)
          if (o3 >= bot2 - avg_near_i) and (o3 <= top2 + avg_near_i):
            if is_white1:
              # Bullish: consecutive higher closes, gap up, engulfs
              if (c2 > c1) and (c3 > c2) and (o4 > c3) and (c4 < o1):
                out[i] = -100
            else:
              # Bearish: consecutive lower closes, gap down, engulfs
              if (c2 < c1) and (c3 < c2) and (o4 < c3) and (c4 > o1):
                out[i] = 100

    # Advance sum
    s_rng += (high[i] - low[i]) - (high[i - 5] - low[i - 5])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_piercing_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Piercing Pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  body_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    if i >= 11:
      # avg_long[i] = mean(i-10..i-1)
      avg_long_i = body_sum_10 / 10.0

      # avg_long[i-1] = mean(i-11..i-2)
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      avg_long_i1 = (body_sum_10 - body_i_1 + body_i_11) / 10.0

      o1, c1, l1 = open_[i - 1], close[i - 1], low[i - 1]
      o0, c0 = open_[i], close[i]

      if (
        c1 < o1
        and body_i_1 > avg_long_i1
        and c0 > o0
        and body_i > avg_long_i
        and o0 < l1
        and c0 < o1
        and c0 > c1 + (o1 - c1) * 0.5
      ):
        out[i] = 100

    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_dark_cloud_cover_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  penetration: float,
) -> NDArray[np.int32]:
  """Detect Dark Cloud Cover (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  body_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    if i >= 11:
      # avg_long[i] = mean(i-10..i-1)

      # avg_long[i-1] = mean(i-11..i-2)
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      avg_long_i1 = (body_sum_10 - body_i_1 + body_i_11) / 10.0

      o1, h1, c1 = open_[i - 1], high[i - 1], close[i - 1]
      o0, c0 = open_[i], close[i]

      if (
        c1 > o1
        and body_i_1 > avg_long_i1
        and c0 < o0
        and o0 > h1
        and c0 < c1 - body_i_1 * penetration
        and c0 > o1
      ):
        out[i] = -100

    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_kicking_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  by_length: bool = False,
) -> NDArray[np.int32]:
  """Detect Kicking Pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Rolling sums for avg_body_long (period 10) and avg_shadow_very_short
  # avg_body_long[i] = sum(rb, i-10..i-1) / 10.0
  # avg_shadow_very_short[i] = sum(rng, i-10..i-1) / 10.0 * 0.1
  s_rb = 0.0
  s_rng = 0.0
  for k in range(10):
    s_rb += abs(close[k] - open_[k])
    s_rng += high[k] - low[k]

  for i in range(10, n):
    if i < 11:
      # Need avg at i-1 which requires i-1 >= 10
      s_rb += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
      s_rng += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
      continue

    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    body1 = abs(c1 - o1)
    body0 = abs(c0 - o0)

    # avg_body_long[i] = s_rb / 10.0 (current sum covers i-10..i-1)
    avg_bl_i = s_rb / 10.0
    avg_svs_i = s_rng * 0.01  # / 10.0 * 0.1

    # avg_body_long[i-1] = sum(i-11..i-2) / 10.0
    rb_i_1 = abs(close[i - 1] - open_[i - 1])
    rb_i_11 = abs(close[i - 11] - open_[i - 11])
    avg_bl_i1 = (s_rb - rb_i_1 + rb_i_11) / 10.0

    rng_i_1 = high[i - 1] - low[i - 1]
    rng_i_11 = high[i - 11] - low[i - 11]
    avg_svs_i1 = (s_rng - rng_i_1 + rng_i_11) * 0.01

    # 1st marubozu
    is_maru1 = (
      body1 > avg_bl_i1
      and (high[i - 1] - max(o1, c1)) < avg_svs_i1
      and (min(o1, c1) - low[i - 1]) < avg_svs_i1
    )

    # 2nd marubozu
    is_maru2 = (
      body0 > avg_bl_i
      and (high[i] - max(o0, c0)) < avg_svs_i
      and (min(o0, c0) - low[i]) < avg_svs_i
    )

    if is_maru1 and is_maru2 and (c1 > o1) != (c0 > o0):
      # Gap check per TA-Lib:
      if (c1 < o1 and low[i] > high[i - 1]) or (c1 > o1 and high[i] < low[i - 1]):
        if by_length:
          if body0 >= body1:
            out[i] = 100 if c0 > o0 else -100
          else:
            out[i] = 100 if c1 > o1 else -100
        else:
          out[i] = 100 if c0 > o0 else -100

    # Advance sums
    s_rb += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    s_rng += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_matching_low_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Matching Low (TA-Lib compliant)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 7:
    return out

  # Equal: Period=5, Factor=0.05, Type=HighLow
  # At i=6: CandleAverage at i-1=5 uses sum(rng[0..4])
  # Advance: total += rng[i-1] - rng[i-6]

  s_eq = 0.0
  for k in range(5):
    s_eq += high[k] - low[k]

  for i in range(6, n):
    c1, o1 = close[i - 1], open_[i - 1]
    c0, o0 = close[i], open_[i]

    # Both Black
    if c0 < o0 and c1 < o1:
      # Same close (within Equal threshold)
      eq_avg = s_eq * 0.01  # s_eq / 5.0 * 0.05
      if c0 <= c1 + eq_avg and c0 >= c1 - eq_avg:
        out[i] = 100

    # Advance: add rng[i-1], remove rng[i-6]
    s_eq += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_spinning_top_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Spinning Top (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  body_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    # avg_short[i] = mean(i-10..i-1)
    avg_short_i = body_sum_10 / 10.0

    if body_i <= avg_short_i:
      # Upper and Lower shadows must be > body
      up = high[i] - max(open_[i], close[i])
      lo = min(open_[i], close[i]) - low[i]
      if up > body_i and lo > body_i:
        out[i] = 100 if close[i] > open_[i] else -100

    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_rickshaw_man_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Rickshaw Man (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  range_sum_10 = 0.0
  for i in range(10):
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    rng_i = high[i] - low[i]
    avg_doji_i = range_sum_10 / 10.0 * 0.1
    avg_near_i = range_sum_10 / 10.0 * 0.2

    o, c = open_[i], close[i]
    body_i = abs(c - o)
    if body_i <= (avg_doji_i + 1e-12):
      # Shadows must both be > body (ShadowLong with period 0)
      up = high[i] - max(o, c)
      lo = min(o, c) - low[i]
      if up > (body_i + 1e-12) and lo > (body_i + 1e-12):
        # Near center: [min(o,c), max(o,c)] overlaps [mid-near, mid+near]
        mid = (high[i] + low[i]) / 2.0
        if min(o, c) <= (mid + avg_near_i + 1e-12) and max(o, c) >= (
          mid - avg_near_i - 1e-12
        ):
          out[i] = 100

    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def detect_high_wave_parallel(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect High Wave (Parallel, Bit-Perfect)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  num_chunks = 16
  chunk_size = (n - 10 + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = 10 + c * chunk_size
    end = min(start + chunk_size, n)
    if start >= end:
      continue

    s_body = 0.0
    for k in range(start - 10, start):
      s_body += abs(close[k] - open_[k])

    for i in range(start, end):
      o_i, c_i = open_[i], close[i]
      body = abs(c_i - o_i)

      if body < (s_body * 0.1 - 1e-12):
        if body > 0:
          thr = 2.0 * body
          up = high[i] - (o_i if o_i > c_i else c_i)
          if up > (thr - 1e-12):
            lo = (c_i if c_i < o_i else o_i) - low[i]
            if lo > (thr - 1e-12):
              out[i] = 100 if c_i > o_i else -100

      s_body += body - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_high_wave_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect High Wave (Bit-Perfect TA-Lib)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # BodyShort: P10, Factor 1.0, RealBody
  s_body = 0.0
  for k in range(10):
    s_body += abs(close[k] - open_[k])

  for i in range(10, n):
    o_i, c_i = open_[i], close[i]
    body = abs(c_i - o_i)

    # Rule 1: Body < SMA(Body, 10)
    if body < (s_body * 0.1 - 1e-12):
      # Rule 2: Both shadows > 2.0 * Body (ShadowVeryLong)
      # TA-Lib returns 0 for Dojis because CandleColor is 0.
      if body > 0:
        thr = 2.0 * body
        up = high[i] - (o_i if o_i > c_i else c_i)
        if up > (thr - 1e-12):
          lo = (c_i if c_i < o_i else o_i) - low[i]
          if lo > (thr - 1e-12):
            out[i] = 100 if c_i > o_i else -100

    s_body += body - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_long_legged_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Long Legged Doji (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  range_sum_10 = 0.0
  for i in range(10):
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    rng_i = high[i] - low[i]
    avg_doji_i = range_sum_10 / 10.0 * 0.1

    o, c = open_[i], close[i]
    body_i = abs(c - o)
    if body_i <= avg_doji_i:
      # Either shadow must be > body (ShadowLong with period 0)
      up = high[i] - max(o, c)
      lo = min(o, c) - low[i]
      if up > body_i or lo > body_i:
        out[i] = 100

    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_tristar_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Tristar pattern (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  # BodyDoji setting: 0.1 * Average Range(10)
  # Lookback: 12
  # Sum range[0..9] to be used at i=12 (uses range sum of [i-12..i-3])
  range_sum_10 = 0.0
  for j in range(10):
    range_sum_10 += high[j] - low[j]

  for i in range(12, n):
    # TA-Lib uses the average calculated AT i-2 for all 3 candles.
    # At i, avg_doji = average(i-12 .. i-3)
    # Our range_sum_10 at the start of loop i=12 is sum(0..9).
    # i-12 = 0, i-3 = 9. Perfect.
    avg_doji_thresh = (range_sum_10 / 10.0) * 0.1

    b0 = abs(close[i] - open_[i])
    b1 = abs(close[i - 1] - open_[i - 1])
    b2 = abs(close[i - 2] - open_[i - 2])

    if b2 <= avg_doji_thresh and b1 <= avg_doji_thresh and b0 <= avg_doji_thresh:
      # Bullish: 2nd gaps down from 1st, 3rd not lower than 2nd
      # TA_REALBODYGAPDOWN(i-1, i-2): max(o2, c2) < min(o1, c1) [Wait, i-1 is 2nd, i-2 is 1st]
      # TA_REALBODYGAPDOWN(i-1, i-2): max(o1, c1) < min(o2, c2)
      m1_top = max(open_[i - 2], close[i - 2])
      m1_bot = min(open_[i - 2], close[i - 2])
      m2_top = max(open_[i - 1], close[i - 1])
      m2_bot = min(open_[i - 1], close[i - 1])
      m3_top = max(open_[i], close[i])
      m3_bot = min(open_[i], close[i])

      # Bearish: 2nd gaps up (min2 > max1), 3rd not higher than 2nd (max3 < max2)
      if m2_bot > m1_top and m3_top < m2_top:
        out[i] = -100
      # Bullish: 2nd gaps down (max2 < min1), 3rd not lower than 2nd (min3 > min2)
      elif m2_top < m1_bot and m3_bot > m2_bot:
        out[i] = 100

    # Update range_sum_10: it currently is sum(i-12..i-3)
    # For next i+1, we need sum(i+1-12 .. i+1-3) = sum(i-11 .. i-2)
    # So we add rng[i-2] and subtract rng[i-12]
    range_sum_10 += (high[i - 2] - low[i - 2]) - (high[i - 12] - low[i - 12])

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
) -> NDArray[np.int32]:
  """Detect Two Crows (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  # avg_body_long[i-2] = SMA(rb, 10, shifted) at i-2
  # = sum(rb, i-12..i-3) / 10.0
  # We maintain s_rb = sum(rb, k-10..k-1) for current k
  s_rb = 0.0
  for k in range(10):
    s_rb += abs(close[k] - open_[k])

  for i in range(12, n):
    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # avg_body_long[i-2]: sum(i-12..i-3) / 10.0
    # s_rb covers (i-10..i-1). We need (i-12..i-3).
    rb_i_1 = abs(close[i - 1] - open_[i - 1])
    rb_i_2 = abs(close[i - 2] - open_[i - 2])
    rb_i_11 = abs(close[i - 11] - open_[i - 11])
    rb_i_12 = abs(close[i - 12] - open_[i - 12])
    avg_bl_i2 = (s_rb - rb_i_1 - rb_i_2 + rb_i_11 + rb_i_12) / 10.0

    # white, black, black
    if (
      c2 > o2
      and (c2 - o2) > avg_bl_i2
      and c1 < o1
      and min(o1, c1) > max(o2, c2)
      and c0 < o0
      and o0 < o1
      and o0 > c1
      and c0 < c2
      and c0 > o2
    ):
      out[i] = -100

    # Advance sum
    s_rb += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_upside_gap_two_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Upside Gap Two Crows (Fused Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  # Single SMA because both avg_body_long and avg_body_short were P=10
  # At i, we need avg(i-10..i-1) - this is s_body_10
  # And at i-1, we need avg(i-11..i-2) - this is s_body_10_shifted
  # And at i-2, we need avg(i-12..i-3) - this is s_body_10_shifted2

  s_body = 0.0
  for k in range(12):
    s_body += abs(close[k] - open_[k])

  # Current s_body is sum(0..11).
  # sum(0..9) is avg for i=10.
  # sum(1..10) is avg for i=11.
  # sum(2..11) is avg for i=12.

  # Correct initialization for i=12:
  # avg_long[i-2] = avg[10] = mean(0..9)
  # avg_short[i-1] = avg[11] = mean(1..10)

  s_body_i2 = 0.0  # for i-2 (mean(i-12..i-3))
  for k in range(10):
    s_body_i2 += abs(close[k] - open_[k])

  s_body_i1 = (
    s_body_i2 + abs(close[10] - open_[10]) - abs(close[0] - open_[0])
  )  # mean(1..10)

  for i in range(12, n):
    o2, c2 = open_[i - 2], close[i - 2]
    o1, c1 = open_[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    # 1. Colors check: Bull, Bear, Bear (Most restrictive)
    if c2 > o2 and c1 < o1 and c0 < o0:
      # 2. Pattern Logic
      # 1st is Long Body
      if abs(c2 - o2) > (s_body_i2 * 0.1 - 1e-12):
        # 2nd is Short Body
        if abs(c1 - o1) <= (s_body_i1 * 0.1 + 1e-12):
          # Real Body Gap Up
          if min(o1, c1) > c2:
            # 3rd Engulfing 2nd
            if o0 > o1 and c0 < c1 and c0 > c2:
              out[i] = -100

    # Advance sums for next i
    # s_body_i2 (for i-2) -> needs mean( (i+1)-12 .. (i+1)-3 ) = mean(i-11..i-2)
    s_body_i2 += abs(close[i - 2] - open_[i - 2]) - abs(close[i - 12] - open_[i - 12])

    # s_body_i1 (for i-1) -> needs mean( (i+1)-11 .. (i+1)-2 ) = mean(i-10..i-1)
    s_body_i1 += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])

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
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Abandoned Baby (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 28:
    return out

  body_sum_25 = 0.0
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(25):
    body_i = abs(close[i] - open_[i])
    body_sum_25 += body_i
    if i >= 15:
      body_sum_10 += body_i
      range_sum_10 += high[i] - low[i]

  for i in range(25, n):
    body_i = abs(close[i] - open_[i])
    rng_i = high[i] - low[i]

    if i >= 27:
      # Pattern i-2, i-1, i.
      # We have sums ending at i-1.
      # avg_short[i] (P10) = mean(i-10..i-1)
      avg_short_i = body_sum_10 / 10.0

      # avg_doji[i-1] (P10) = mean(i-11..i-2)
      rng_i_1 = high[i - 1] - low[i - 1]
      rng_i_11 = high[i - 11] - low[i - 11]
      avg_doji_i1 = (range_sum_10 - rng_i_1 + rng_i_11) / 10.0 * 0.1

      # avg_long[i-2] (P25)
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_26 = abs(close[i - 26] - open_[i - 26])
      avg_long_i2 = (body_sum_25 - body_i_1 + body_i_26) / 25.0

      body1 = abs(close[i - 2] - open_[i - 2])
      if body1 > avg_long_i2:
        body2 = abs(close[i - 1] - open_[i - 1])
        if body2 <= avg_doji_i1:
          if body_i > avg_short_i:
            c1, o1 = close[i - 2], open_[i - 2]
            h1, l1 = high[i - 2], low[i - 2]
            h2, l2 = high[i - 1], low[i - 1]
            c3, o3 = close[i], open_[i]
            h3, l3 = high[i], low[i]
            if c1 > o1 and (not (c3 > o3)):  # Top
              if (
                l2 > (h1 - 1e-12)
                and h3 < (l2 + 1e-12)
                and c3 < (c1 - body1 * penetration + 1e-12)
              ):
                out[i] = -100
            elif (not (c1 > o1)) and c3 > o3:  # Bottom
              if (
                h2 < (l1 + 1e-12)
                and l3 > (h2 - 1e-12)
                and c3 > (c1 + body1 * penetration - 1e-12)
              ):
                out[i] = 100

    body_sum_25 += body_i - abs(close[i - 25] - open_[i - 25])
    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_advance_block_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Advance Block (Precision Multi-Window Fused)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  # TA-Lib settings for AdvBlock:
  # BodyLong: RealBody, 10, 1.0
  # ShadowShort: Shadows (Total), 10, 1.0 (But comment says "half", so we'll test both. Factor 1.0 is default.)
  # ShadowLong: RealBody, 0, 1.0 (Current RB)
  # Near: HighLow, 5, 0.2
  # Far: HighLow, 5, 0.6

  rb_sum_12 = 0.0
  hl_sum_12 = 0.0
  sh_sum_12 = 0.0

  for j in range(12):
    rb = abs(close[j] - open_[j])
    hl = high[j] - low[j]
    rb_sum_12 += rb
    hl_sum_12 += hl
    sh_sum_12 += hl - rb

  for i in range(12, n):
    o1, h1, c1 = open_[i - 2], high[i - 2], close[i - 2]
    o2, h2, c2 = open_[i - 1], high[i - 1], close[i - 1]
    o3, h3, c3 = open_[i], high[i], close[i]
    rb1, rb2, rb3 = abs(c1 - o1), abs(c2 - o2), abs(c3 - o3)
    us1, us2, us3 = h1 - max(o1, c1), h2 - max(o2, c2), h3 - max(o3, c3)

    if c1 > o1 and c2 > o2 and c3 > o3 and c3 > c2 and c2 > c1:
      # BodyLong(i-2)
      s_bl_i2 = (
        rb_sum_12 - abs(close[i - 1] - open_[i - 1]) - abs(close[i - 2] - open_[i - 2])
      )

      # Near/Far
      s_n_i2 = hl_sum_12 - (high[i - 1] - low[i - 1]) - (high[i - 2] - low[i - 2])
      for k in range(i - 12, i - 7):
        s_n_i2 -= high[k] - low[k]
      s_n_i1 = hl_sum_12 - (high[i - 1] - low[i - 1])
      for k in range(i - 12, i - 6):
        s_n_i1 -= high[k] - low[k]

      # ShadowShort (Total Shadows)
      s_ss_i2 = (
        sh_sum_12
        - ((high[i - 1] - low[i - 1]) - abs(close[i - 1] - open_[i - 1]))
        - ((high[i - 2] - low[i - 2]) - abs(close[i - 2] - open_[i - 2]))
      )
      s_ss_i1 = (
        sh_sum_12
        - ((high[i - 1] - low[i - 1]) - abs(close[i - 1] - open_[i - 1]))
        - ((high[i - 12] - low[i - 12]) - abs(close[i - 12] - open_[i - 12]))
      )
      s_ss_i0 = (
        sh_sum_12
        - ((high[i - 11] - low[i - 11]) - abs(close[i - 11] - open_[i - 11]))
        - ((high[i - 12] - low[i - 12]) - abs(close[i - 12] - open_[i - 12]))
      )

      # Containment
      if (o2 > o1 and o2 <= c1 + (s_n_i2 / 5.0 * 0.2)) and (
        o3 > o2 and o3 <= c2 + (s_n_i1 / 5.0 * 0.2)
      ):
        # 1st candle
        # Testing factor 0.5 based on TA-Lib global.c comment "half the average"
        if rb1 > (s_bl_i2 / 10.0) and us1 < (s_ss_i2 / 10.0 * 0.5):
          # Weakening Logic
          cond1 = (rb2 < rb1 - (s_n_i2 / 5.0 * 0.6)) and (
            rb3 < rb2 + (s_n_i1 / 5.0 * 0.2)
          )
          cond2 = rb3 < rb2 - (s_n_i1 / 5.0 * 0.6)
          cond3 = (
            (rb3 < rb2)
            and (rb2 < rb1)
            and (us3 > (s_ss_i0 / 10.0 * 0.5) or us2 > (s_ss_i1 / 10.0 * 0.5))
          )
          cond4 = (rb3 < rb2) and (us3 > rb3)

          if cond1 or cond2 or cond3 or cond4:
            out[i] = -100

    # update
    rb_i = abs(close[i] - open_[i])
    hl_i = high[i] - low[i]
    rb_sum_12 += rb_i - abs(close[i - 12] - open_[i - 12])
    hl_sum_12 += hl_i - (high[i - 12] - low[i - 12])
    sh_sum_12 += (hl_i - rb_i) - (
      (high[i - 12] - low[i - 12]) - abs(close[i - 12] - open_[i - 12])
    )

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_belt_hold_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Belt Hold (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body = abs(c - o)

    # Must be Long Body
    if body > (body_sum_10 / 10.0):
      if c > o:  # Bull (White Opening Marubozu)
        # No lower shadow (Very Short factor 0.1)
        if (o - l) < (range_sum_10 / 10.0) * 0.1:
          out[i] = 100
      else:  # Bear (Black Opening Marubozu)
        # No upper shadow
        if (h - o) < (range_sum_10 / 10.0) * 0.1:
          out[i] = -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_breakaway_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Breakaway (Fused-Rolling)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 15:
    return out

  # Rolling body sum for avg body at i-4 (period 10)
  # Uses candles [i-14..i-5]
  s_body = 0.0
  for k in range(10):
    s_body += abs(close[k] - open_[k])

  for i in range(14, n):
    c1, o1 = close[i - 4], open_[i - 4]
    c5, o5 = close[i], open_[i]
    is_white1 = c1 > o1
    is_white5 = c5 > o5

    if is_white1 != is_white5:
      # avg_body(10) at i-4 uses candles (i-4)-10.. (i-4)-1 = i-14..i-5
      # Currently, s_body is sum(i-14..i-5).
      if abs(c1 - o1) > (s_body * 0.1):
        c2, o2 = close[i - 3], open_[i - 3]
        c4, o4 = close[i - 1], open_[i - 1]
        is_white2 = c2 > o2
        is_white4 = c4 > o4
        if is_white1 == is_white2 and is_white2 == is_white4:
          if is_white1:
            if min(o2, c2) > max(o1, c1):
              h2, l2 = high[i - 3], low[i - 3]
              h3, l3 = high[i - 2], low[i - 2]
              if h3 > h2 and l3 > l2:
                h4 = high[i - 1]
                if h4 > h3 and low[i - 1] > l3:
                  if c5 < o2 and c5 > c1:
                    out[i] = -100
          else:
            if max(o2, c2) < min(o1, c1):
              h2, l2 = high[i - 3], low[i - 3]
              h3, l3 = high[i - 2], low[i - 2]
              if h3 < h2 and l3 < l2:
                h4 = high[i - 1]
                if h4 < h3 and low[i - 1] < l3:
                  if c5 > o2 and c5 < c1:
                    out[i] = 100

    # Advance s_body: sum(i-14..i-5) -> sum(i-13..i-4)
    # At start of i=14: sum is 0..9.
    # At end of i=14: s_body += body[14-4] - body[14-14] = body[10] - body[0] -> sum(1..10)
    # Next i=15: sum(1..10) is avg for candle 11 (i-4=11). OK!
    s_body += abs(close[i - 4] - open_[i - 4]) - abs(close[i - 14] - open_[i - 14])

  return out

  # Prefix sum of bodies for O(1) average
  p_body = np.zeros(n + 1)
  for k in range(n):
    p_body[k + 1] = p_body[k] + abs(close[k] - open_[k])

  for i in range(4, n):
    c1, o1 = close[i - 4], open_[i - 4]
    c5, o5 = close[i], open_[i]
    is_white1 = c1 > o1
    is_white5 = c5 > o5

    # 1. Colors check: 1st and 5th must be opposite
    if is_white1 != is_white5:
      # 2. 1st must be Long
      # BodyLong(i-4) needs avg(10) at i-4 -> sum(i-14..i-5)
      if i >= 14:
        avg_i4 = (p_body[i - 4] - p_body[i - 14]) * 0.1
        if abs(c1 - o1) > avg_i4:
          # 3. Intermediate Colors Trend (1, 2, 4 same color)
          c2, o2 = close[i - 3], open_[i - 3]
          c4, o4 = close[i - 1], open_[i - 1]
          is_white2 = c2 > o2
          is_white4 = c4 > o4
          if is_white1 == is_white2 and is_white2 == is_white4:
            # Candle 3 can be any color (not checked in original??)
            # Wait, candles 1, 2, 3, 4 are supposedly in trend.
            # TA-Lib rule: candles 1, 2, 3, 4 same color?
            # Original Indie kernel: (is_white1 != is_white2) or (is_white2 != is_white4) skips.
            # So 1, 2, 4 must be same.

            if is_white1:
              # Bearish Breakaway: 1, 2, 3, 4 White? No, TA-Lib only checks 1, 2, 4?
              # Actually, TA-Lib checks 1, 2, 3, 4 same color usually.
              # But original kernel didn't check 3.
              # Let's check original logic: gap_up = min(o2,c2) > max(o1,c1)
              if min(o2, c2) > max(o1, c1):
                h2, l2 = high[i - 3], low[i - 3]
                h3, l3 = high[i - 2], low[i - 2]
                if h3 > h2 and l3 > l2:
                  h4, l4 = high[i - 1], low[i - 1]
                  if h4 > h3 and l4 > l3:
                    if c5 < o2 and c5 > c1:
                      out[i] = -100
            else:
              # Bullish Breakaway: 1, 2, 4 Black
              if max(o2, c2) < min(o1, c1):
                h2, l2 = high[i - 3], low[i - 3]
                h3, l3 = high[i - 2], low[i - 2]
                if h3 < h2 and l3 < l2:
                  h4, l4 = high[i - 1], low[i - 1]
                  if h4 < h3 and l4 < l3:
                    if c5 > o2 and c5 < c1:
                      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_closing_marubozu_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Closing Marubozu (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body = abs(c - o)

    # 1. Long Body
    if body > (body_sum_10 / 10.0):
      # 2. No shadow at closing end (Very Short factor 0.1)
      if c > o:  # Bull
        if (h - c) < (range_sum_10 / 10.0) * 0.1:
          out[i] = 100
      else:  # Bear
        if (c - l) < (range_sum_10 / 10.0) * 0.1:
          out[i] = -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_dragonfly_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Dragonfly Doji (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  range_sum_10 = 0.0
  for i in range(10):
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    rng_i = high[i] - low[i]
    avg_ref_i = range_sum_10 / 10.0
    if avg_ref_i < 1e-12:
      avg_ref_i = rng_i

    avg_doji_i = avg_ref_i * 0.1
    avg_shadow_very_short_i = avg_ref_i * 0.1

    body_i = abs(close[i] - open_[i])
    if body_i <= avg_doji_i:
      # Upper shadow must be very small
      up = high[i] - max(open_[i], close[i])
      if up <= avg_shadow_very_short_i:
        # Lower shadow must exist and be long
        lo = min(open_[i], close[i]) - low[i]
        if lo > avg_doji_i:
          out[i] = 100

    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_gravestone_doji_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Gravestone Doji (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  range_sum_10 = 0.0
  for i in range(10):
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    rng_i = high[i] - low[i]
    avg_ref_i = range_sum_10 / 10.0
    if avg_ref_i < 1e-12:
      avg_ref_i = rng_i

    avg_doji_i = avg_ref_i * 0.1
    avg_shadow_very_short_i = avg_ref_i * 0.1

    body_i = abs(close[i] - open_[i])
    if body_i <= avg_doji_i:
      # Lower shadow very short
      lo = min(open_[i], close[i]) - low[i]
      if lo <= avg_shadow_very_short_i:
        # Upper shadow must exist
        up = high[i] - max(open_[i], close[i])
        if up > avg_doji_i:
          out[i] = 100

    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

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
) -> NDArray[np.int32]:
  """Detect Homing Pigeon (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  body_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    if i >= 11:
      # Pattern i-1, i. Both black.
      # avg_long[i-1] = mean(i-11..i-2)
      body_i_1 = open_[i - 1] - close[i - 1]  # Must be black
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      avg_long_i1 = (body_sum_10 - abs(body_i_1) + body_i_11) / 10.0
      # avg_short[i] = mean(i-10..i-1)
      avg_short_i = body_sum_10 / 10.0

      if avg_long_i1 < 1e-12:
        avg_long_i1 = body_i_1
        avg_short_i = body_i_1

      if body_i_1 > avg_long_i1:
        o2, c2 = open_[i], close[i]
        b2 = o2 - c2
        if b2 > 0 and b2 < avg_short_i:
          o1, c1 = open_[i - 1], close[i - 1]
          if o2 < o1 and c2 > c1:
            out[i] = 100

    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_identical_three_crows_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Identical Three Crows (Fused Multi-SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  rng_sum_10 = 0.0
  rng_sum_5 = 0.0

  # i=12. Need sums ending at 11, 10, 9.
  for j in range(2, 12):
    rr = high[j] - low[j]
    rng_sum_10 += rr
    if j >= 7:
      rng_sum_5 += rr

  for i in range(12, n):
    o1, c1, l1 = open_[i - 2], close[i - 2], low[i - 2]
    o2, c2, l2 = open_[i - 1], close[i - 1], low[i - 1]
    o3, c3, l3 = open_[i], close[i], low[i]

    # Range Trailing
    r_idx1 = high[i - 1] - low[i - 1]
    r_idx2 = high[i - 2] - low[i - 2]
    r_idx6 = high[i - 6] - low[i - 6]
    r_idx7 = high[i - 7] - low[i - 7]
    r_idx11 = high[i - 11] - low[i - 11]

    # ShadowVeryShort: SMA(10, Rng) ending at i-3 (i2), i-2 (i1), i-1 (i0)
    sum_r10_i2 = rng_sum_10 - r_idx1 - r_idx2 + r_idx11 + (high[i - 12] - low[i - 12])
    sum_r10_i1 = rng_sum_10 - r_idx1 + r_idx11
    sum_r10_i0 = rng_sum_10

    # Equal: SMA(5, Rng) ending at i-3 (i2), i-2 (i1)
    sum_r5_i2 = rng_sum_5 - r_idx1 - r_idx2 + r_idx6 + r_idx7
    sum_r5_i1 = rng_sum_5 - r_idx1 + r_idx6

    if c1 < o1 and c2 < o2 and c3 < o3:  # 3 Blacks
      if c1 > c2 and c2 > c3:  # Declining Closes
        # 1st Open Equal to 2nd Open?
        # NO: TA-Lib says 2nd Open Equal to 1st Close.
        e2 = sum_r5_i2 / 5.0 * 0.05
        if o2 <= c1 + e2 and o2 >= c1 - e2:
          # 2nd Open Equal to 3rd Open?
          # NO: TA-Lib says 3rd Open Equal to 2nd Close.
          e1 = sum_r5_i1 / 5.0 * 0.05
          if o3 <= c2 + e1 and o3 >= c2 - e1:
            # Shadows
            if (
              c1 - l1 < sum_r10_i2 / 10.0 * 0.1
              and c2 - l2 < sum_r10_i1 / 10.0 * 0.1
              and c3 - l3 < sum_r10_i0 / 10.0 * 0.1
            ):
              out[i] = -100

    # Update sums
    rng_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
    rng_sum_5 += (high[i] - low[i]) - (high[i - 5] - low[i - 5])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_in_neck_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect In-Neck (Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Shifted sums:
  # at i=11:
  # body_sum_10 (avg for i-1=10) needs sum(0..9)
  # range_sum_5 (avg eq for i-1=10) needs sum(5..9)

  s_body = 0.0
  s_range = 0.0
  for k in range(10):
    s_body += abs(close[k] - open_[k])
    if k >= 5:
      s_range += high[k] - low[k]

  for i in range(11, n):
    c1, o1 = close[i - 1], open_[i - 1]
    c0, o0 = close[i], open_[i]

    # 1. Colors check: Bear followed by Bull (Restrictive)
    if c1 < o1 and c0 > o0:
      # 2. Pattern Logic
      # 1st is Long Body
      if (o1 - c1) > (s_body * 0.1 - 1e-12):
        # 3. Same Level
        # c0 matches c1 (within avg_range_5[i-1] * 0.05)
        limit = s_range * 0.01
        l1 = low[i - 1]
        if o0 < l1 and c0 <= (c1 + limit + 1e-12) and c0 >= c1:
          out[i] = -100

    # Advance sums for next i
    s_body += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
    s_range += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_ladder_bottom_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Ladder Bottom (TA-Lib compliant)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 15:
    return out

  # ShadowVeryShort: Period=10, Factor=0.1, Type=HighLow
  # At i=14: CandleAverage at i-1=13 uses sum(rng[3..12]) → 10 values
  # Advance: total += rng[i-2] - rng[i-12]

  # ShadowVeryShort: Period=10, Factor=0.1, Type=HighLow
  # At i=14: CandleAverage at i-1=13 uses sum(rng[3..12]) → 10 values
  # Advance: total += rng[i-2] - rng[i-12]

  s_svs = 0.0
  for k in range(3, 13):
    s_svs += high[k] - low[k]

  for i in range(14, n):
    c5, o5 = close[i], open_[i]

    # 5th: White
    if c5 > o5:
      c4, o4, h4 = close[i - 1], open_[i - 1], high[i - 1]

      # 4th: Black with upper shadow > ShadowVeryShort avg
      if c4 < o4 and (h4 - o4) > (s_svs * 0.01):  # / 10.0 * 0.1
        # 5th opens above 4th open, closes above 4th high
        if o5 > o4 and c5 > h4:
          # 1st-3rd: Black with consecutively lower opens and closes
          c1, o1 = close[i - 4], open_[i - 4]
          c2, o2 = close[i - 3], open_[i - 3]
          c3, o3 = close[i - 2], open_[i - 2]
          if (
            c1 < o1
            and c2 < o2
            and c3 < o3
            and o2 < o1
            and o3 < o2
            and c2 < c1
            and c3 < c2
          ):
            out[i] = 100

    # Advance: add rng[i-1], remove rng[i-11]
    s_svs += (high[i - 1] - low[i - 1]) - (high[i - 11] - low[i - 11])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_long_line_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Long Line (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body = abs(c - o)
    body_top = o if o > c else c
    body_bot = o if o < c else c

    # 1. Body must be Long
    # 2. Both shadows must be Short (factor 0.5 * avg_shadow_sum_10)
    avg_shadow_short = (range_sum_10 - body_sum_10) / 10.0 * 0.5
    avg_body = body_sum_10 / 10.0
    avg_range = range_sum_10 / 10.0

    # Fallback for flat background
    if avg_body < 1e-12:
      avg_body = body * 0.8  # Assume 80% of current is "longish"
      avg_range = h - l

    avg_shadow_short = (avg_range - avg_body) * 0.5

    if (
      body >= avg_body
      and (h - body_top) <= avg_shadow_short
      and (body_bot - l) <= avg_shadow_short
    ):
      out[i] = 100 if c > o else -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_mat_hold_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  penetration: float = 0.5,
) -> NDArray[np.int32]:
  """Detect Mat Hold (Fused Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 29:
    return out

  # Two SMAs (Shifted):
  # at i:
  # avg_long[i-4] needs body_sum_25 ending at i-5 (mean(i-29..i-5))
  # avg_short[i-3] needs body_sum_10 ending at i-4 (mean(i-13..i-4))
  # avg_short[i-2] needs body_sum_10 ending at i-3 (mean(i-12..i-3))
  # avg_short[i-1] needs body_sum_10 ending at i-2 (mean(i-11..i-2))

  s_body_25 = 0.0
  for k in range(25):
    s_body_25 += abs(close[k] - open_[k])

  # i=29:
  # avg_long[25] needs mean(0..24). Our s_body_25 is sum(0..24).
  # avg_short[26] (i-3) needs mean(16..25).
  # avg_short[27] (i-2) needs mean(17..26).
  # avg_short[28] (i-1) needs mean(18..27).

  s_body_10_i1 = 0.0  # for i-1 (mean(18..27))
  for k in range(18, 28):
    s_body_10_i1 += abs(close[k] - open_[k])

  s_body_10_i2 = 0.0  # for i-2 (mean(17..26))
  for k in range(17, 27):
    s_body_10_i2 += abs(close[k] - open_[k])

  s_body_10_i3 = 0.0  # for i-3 (mean(16..25))
  for k in range(16, 26):
    s_body_10_i3 += abs(close[k] - open_[k])

  for i in range(29, n):
    o4, c4 = open_[i - 4], close[i - 4]
    o0, c0 = open_[i], close[i]

    # 1. Colors and Confirmation (Restrictive)
    if c4 > o4 and c0 > o0:
      o3, c3 = open_[i - 3], close[i - 3]
      o2, c2 = open_[i - 2], close[i - 2]
      o1, c1 = open_[i - 1], close[i - 1]

      # 2nd Black
      if c3 < o3:
        # Body sizes check
        # 1st > LongAvg (Factor=1.0 per TA-Lib default for BodyLong)
        if (c4 - o4) > (s_body_25 / 25.0):
          # 2nd, 3rd, 4th < ShortAvg (Factor=1.0 per TA-Lib default for BodyShort)
          if (
            abs(c3 - o3) < (s_body_10_i3 / 10.0)
            and abs(c2 - o2) < (s_body_10_i2 / 10.0)
            and abs(c1 - o1) < (s_body_10_i1 / 10.0)
          ):
            # Gap Up 1st to 2nd (2nd body above 1st body)
            # Since 1st white, 2nd black: close2 > close1
            if c3 > c4:
              # Hold within 1st body checks (Penetration)
              # Bottom of 3rd/4th > (Close1 - Body1 * pen)
              limit = c4 - (c4 - o4) * penetration
              if (
                min(o2, c2) < c4
                and min(o2, c2) > (limit - 1e-12)
                and min(o1, c1) < c4
                and min(o1, c1) > (limit - 1e-12)
              ):
                # Falling reaction
                # 3rd top < 2nd top (2nd is black, so open is top)
                if max(o2, c2) < o3:
                  # 4th top < 3rd top
                  if max(o1, c1) < max(o2, c2):
                    # 5th opens above 4th close
                    if o0 > c1:
                      # 5th closes above highest high of reaction
                      reaction_high = max(high[i - 3], high[i - 2], high[i - 1])
                      if c0 > reaction_high:
                        out[i] = 100

    # Advance sums
    s_body_25 += abs(close[i - 4] - open_[i - 4]) - abs(close[i - 29] - open_[i - 29])
    s_body_10_i3 += abs(close[i - 3] - open_[i - 3]) - abs(
      close[i - 13] - open_[i - 13]
    )
    s_body_10_i2 += abs(close[i - 2] - open_[i - 2]) - abs(
      close[i - 12] - open_[i - 12]
    )
    s_body_10_i1 += abs(close[i - 1] - open_[i - 1]) - abs(
      close[i - 11] - open_[i - 11]
    )

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_on_neck_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect On-Neck (Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Shifted sums:
  # at i=11:
  # body_sum_10 (avg for i-1=10) needs sum(0..9)
  # range_sum_5 (avg eq for i-1=10) needs sum(5..9)

  s_body = 0.0
  s_range = 0.0
  for k in range(10):
    s_body += abs(close[k] - open_[k])
    if k >= 5:
      s_range += high[k] - low[k]

  for i in range(11, n):
    c1, o1 = close[i - 1], open_[i - 1]
    c0, o0 = close[i], open_[i]

    # 1. Colors check: Bear followed by Bull (Restrictive)
    if c1 < o1 and c0 > o0:
      # 2. Pattern Logic
      # 1st is Long Body (mean(i-11..i-2))
      # At i=11, needs mean(0..9). Our s_body is sum(0..9).
      # Wait! mean(i-11..i-2)? No, TA-Lib shifted SMA for index j is mean(j-p..j-1)? No.
      # CandleAverage(period, scale, index) uses sum of 'period' bars ending at index-1.
      # For OnNeck at index i, index for CandleAverage is i-1.
      # So it uses sum of period bars ending at (i-1)-1 = i-2.
      # At i=11, it uses sum ending at 9. Period 10 -> 0..9. Correct.

      # avg_body_long[i-1] = s_body / 10
      if (o1 - c1) > (s_body * 0.1 - 1e-12):
        # 3. Same Level
        # c0 matches l1 (within avg_range_5[i-1] * 0.05)
        # avg_range_5[i-1] = s_range / 5
        # limit = (s_range / 5) * 0.05 = s_range * 0.01
        l1 = low[i - 1]
        limit = s_range * 0.01
        if o0 < l1 and c0 <= (l1 + limit + 1e-12) and c0 >= (l1 - limit - 1e-12):
          out[i] = -100

    # Advance sums for next i
    # i=11: we used 0..9. Next i=12 needs 1..10.
    s_body += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
    s_range += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_rise_fall_three_methods_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Rise/Fall Three Methods (Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 24:  # Need enough data for shifted SMAs
    return out

  # Prefix sum of bodies for O(1) average
  p_body = np.zeros(n + 1)
  for k in range(n):
    p_body[k + 1] = p_body[k] + abs(close[k] - open_[k])

  for i in range(14, n):
    o0, c0 = open_[i], close[i]
    is_bull0 = c0 > o0
    is_bear0 = c0 < o0

    # 1. 5th Candle color (Most restrictive)
    if is_bull0:
      # Rising Three Methods
      o4, h4, l4, c4 = open_[i - 4], high[i - 4], low[i - 4], close[i - 4]
      if c4 > o4 and c0 > c4:  # 1st must be Bullish
        # Check Long/Short bodies (Prefix Sum SMA)
        # avg[i] = (p[i] - p[i-10])/10
        if (c0 - o0) > (p_body[i] - p_body[i - 10]) * 0.1:
          if (c4 - o4) > (p_body[i - 4] - p_body[i - 14]) * 0.1:
            # Middle 3 must be Bearish and contained
            c1, o1 = close[i - 1], open_[i - 1]
            c2, o2 = close[i - 2], open_[i - 2]
            c3, o3 = close[i - 3], open_[i - 3]
            if c1 < o1 and c2 < o2 and c3 < o3:
              # Check containment
              if (
                min(o1, c1) < h4
                and max(o1, c1) > l4
                and min(o2, c2) < h4
                and max(o2, c2) > l4
                and min(o3, c3) < h4
                and max(o3, c3) > l4
              ):
                # BodyShort and decreasing closes
                if (
                  (o1 - c1) < (p_body[i - 1] - p_body[i - 11]) * 0.1
                  and (o2 - c2) < (p_body[i - 2] - p_body[i - 12]) * 0.1
                  and (o3 - c3) < (p_body[i - 3] - p_body[i - 13]) * 0.1
                ):
                  if c3 > c2 and c2 > c1 and o0 > c1:
                    out[i] = 100

    elif is_bear0:
      # Falling Three Methods
      o4, h4, l4, c4 = open_[i - 4], high[i - 4], low[i - 4], close[i - 4]
      if c4 < o4 and c0 < c4:  # 1st must be Bearish
        if (o0 - c0) > (p_body[i] - p_body[i - 10]) * 0.1:
          if (o4 - c4) > (p_body[i - 4] - p_body[i - 14]) * 0.1:
            c1, o1 = close[i - 1], open_[i - 1]
            c2, o2 = close[i - 2], open_[i - 2]
            c3, o3 = close[i - 3], open_[i - 3]
            if c1 > o1 and c2 > o2 and c3 > o3:
              if (
                min(o1, c1) < h4
                and max(o1, c1) > l4
                and min(o2, c2) < h4
                and max(o2, c2) > l4
                and min(o3, c3) < h4
                and max(o3, c3) > l4
              ):
                if (
                  (c1 - o1) < (p_body[i - 1] - p_body[i - 11]) * 0.1
                  and (c2 - o2) < (p_body[i - 2] - p_body[i - 12]) * 0.1
                  and (c3 - o3) < (p_body[i - 3] - p_body[i - 13]) * 0.1
                ):
                  if c3 < c2 and c2 < c1 and o0 < c1:
                    out[i] = -100

  return out

  # Rolling body sum for 10-period average
  body_sum_10 = 0.0
  for j in range(10):
    body_sum_10 += abs(close[j] - open_[j])

  for i in range(14, n):
    # Compute shifted body averages
    # avg_body[i] = mean(i-10..i-1) = body_sum_10 / 10
    # We need avg_body at indices: i-4, i-3, i-2, i-1, i
    # avg_body[i-4] = mean(i-14..i-5)
    # avg_body[i-3] = mean(i-13..i-4)
    # avg_body[i-2] = mean(i-12..i-3)
    # avg_body[i-1] = mean(i-11..i-2)
    # avg_body[i] = mean(i-10..i-1) = body_sum_10 / 10

    avg_body_i = body_sum_10 / 10.0

    # Calculate shifted averages
    body_i_1 = abs(close[i - 1] - open_[i - 1])
    body_i_11 = abs(close[i - 11] - open_[i - 11])
    avg_body_i1 = (body_sum_10 - body_i_1 + body_i_11) / 10.0

    body_i_2 = abs(close[i - 2] - open_[i - 2])
    body_i_12 = abs(close[i - 12] - open_[i - 12])
    avg_body_i2 = (body_sum_10 - body_i_1 - body_i_2 + body_i_11 + body_i_12) / 10.0

    body_i_3 = abs(close[i - 3] - open_[i - 3])
    body_i_13 = abs(close[i - 13] - open_[i - 13])
    avg_body_i3 = (
      body_sum_10 - body_i_1 - body_i_2 - body_i_3 + body_i_11 + body_i_12 + body_i_13
    ) / 10.0

    body_i_4 = abs(close[i - 4] - open_[i - 4])
    body_i_14 = abs(close[i - 14] - open_[i - 14])
    avg_body_i4 = (
      body_sum_10
      - body_i_1
      - body_i_2
      - body_i_3
      - body_i_4
      + body_i_11
      + body_i_12
      + body_i_13
      + body_i_14
    ) / 10.0

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
      and body4 > avg_body_i4
      and c3 < o3
      and body3 < avg_body_i3
      and c2 < o2
      and body2 < avg_body_i2
      and c1 < o1
      and body1 < avg_body_i1
      and c0 > o0
      and body0 > avg_body_i
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
      and body4 > avg_body_i4
      and c3 > o3
      and body3 < avg_body_i3
      and c2 > o2
      and body2 < avg_body_i2
      and c1 > o1
      and body1 < avg_body_i1
      and c0 < o0
      and body0 > avg_body_i
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

    # Update sum
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_short_line_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Short Line (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    o, c, h, l = open_[i], close[i], high[i], low[i]
    body = abs(c - o)
    body_top = max(o, c)
    body_bot = min(o, c)
    upper = h - body_top
    lower = body_bot - l

    # 1. Body must be Short
    # 2. Both shadows must be Short (factor 0.5 * avg_shadow_sum_10)
    avg_shadow_short = (range_sum_10 - body_sum_10) / 10.0 * 0.5
    if (
      body < (body_sum_10 / 10.0)
      and upper < avg_shadow_short
      and lower < avg_shadow_short
    ):
      out[i] = 100 if c >= o else -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_stalled_pattern_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Stalled Pattern (Fused Multi-SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  body_sum_10 = 0.0
  rng_sum_10 = 0.0
  rng_sum_5 = 0.0

  # i=12. Need sums ending at 11, 10, 9.
  for j in range(2, 12):
    rb = abs(close[j] - open_[j])
    rr = high[j] - low[j]
    body_sum_10 += rb
    rng_sum_10 += rr
    if j >= 7:
      rng_sum_5 += rr

  for i in range(12, n):
    o1, c1 = open_[i - 2], close[i - 2]
    o2, c2, h2 = open_[i - 1], close[i - 1], high[i - 1]
    o3, c3 = open_[i], close[i]
    rb1, rb2, rb3 = abs(c1 - o1), abs(c2 - o2), abs(c3 - o3)

    # Body Sum Trailing
    rb11 = abs(close[i - 11] - open_[i - 11])
    rb12 = abs(close[i - 12] - open_[i - 12])

    # BodyLong: SMA(10, RB) ending at i-3 (sum_b10_i2), i-2 (sum_b10_i1)
    sum_b10_i2 = (
      body_sum_10
      - abs(close[i - 1] - open_[i - 1])
      - abs(close[i - 2] - open_[i - 2])
      + rb11
      + rb12
    )
    sum_b10_i1 = body_sum_10 - abs(close[i - 1] - open_[i - 1]) + rb11
    # BodyShort: SMA(10, RB) ending at i-1 (sum_b10_i0)
    sum_b10_i0 = body_sum_10

    # ShadowVeryShort: SMA(10, Rng) ending at i-2 (sum_r10_i1)
    sum_r10_i1 = rng_sum_10 - (high[i - 1] - low[i - 1]) + (high[i - 11] - low[i - 11])

    # Near: SMA(5, Rng) ending at i-3 (sum_r5_i2), i-2 (sum_r5_i1)
    sum_r5_i2 = (
      rng_sum_5
      - (high[i - 1] - low[i - 1])
      - (high[i - 2] - low[i - 2])
      + (high[i - 6] - low[i - 6])
      + (high[i - 7] - low[i - 7])
    )
    sum_r5_i1 = rng_sum_5 - (high[i - 1] - low[i - 1]) + (high[i - 6] - low[i - 6])

    if c1 > o1 and c2 > o2 and c3 > o3:  # 3 Whites
      if c3 > c2 and c2 > c1:  # Higher Closes
        if rb1 > sum_b10_i2 / 10.0:  # 1st Long
          if rb2 > sum_b10_i1 / 10.0:  # 2nd Long
            if h2 - c2 < sum_r10_i1 / 10.0 * 0.1:  # 2nd Short Shadow
              if o2 > o1 and o2 <= c1 + sum_r5_i2 / 5.0 * 0.2:  # 2nd Near 1st
                if rb3 < sum_b10_i0 / 10.0:  # 3rd Small
                  # 3rd rides on shoulder of 2nd
                  if o3 >= c2 - rb3 - sum_r5_i1 / 5.0 * 0.2:
                    out[i] = -100

    # Update sums
    body_sum_10 += rb3 - abs(close[i - 10] - open_[i - 10])
    rng_sum_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
    rng_sum_5 += (high[i] - low[i]) - (high[i - 5] - low[i - 5])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_stick_sandwich_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Stick Sandwich (TA-Lib compliant)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 8:
    return out

  # Equal: Period=5, Factor=0.05, Type=HighLow
  # CandleAverage(Equal, total, i-2) = total / 5.0
  # CandleRange(Equal, i-2) = high[i-2] - low[i-2]
  # Rolling sum of HighLow range at i-2, period 5
  # At iteration i, we need sum(rng[i-7..i-3]) for the Equal avg at i-2

  s_eq = 0.0
  for k in range(5):
    s_eq += high[k] - low[k]

  # s_eq = sum(rng[0..4]). At i=7: need sum(rng[0..4]) for avg at i-2=5.
  # CandleAverage at i-2: sum of 5 ranges ending at (i-2)-1 = i-3.
  # At i=7: ranges ending at 4, so sum(0..4). Correct.

  for i in range(7, n):
    c1, o1 = close[i - 2], open_[i - 2]
    c2, o2 = close[i - 1], open_[i - 1]
    c3, o3 = close[i], open_[i]

    # Black, White, Black
    if c1 < o1 and c2 > o2 and c3 < o3:
      # 2nd candle low > 1st candle close
      if low[i - 1] > c1:
        # 3rd close equals 1st close (within Equal threshold)
        eq_avg = s_eq * 0.01  # / 5.0 * 0.05
        if c3 <= c1 + eq_avg and c3 >= c1 - eq_avg:
          out[i] = 100

    # Advance: add range at i-2, remove range at i-7
    s_eq += (high[i - 2] - low[i - 2]) - (high[i - 7] - low[i - 7])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_takuri_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Takuri Doji (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 11:
    return out

  range_sum_10 = 0.0
  for i in range(10):
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    rng_i = high[i] - low[i]
    avg_ref_i = range_sum_10 / 10.0
    if avg_ref_i < 1e-12:
      avg_ref_i = rng_i

    avg_doji_i = avg_ref_i * 0.1
    avg_shadow_very_short_i = avg_ref_i * 0.1

    body_i = abs(close[i] - open_[i])
    if body_i <= avg_doji_i:
      # Upper shadow very short
      up = high[i] - max(open_[i], close[i])
      if up <= avg_shadow_very_short_i:
        # Lower shadow very long
        lo = min(open_[i], close[i]) - low[i]
        # In Takuri, lower shadow is relative to AVG range.
        # Using 1.5 * avg_ref per TA-Lib
        if lo > (avg_ref_i * 1.5):
          out[i] = 100

    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_thrusting_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Thrusting (Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Shifted sums:
  # at i=11:
  # body_sum_10 (avg for i-1=10) needs sum(0..9)
  # range_sum_5 (avg eq for i-1=10) needs sum(5..9)

  s_body = 0.0
  s_range = 0.0
  for k in range(10):
    s_body += abs(close[k] - open_[k])
    if k >= 5:
      s_range += high[k] - low[k]

  for i in range(11, n):
    c1, o1 = close[i - 1], open_[i - 1]
    c0, o0 = close[i], open_[i]

    # 1. Colors check: Bear followed by Bull (Restrictive)
    if c1 < o1 and c0 > o0:
      # 2. Pattern Logic
      # 1st is Long Body
      if (o1 - c1) > (s_body * 0.1 - 1e-12):
        # 3. Same Level
        # c0 matches c1 (within avg_range_5[i-1] * 0.05)
        limit = s_range * 0.01
        l1 = low[i - 1]
        if (
          o0 < l1
          and c0 >= (c1 + limit + 1e-12)
          and c0 <= (c1 + (o1 - c1) * 0.5 + 1e-12)
        ):
          out[i] = -100

    # Advance sums for next i
    s_body += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
    s_range += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_unique_three_river_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Unique 3 River (Fused History)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  # Rolling sum for real body (Period 10)
  sb = 0.0
  for k in range(10):
    sb += abs(close[k] - open_[k])

  sb_hist = np.zeros(3)
  sb_hist[10 % 3] = sb

  for i in range(11, n):
    # Sum for [i-10, i-1]
    sb += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
    sb_hist[i % 3] = sb

    if i < 12:
      continue

    # Unique 3 River Logic
    # 1st: Long Black (avg_body_long[i-2])
    # 2nd: Black Harami with Lower Low
    # 3rd: Short White (avg_body_short[i])

    o2, l2, c2 = open_[i - 2], low[i - 2], close[i - 2]
    o1, l1, c1 = open_[i - 1], low[i - 1], close[i - 1]
    o0, c0 = open_[i], close[i]

    body2 = abs(c2 - o2)
    body0 = abs(c0 - o0)

    # 1st: body2 > avg_body_long[i-2]
    # 3rd: body0 < avg_body_short[i]
    if (
      body2 > (sb_hist[(i - 2) % 3] * 0.1)
      and c2 < o2
      and c1 < o1
      and c1 > c2
      and o1 <= o2
      and l1 < l2
      and body0 < (sb * 0.1)
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
) -> NDArray[np.int32]:
  """Detect Separating Lines (Fused Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Three SMAs:
  # 1. Equal (Range, P=5, shifted at i-1? No, wait)
  # wrapper: avg_equal = _rolling_sma(rng, 5, scale=0.05)
  # kernel: abs(o1-o2) <= avg_equal[i-2]
  # _rolling_sma(rng, 5) at i-2 is mean(rng[i-6..i-2])

  # 2. BodyLong (RealBody, P=10, shifted)
  # wrapper: _rolling_sma_prev(real_body, 10, scale=1.0)
  # kernel: avg_body_long[i] = mean(real_body[i-10..i-1])

  # 3. ShadowVeryShort (Range, P=10, shifted)
  # wrapper: _rolling_sma_prev(rng, 10, scale=0.1)
  # kernel: avg_shadow_very_short[i] = mean(range[i-10..i-1]) * 0.1

  s_range_5 = 0.0  # for i-6..i-2
  s_range_10 = 0.0  # for i-10..i-1
  s_body_10 = 0.0  # for i-10..i-1

  for k in range(10):
    rb = abs(close[k] - open_[k])
    rng_k = high[k] - low[k]
    s_body_10 += rb
    s_range_10 += rng_k
    if k >= 4 and k <= 8:  # for i=10 check, needs mean(rng[4..8])
      s_range_5 += rng_k

  for i in range(10, n):
    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]
    is_bull1 = c1 > o1
    is_bull2 = c2 > o2

    # 1. Opposite Colors
    if is_bull1 != is_bull2:
      # 2. Same Open
      # abs(o1 - o2) <= mean(rng[i-6..i-2]) * 0.05
      # sum(i-6..i-2) is s_range_5 adjusted for i?
      # At i=10, s_range_5 is sum(4..8). i-6=4, i-2=8. Correct.
      if abs(o1 - o2) <= (s_range_5 * 0.01):  # 5.0 * 0.05 / 5 = 0.01
        # 3. 2nd is Long Body
        if abs(c2 - o2) > (s_body_10 * 0.1):
          # 4. Shadow Very Short
          # Bearish: Bull then Bear. (h2-o2) < mean(rng[i-10..i-1]) * 0.1
          # Bullish: Bear then Bull. (o2-l2) < mean(rng[i-10..i-1]) * 0.1
          limit = s_range_10 * 0.01  # 10.0 * 0.1 / 10 = 0.01
          if is_bull2:  # Bullish
            if (o2 - low[i]) < (limit + 1e-12):
              out[i] = 100
          else:  # Bearish
            if (high[i] - o2) < (limit + 1e-12):
              out[i] = -100

    # Advance sums
    # s_range_5 needs sum( (i+1)-6 .. (i+1)-2 ) = sum(i-5 .. i-1)
    # Current i is 10. Needs sum(5..9).
    # sum(4..8) + rng[9] - rng[4] = sum(5..9).
    s_range_5 += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

    # s_range_10 needs sum(i-9 .. i)
    s_range_10 += (high[i] - low[i]) - (high[i - 10] - low[i - 10])
    s_body_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_counterattack_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Counterattack (Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Initialization
  s_body = 0.0
  s_range = 0.0
  for k in range(10):
    s_body += abs(close[k] - open_[k])
    if k >= 5:
      s_range += high[k] - low[k]

  # At i=10:
  # s_body = sum(0..9)  -> avg[10]
  # s_range = sum(5..9) -> avg_eq[10]

  for i in range(10, n):
    c1, o1 = close[i - 1], open_[i - 1]
    c2, o2 = close[i], open_[i]

    # 1. Opposite Colors
    is_bull1 = c1 > o1
    is_bull2 = c2 > o2
    if is_bull1 != is_bull2:
      b1, b2 = abs(c1 - o1), abs(c2 - o2)
      # 2. Both Long
      avg_i = s_body * 0.1
      if b2 > avg_i:
        # avg[i-1] = sum(i-11..i-2) / 10
        # For i=10, sh_sum = sum(-1..8)? Need 11 bars.
        if i >= 11:
          avg_i1 = (s_body - b1 + abs(close[i - 11] - open_[i - 11])) * 0.1
          if b1 > avg_i1:
            # 3. Equal Closes
            # avg_eq[i] = sum(i-5..i-1) / 5 * 0.05 = s_range * 0.01
            # cdl_counterattack uses avg_equal[i-1]
            # avg_equal[i-1] = sum(i-6..i-2) / 5 * 0.05
            sh_rng = s_range - (high[i - 1] - low[i - 1]) + (high[i - 6] - low[i - 6])
            if abs(c1 - c2) <= (sh_rng * 0.01 + 1e-12):
              out[i] = 100 if is_bull2 else -100

    # Updates
    val_body = abs(close[i] - open_[i])
    val_range = high[i] - low[i]
    s_body += val_body - abs(close[i - 10] - open_[i - 10])
    s_range += val_range - (high[i - 5] - low[i - 5])

  return out

  # Rolling sums: body_10, range_5
  s_body = 0.0
  s_range = 0.0
  for k in range(11):
    s_body += abs(close[k] - open_[k])
    if k >= 5 and k <= 9:
      s_range += high[k] - low[k]

  # Current s_body is sum(0..10)
  # Current s_range is sum(5..9)

  for i in range(11, n):
    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]

    # 1. Opposite Colors (Most restrictive)
    is_bull1 = c1 > o1
    is_bull2 = c2 > o2
    if is_bull1 != is_bull2:
      # 2. Both Long bodies
      # avg_body_long_i (signal) needs mean(i-10..i-1) = s_body - body[i] + body[i-10]? No.
      # Simplified: body_sum_10 at index i should be sum(i-10..i-1).
      # Current s_body is sum(i-11..i-1)?

      b1 = abs(c1 - o1)
      b2 = abs(c2 - o2)

      # avg_body at i-1 (prior): sum(i-11..i-2) / 10
      # avg_body at i (signal): sum(i-10..i-1) / 10

      # Correct indices for Counterattack in TA-Lib:
      # Let's fix rolling sum logic to be very precise.

      # REFACTORED SUM LOGIC:
      # s_body = sum(i-10..i-1)
      # s_range = sum(i-5..i-1)

      avg_long_i = s_body * 0.1
      if b2 > avg_long_i:
        avg_long_i1 = (s_body - b1 + abs(close[i - 11] - open_[i - 11])) * 0.1
        if b1 > avg_long_i1:
          # 3. Equal Closes
          avg_eq_i1 = (
            s_range - (high[i - 1] - low[i - 1]) + (high[i - 6] - low[i - 6])
          ) * 0.01  # 5.0 * 0.05
          if abs(c1 - c2) <= (avg_eq_i1 + 1e-12):
            out[i] = 100 if is_bull2 else -100

    # Updates
    s_body += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    s_range += (high[i] - low[i]) - (high[i - 5] - low[i - 5])

  return out

  # Initialize rolling sums
  # body_sum_10 for [0, 10) -> avg at i=10 is sum/10
  # rng_sum_5 for [0, 5) -> avg_equal at i=5 is sum/5 * 0.05
  body_sum_10 = 0.0
  rng_sum_5 = 0.0

  for j in range(10):
    body_sum_10 += abs(close[j] - open_[j])
    if j < 5:
      rng_sum_5 += high[j] - low[j]

  for i in range(10, n):
    # Compute shifted averages
    # avg_body_long[i] = mean(i-10..i-1) = body_sum_10 / 10
    # avg_body_long[i-1] = mean(i-11..i-2) = (body_sum_10 - body[i-1] + body[i-11]) / 10
    body_i_1 = abs(close[i - 1] - open_[i - 1])
    body_i_11 = abs(close[i - 11] - open_[i - 11]) if i >= 11 else 0.0
    avg_body_long_i = body_sum_10 / 10.0
    avg_body_long_i1 = (body_sum_10 - body_i_1 + body_i_11) / 10.0

    # avg_equal[i-1] = mean(i-6..i-2) * 0.05 = (rng_sum_5 - rng[i-1] + rng[i-6]) / 5 * 0.05
    rng_i_1 = high[i - 1] - low[i - 1]
    rng_i_6 = (high[i - 6] - low[i - 6]) if i >= 6 else 0.0
    avg_equal_i1 = (rng_sum_5 - rng_i_1 + rng_i_6) / 5.0 * 0.05

    o1, c1 = open_[i - 1], close[i - 1]
    o2, c2 = open_[i], close[i]

    # 1. Opposite Colors
    is_bull1 = c1 > o1
    is_bull2 = c2 > o2

    if is_bull1 != is_bull2:
      # 2. Both Long bodies
      body1 = abs(c1 - o1)
      body2 = abs(c2 - o2)

      if body1 > avg_body_long_i1 and body2 > avg_body_long_i:
        # 3. Equal Closes
        if abs(c1 - c2) <= avg_equal_i1:
          if is_bull2:  # Bear then Bull (Bullish Counter)
            out[i] = 100
          else:  # Bull then Bear (Bearish Counter)
            out[i] = -100

    # Update sums
    body_sum_10 += abs(close[i] - open_[i]) - abs(close[i - 10] - open_[i - 10])
    rng_sum_5 += (high[i] - low[i]) - (high[i - 5] - low[i - 5])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Doji Star (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Rolling sums: body_10, range_10
  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    rng_i = high[i] - low[i]

    if i >= 11:
      # Pattern i-1, i
      # Dependencies: avg_long[i-1], avg_doji[i]
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      avg_long_i1 = (body_sum_10 - body_i_1 + body_i_11) / 10.0
      avg_doji_i = range_sum_10 / 10.0 * 0.1

      o1, c1 = open_[i - 1], close[i - 1]
      o2, c2 = open_[i], close[i]
      body1 = abs(c1 - o1)
      body2 = abs(c2 - o2)

      if body1 > avg_long_i1 and body2 <= avg_doji_i:
        top1 = max(o1, c1)
        bot1 = min(o1, c1)
        top2 = max(o2, c2)
        bot2 = min(o2, c2)
        if c1 > o1 and bot2 > top1:
          out[i] = -100
        elif c1 < o1 and top2 < bot1:
          out[i] = 100

    # Update sums
    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += rng_i - abs(high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_conceal_baby_swallow_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Concealing Baby Swallow (Fused Hyper-Perf)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 14:
    return out

  # SMA(Range, 10, shifted) factor 0.1
  # Baseline for candle k is mean(range[k-10..k-1]) * 0.1

  s_range = 0.0
  for k in range(13):
    s_range += high[k] - low[k]

  # i=13:
  # mean(i-10..i-1) = sum(3..12) / 10
  # mean(i-12..i-3) = sum(1..10) / 10
  # mean(i-13..i-4) = sum(0..9) / 10

  s_range_i3 = 0.0  # for i-3 (mean(i-13..i-4)) -> sum(0..9)
  for k in range(10):
    s_range_i3 += high[k] - low[k]

  s_range_i2 = s_range_i3 + (high[10] - low[10]) - (high[0] - low[0])  # sum(1..10)
  s_range_i1 = s_range_i2 + (high[11] - low[11]) - (high[1] - low[1])  # sum(2..11)
  s_range_i = s_range_i1 + (high[12] - low[12]) - (high[2] - low[2])  # sum(3..12)

  for i in range(13, n):
    c1, o1 = close[i - 3], open_[i - 3]
    c2, o2 = close[i - 2], open_[i - 2]
    c3, o3 = close[i - 1], open_[i - 1]
    c4, o4 = close[i], open_[i]

    # 1. Colors check: All Black (Most restrictive rejection)
    if c1 < o1 and c2 < o2 and c3 < o3 and c4 < o4:
      # 2. 1st Marubozu
      limit_i3 = s_range_i3 * 0.01  # 10.0 * 0.1 / 10 = 0.01
      if (high[i - 3] - o1) < (limit_i3 + 1e-12) and (c1 - low[i - 3]) < (
        limit_i3 + 1e-12
      ):
        # 3. 2nd Marubozu
        limit_i2 = s_range_i2 * 0.01
        if (high[i - 2] - o2) < (limit_i2 + 1e-12) and (c2 - low[i - 2]) < (
          limit_i2 + 1e-12
        ):
          # 4. Pattern specifics (Gap Down, High invade, Engulf)
          if o3 < c2 and high[i - 1] > c2:
            if o4 > high[i - 1] and c4 < low[i - 1]:
              out[i] = 100

    # Advance sums
    # s_range_i3 (i-3) -> needs mean( (i+1)-13 .. (i+1)-4 ) = mean(i-12..i-3)
    s_range_i3 += (high[i - 3] - low[i - 3]) - (high[i - 13] - low[i - 13])
    # s_range_i2 (i-2) -> needs mean( (i+1)-12 .. (i+1)-3 ) = mean(i-11..i-2)
    s_range_i2 += (high[i - 2] - low[i - 2]) - (high[i - 12] - low[i - 12])
    # s_range_i1 (i-1) -> needs mean( (i+1)-11 .. (i+1)-2 ) = mean(i-10..i-1)
    s_range_i1 += (high[i - 1] - low[i - 1]) - (high[i - 11] - low[i - 11])
    # s_range_i (i) -> needs mean( (i+1)-10 .. (i+1)-1 ) = mean(i-9..i)
    s_range_i += (high[i] - low[i]) - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_harami_cross_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Harami Cross (Optimized Fused)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # rb_sum tracks sum(rb, i-10..i-1)
  # hl_sum tracks sum(high-low, i-10..i-1)
  rb_sum = 0.0
  hl_sum = 0.0
  for i in range(1, 11):
    rb_sum += abs(close[i] - open_[i])
    hl_sum += high[i] - low[i]

  # Initial state for the loop (for i=11, prev is i=10)
  # We need to know if body[10] was Long relative to avg(0..9)
  body_10 = abs(close[10] - open_[10])
  body_0 = abs(close[0] - open_[0])
  sum_0_9 = rb_sum - body_10 + body_0
  is_prev_long = (body_10 * 10.0) >= sum_0_9

  for i in range(11, n):
    rb = abs(close[i] - open_[i])
    hl = high[i] - low[i]

    # Check 1: Current body is Doji (<= 1% of avg range)
    # Note: hl_sum is sum(i-10..i-1) of High-Low
    is_doji = (rb * 100.0) <= hl_sum

    # Check 2: Previous body was Long (already computed)
    if is_prev_long and is_doji:
      o1, c1 = open_[i - 1], close[i - 1]
      o2, c2 = open_[i], close[i]

      # Harami (containment) logic
      p_t = o1 if o1 > c1 else c1
      p_b = c1 if o1 > c1 else o1
      c_t = o2 if o2 > c2 else c2
      c_b = c2 if o2 > c2 else o2

      if c_t <= p_t and c_b >= p_b:
        if c_t < p_t and c_b > p_b:
          out[i] = 100 if c1 < o1 else -100
        else:
          out[i] = 80 if c1 < o1 else -80

    # Prepare for next iteration
    # Current body becomes previous body
    # We need to know if current body is Long relative to current avg(i-10..i-1)
    # rb_sum is currently sum(i-10..i-1)
    is_prev_long = (rb * 10.0) >= rb_sum

    # Advance sums for next i
    body_trailing = abs(close[i - 10] - open_[i - 10])
    hl_trailing = high[i - 10] - low[i - 10]

    rb_sum += rb - body_trailing
    hl_sum += hl - hl_trailing

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
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Morning Doji Star (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    rng_i = high[i] - low[i]
    if i >= 12:
      # avg_short[i]
      avg_short_i = body_sum_10 / 10.0

      # avg_doji[i-1]
      rng_i_1 = high[i - 1] - low[i - 1]
      rng_i_11 = high[i - 11] - low[i - 11]
      avg_doji_i1 = (range_sum_10 - rng_i_1 + rng_i_11) / 10.0 * 0.1

      # avg_long[i-2]
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      body_i_2 = abs(close[i - 2] - open_[i - 2])
      body_i_12 = abs(close[i - 12] - open_[i - 12])
      avg_long_i2 = (body_sum_10 - body_i_1 - body_i_2 + body_i_11 + body_i_12) / 10.0

      o2, c2 = open_[i - 2], close[i - 2]
      o1, c1 = open_[i - 1], close[i - 1]
      c0, o0 = close[i], open_[i]

      body2 = abs(c2 - o2)
      body1 = abs(c1 - o1)
      body0 = abs(c0 - o0)

      if (
        body2 > avg_long_i2
        and c2 < o2
        and body1 <= avg_doji_i1
        and max(o1, c1) < min(o2, c2)
        and body0 > avg_short_i
        and c0 > o0
        and c0 > c2 + body2 * penetration
      ):
        out[i] = 100

    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_evening_doji_star_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  penetration: float = 0.3,
) -> NDArray[np.int32]:
  """Detect Evening Doji Star (Fused SMA)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  body_sum_10 = 0.0
  range_sum_10 = 0.0
  for i in range(10):
    body_sum_10 += abs(close[i] - open_[i])
    range_sum_10 += high[i] - low[i]

  for i in range(10, n):
    body_i = abs(close[i] - open_[i])
    rng_i = high[i] - low[i]
    if i >= 12:
      # avg_short[i]
      avg_short_i = body_sum_10 / 10.0
      # avg_doji[i-1]
      rng_i_1 = high[i - 1] - low[i - 1]
      rng_i_11 = high[i - 11] - low[i - 11]
      avg_doji_i1 = (range_sum_10 - rng_i_1 + rng_i_11) / 10.0 * 0.1
      # avg_long[i-2]
      body_i_1 = abs(close[i - 1] - open_[i - 1])
      body_i_11 = abs(close[i - 11] - open_[i - 11])
      body_i_2 = abs(close[i - 2] - open_[i - 2])
      body_i_12 = abs(close[i - 12] - open_[i - 12])
      avg_long_i2 = (body_sum_10 - body_i_1 - body_i_2 + body_i_11 + body_i_12) / 10.0

      o2, c2 = open_[i - 2], close[i - 2]
      o1, c1 = open_[i - 1], close[i - 1]
      c0, o0 = close[i], open_[i]

      body2 = abs(c2 - o2)
      body1 = abs(c1 - o1)
      body0 = abs(c0 - o0)

      if (
        body2 > avg_long_i2
        and c2 > o2
        and body1 <= avg_doji_i1
        and min(o1, c1) > max(o2, c2)
        and body0 > avg_short_i
        and c0 < o0
        and c0 < c2 - body2 * penetration
      ):
        out[i] = -100

    body_sum_10 += body_i - abs(close[i - 10] - open_[i - 10])
    range_sum_10 += rng_i - (high[i - 10] - low[i - 10])

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_three_stars_in_south_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Three Stars In The South (Fused History)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 13:
    return out

  body_sum = 0.0
  range_sum = 0.0

  # Circular buffers for sums at i, i-1, i-2 (size 3)
  sb_hist = np.zeros(3)
  sr_hist = np.zeros(3)

  # Initial sum 0-9
  for k in range(10):
    body_sum += abs(close[k] - open_[k])
    range_sum += high[k] - low[k]

  sb_hist[10 % 3] = body_sum
  sr_hist[10 % 3] = range_sum

  for i in range(11, n):
    # Update rolling sums for index i (sum over [i-10, i-1])
    body_sum += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
    range_sum += (high[i - 1] - low[i - 1]) - (high[i - 11] - low[i - 11])

    sb_hist[i % 3] = body_sum
    sr_hist[i % 3] = range_sum

    if i < 12:
      continue

    # 3StarsSouth Logic (indices i-2, i-1, i)
    # avg_body_long[i-2] = sb_hist[(i-2)%3] / 10.0
    # avg_shadow_very_short[i-1] = sr_hist[(i-1)%3] * 0.01
    # avg_body_short[i] = sb_hist[i%3] / 10.0
    # avg_shadow_very_short[i] = sr_hist[i%3] * 0.01

    o2, l2, c2 = open_[i - 2], low[i - 2], close[i - 2]
    o1, h1, l1, c1 = open_[i - 1], high[i - 1], low[i - 1], close[i - 1]
    o0, h0, l0, c0 = open_[i], high[i], low[i], close[i]

    # Conditions based on original TA-Lib logic
    if (
      c2 < o2
      and c1 < o1
      and c0 < o0
      and (o2 - c2) > (sb_hist[(i - 2) % 3] * 0.1)
      and (c2 - l2) > (o2 - c2)
      and (o1 - c1) < (o2 - c2)
      and o1 > c2
      and o1 <= high[i - 2]
      and l1 < c2
      and l1 >= l2
      and (c1 - l1) > (sr_hist[(i - 1) % 3] * 0.01)
      and abs(c0 - o0) < (sb_hist[i % 3] * 0.1)
      and (c0 - l0) < (sr_hist[i % 3] * 0.01)
      and (h0 - o0) < (sr_hist[i % 3] * 0.01)
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


@jit(nopython=True, fastmath=True, parallel=True)
def detect_harami_cross_parallel(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Harami Cross pattern (Parallel Chunked Strategy #2)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)

    proc_start = max(start, 11)
    if proc_start >= end:
      continue

    # Warmup for rb_sum and hl_sum
    rb_sum = 0.0
    hl_sum = 0.0
    for k in range(proc_start - 10, proc_start):
      rb_sum += abs(close[k] - open_[k])
      hl_sum += high[k] - low[k]

    # Warmup for is_prev_long
    body_prev = abs(close[proc_start - 1] - open_[proc_start - 1])
    rb_sum_prev = 0.0
    for k in range(proc_start - 11, proc_start - 1):
      rb_sum_prev += abs(close[k] - open_[k])
    is_prev_long = (body_prev * 10.0) >= rb_sum_prev

    for i in range(proc_start, end):
      rb = abs(close[i] - open_[i])
      hl = high[i] - low[i]

      # Harami Cross BodyDoji requirement: rb <= 10% of HL_SUM / 10
      is_doji = (rb * 100.0) <= hl_sum

      if is_prev_long and is_doji:
        o1, c1 = open_[i - 1], close[i - 1]
        o2, c2 = open_[i], close[i]
        p_t = o1 if o1 > c1 else c1
        p_b = c1 if o1 > c1 else o1
        c_t = o2 if o2 > c2 else c2
        c_b = c2 if o2 > c2 else o2

        if c_t <= p_t and c_b >= p_b:
          if c_t < p_t and c_b > p_b:
            out[i] = 100 if c1 < o1 else -100
          else:
            out[i] = 80 if c1 < o1 else -80

      # Advance
      is_prev_long = (rb * 10.0) >= rb_sum
      body_trailing = abs(close[i - 10] - open_[i - 10])
      hl_trailing = high[i - 10] - low[i - 10]
      rb_sum += rb - body_trailing
      hl_sum += hl - hl_trailing

  return out


@jit(nopython=True, fastmath=True, parallel=True)
def detect_harami_parallel(
  open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
  """Detect Harami pattern (Parallel Chunked Strategy #2)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  # Strategy #2: Parallel Chunking with Local Warmup
  # Used because Harami on 1M rows benefits from threading

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)

    proc_start = max(start, 11)
    if proc_start >= end:
      continue

    rb_sum = 0.0
    for k in range(proc_start - 10, proc_start):
      rb_sum += abs(close[k] - open_[k])

    rb_sum_prev = 0.0
    for k in range(proc_start - 11, proc_start - 1):
      rb_sum_prev += abs(close[k] - open_[k])

    body_prev = abs(close[proc_start - 1] - open_[proc_start - 1])
    is_prev_long = (body_prev * 10.0) >= rb_sum_prev

    for i in range(proc_start, end):
      rb = abs(close[i] - open_[i])
      is_short = (rb * 10.0) <= rb_sum

      if is_prev_long and is_short:
        o1, c1 = open_[i - 1], close[i - 1]
        o2, c2 = open_[i], close[i]
        p_t = o1 if o1 > c1 else c1
        p_b = c1 if o1 > c1 else o1
        c_t = o2 if o2 > c2 else c2
        c_b = c2 if o2 > c2 else o2

        if c_t <= p_t and c_b >= p_b:
          if c_t < p_t and c_b > p_b:
            out[i] = 100 if c1 < o1 else -100
          else:
            out[i] = 80 if c1 < o1 else -80

      # Advance
      is_prev_long = (rb * 10.0) >= rb_sum
      body_trailing = abs(close[i - 10] - open_[i - 10])
      rb_sum += rb - body_trailing

  return out

  # `avg`: returns sum of `period` ranges ENDING at `index-1`.
  # So for `i-1`, it sums ranges ending at `i-2`.


# Corrected Logic for Ladder Bottom Parallel
@jit(nopython=True, fastmath=True, parallel=True)
def detect_ladder_bottom_parallel(
  open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
  """Detect Ladder Bottom pattern (Parallel)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 20:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)
    proc_start = max(start, 20)
    if proc_start >= end:
      continue

    # Warmup: Sum ending at proc_start-2
    # Window: [proc_start-11 ... proc_start-2]
    s_svs = 0.0
    for k in range(proc_start - 11, proc_start - 1):
      s_svs += high[k] - low[k]

    for i in range(proc_start, end):
      c5, o5 = close[i], open_[i]
      if c5 > o5:
        c4, o4, h4 = close[i - 1], open_[i - 1], high[i - 1]
        if c4 < o4 and (h4 - o4) > (s_svs * 0.01):
          if o5 > o4 and c5 > h4:
            c1, o1 = close[i - 4], open_[i - 4]
            c2, o2 = close[i - 3], open_[i - 3]
            c3, o3 = close[i - 2], open_[i - 2]
            if (
              c1 < o1
              and c2 < o2
              and c3 < o3
              and o2 < o1
              and o3 < o2
              and c2 < c1
              and c3 < c2
            ):
              out[i] = 100

      # Advance: At end of i, prepare for i+1.
      # Next iter needs sum ending at i-1.
      # Curr sum ends at i-2.
      # Add i-1, Remove i-11.
      s_svs += (high[i - 1] - low[i - 1]) - (high[i - 11] - low[i - 11])

  return out


@jit(nopython=True, fastmath=True, parallel=True)
def detect_on_neck_parallel(
  open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
  """Detect On Neck pattern (Parallel)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 15:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)
    proc_start = max(start, 15)
    if proc_start >= end:
      continue

    # Warmup
    # Body Long (10) for i: needs index i-1. Avg ending at i-2.
    # Window [proc_start-11 ... proc_start-2]
    s_body = 0.0
    for k in range(proc_start - 11, proc_start - 1):
      s_body += abs(close[k] - open_[k])

    # Range (5) for i: needs index i-1. Avg ending at i-2.
    # Window [proc_start-6 ... proc_start-2]
    s_range = 0.0
    for k in range(proc_start - 6, proc_start - 1):
      s_range += high[k] - low[k]

    for i in range(proc_start, end):
      c1, o1 = close[i - 1], open_[i - 1]
      c0, o0 = close[i], open_[i]

      if c1 < o1 and c0 > o0:
        if (o1 - c1) > (s_body * 0.1 - 1e-12):
          l1 = low[i - 1]
          limit = s_range * 0.01
          if o0 < l1 and c0 <= (l1 + limit + 1e-12) and c0 >= (l1 - limit - 1e-12):
            out[i] = -100

      # Advance
      # Prepare for i+1. Needs sum ending at i-1.
      # Add i-1, remove i-11 (for body) / i-6 (for range)
      s_body += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
      s_range += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, fastmath=True, parallel=True)
def detect_mat_hold_parallel(
  open_: np.ndarray,
  high: np.ndarray,
  low: np.ndarray,
  close: np.ndarray,
  penetration: float = 0.5,
) -> np.ndarray:
  """Detect Mat Hold pattern (Parallel)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 29:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)
    proc_start = max(start, 29)
    if proc_start >= end:
      continue

    # Warmup
    # s_body_25: for i, needs sum ending at i-5.
    # Window: [proc_start-29 ... proc_start-5]
    s_body_25 = 0.0
    for k in range(proc_start - 29, proc_start - 4):
      s_body_25 += abs(close[k] - open_[k])

    # s_body_10_i3 (i-3): needs sum ending at i-4.
    # Window: [proc_start-13 ... proc_start-4]
    s_body_10_i3 = 0.0
    for k in range(proc_start - 13, proc_start - 3):
      s_body_10_i3 += abs(close[k] - open_[k])

    # s_body_10_i2 (i-2): needs sum ending at i-3.
    # Window: [proc_start-12 ... proc_start-3]
    s_body_10_i2 = 0.0
    for k in range(proc_start - 12, proc_start - 2):
      s_body_10_i2 += abs(close[k] - open_[k])

    # s_body_10_i1 (i-1): needs sum ending at i-2.
    # Window: [proc_start-11 ... proc_start-2]
    s_body_10_i1 = 0.0
    for k in range(proc_start - 11, proc_start - 1):
      s_body_10_i1 += abs(close[k] - open_[k])

    for i in range(proc_start, end):
      o4, c4 = open_[i - 4], close[i - 4]
      o0, c0 = open_[i], close[i]

      if c4 > o4 and c0 > o0:
        o3, c3 = open_[i - 3], close[i - 3]
        o2, c2 = open_[i - 2], close[i - 2]
        o1, c1 = open_[i - 1], close[i - 1]

        if c3 < o3:
          if (c4 - o4) > (s_body_25 / 25.0):
            if (
              abs(c3 - o3) < (s_body_10_i3 / 10.0)
              and abs(c2 - o2) < (s_body_10_i2 / 10.0)
              and abs(c1 - o1) < (s_body_10_i1 / 10.0)
            ):
              limit = c4 - (c4 - o4) * penetration
              if c3 > c4:
                if (
                  min(o2, c2) < c4
                  and min(o2, c2) > (limit - 1e-12)
                  and min(o1, c1) < c4
                  and min(o1, c1) > (limit - 1e-12)
                ):
                  if max(o2, c2) < o3:
                    if max(o1, c1) < max(o2, c2):
                      if o0 > c1:
                        if c0 > max(high[i - 3], high[i - 2], high[i - 1]):
                          out[i] = 100

      # Advance
      # Prepare for i+1.
      # s_body_25: add i-4, remove i-29
      s_body_25 += abs(close[i - 4] - open_[i - 4]) - abs(close[i - 29] - open_[i - 29])
      # s_body_10_i3: add i-3, remove i-13
      s_body_10_i3 += abs(close[i - 3] - open_[i - 3]) - abs(
        close[i - 13] - open_[i - 13]
      )
      # s_body_10_i2: add i-2, remove i-12
      s_body_10_i2 += abs(close[i - 2] - open_[i - 2]) - abs(
        close[i - 12] - open_[i - 12]
      )
      # s_body_10_i1: add i-1, remove i-11
      s_body_10_i1 += abs(close[i - 1] - open_[i - 1]) - abs(
        close[i - 11] - open_[i - 11]
      )

  return out


@jit(nopython=True, fastmath=True, parallel=True)
def detect_stick_sandwich_parallel(
  open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
  """Detect Stick Sandwich pattern (Parallel)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 8:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)

    proc_start = max(start, 7)
    if proc_start >= end:
      continue

    # Warmup CandleAverage(Equal, period 5) at i-2.
    # Window: [proc_start-7 ... proc_start-3]
    s_eq = 0.0
    for k in range(proc_start - 7, proc_start - 2):
      s_eq += high[k] - low[k]

    for i in range(proc_start, end):
      c1, o1 = close[i - 2], open_[i - 2]
      c2, o2 = close[i - 1], open_[i - 1]
      c3, o3 = close[i], open_[i]

      if c1 < o1 and c2 > o2 and c3 < o3:
        if low[i - 1] > c1:
          eq_avg = s_eq * 0.01
          if c3 <= c1 + eq_avg and c3 >= c1 - eq_avg:
            out[i] = 100

      # Advance
      s_eq += (high[i - 2] - low[i - 2]) - (high[i - 7] - low[i - 7])

  return out


@jit(nopython=True, fastmath=True, parallel=True)
def detect_thrusting_parallel(
  open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
  """Detect Thrusting pattern (Parallel)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)
    proc_start = max(start, 11)
    if proc_start >= end:
      continue

    # Warmup
    s_body = 0.0
    for k in range(proc_start - 11, proc_start - 1):
      s_body += abs(close[k] - open_[k])

    s_range = 0.0
    for k in range(proc_start - 6, proc_start - 1):
      s_range += high[k] - low[k]

    for i in range(proc_start, end):
      c1, o1 = close[i - 1], open_[i - 1]
      c0, o0 = close[i], open_[i]

      if c1 < o1 and c0 > o0:
        if (o1 - c1) > (s_body * 0.1 - 1e-12):
          limit = s_range * 0.01
          l1 = low[i - 1]
          if (
            o0 < l1
            and c0 >= (c1 + limit + 1e-12)
            and c0 <= (c1 + (o1 - c1) * 0.5 + 1e-12)
          ):
            out[i] = -100

      # Advance
      s_body += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
      s_range += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, fastmath=True, parallel=True)
def detect_matching_low_parallel(
  open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
  """Detect Matching Low pattern (Parallel)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 7:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)
    proc_start = max(start, 6)
    if proc_start >= end:
      continue

    # Warmup
    s_eq = 0.0
    for k in range(proc_start - 6, proc_start - 1):
      s_eq += high[k] - low[k]

    for i in range(proc_start, end):
      c1, o1 = close[i - 1], open_[i - 1]
      c0, o0 = close[i], open_[i]

      if c0 < o0 and c1 < o1:
        eq_avg = s_eq * 0.01
        if c0 <= c1 + eq_avg and c0 >= c1 - eq_avg:
          out[i] = 100

      # Advance
      s_eq += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out


@jit(nopython=True, fastmath=True, parallel=True)
def detect_in_neck_parallel(
  open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
  """Detect In-Neck pattern (Parallel)."""
  n = len(close)
  out = np.zeros(n, dtype=np.int32)
  if n < 12:
    return out

  num_chunks = 16
  chunk_size = (n + num_chunks - 1) // num_chunks

  for c in prange(num_chunks):
    start = c * chunk_size
    end = min((c + 1) * chunk_size, n)
    proc_start = max(start, 11)
    if proc_start >= end:
      continue

    # Warmup
    s_body = 0.0
    for k in range(proc_start - 11, proc_start - 1):
      s_body += abs(close[k] - open_[k])

    s_range = 0.0
    for k in range(proc_start - 6, proc_start - 1):
      s_range += high[k] - low[k]

    for i in range(proc_start, end):
      c1, o1 = close[i - 1], open_[i - 1]
      c0, o0 = close[i], open_[i]

      if c1 < o1 and c0 > o0:
        if (o1 - c1) > (s_body * 0.1 - 1e-12):
          limit = s_range * 0.01
          l1 = low[i - 1]
          if o0 < l1 and c0 <= (c1 + limit + 1e-12) and c0 >= c1:
            out[i] = -100

      # Advance
      s_body += abs(close[i - 1] - open_[i - 1]) - abs(close[i - 11] - open_[i - 11])
      s_range += (high[i - 1] - low[i - 1]) - (high[i - 6] - low[i - 6])

  return out
