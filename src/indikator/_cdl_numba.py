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
    # Cast boolean to int (0 or 1) then multiply by 100
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

    # Primitives (Branchless min/max)
    body_top = o if o > c else c
    body_bot = o if o < c else c

    real_body = body_top - body_bot
    upper_shadow = h - body_top
    lower_shadow = body_bot - l
    rng = h - l

    # Zero range check handled by logic or tiny epsilon?
    # Branchless way: condition & (rng > 0)

    body_val = real_body if real_body > 0 else 1e-9

    # Conditions (Boolean)
    # 1. Long lower shadow (>= 2 * body)
    c1 = lower_shadow >= (2.0 * body_val)

    # 2. Small upper shadow (<= 10% of range) to allow minor wick
    c2 = upper_shadow <= (0.1 * rng)

    # 3. Small body (<= 30% of range)
    c3 = real_body < (0.3 * rng)

    # 4. Range valid
    c4 = rng > 0

    # Fuse
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
  """Detect Engulfing pattern (Branchless).

  Evaluates both Bullish and Bearish conditions using bitwise logic
  and combines them: out = (is_bull * 100) - (is_bear * 100).
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    pc = close[i - 1]
    po = open_[i - 1]
    cc = close[i]
    co = open_[i]

    # Bullish Components
    # Prev Red: po > pc
    # Curr Green: cc > co
    # Engulf: co <= pc AND cc >= po
    is_bull = (po > pc) & (cc > co) & (co <= pc) & (cc >= po)

    # Bearish Components
    # Prev Green: pc > po
    # Curr Red: co < cc ?? No, co > cc (Red)
    # Engulf: co >= pc AND cc <= po
    is_bear = (pc > po) & (co > cc) & (co >= pc) & (cc <= po)

    # Combine (True=1, False=0)
    # If both true (impossible per logic), they cancel?
    # Can't be both Prev Red and Prev Green.
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

    # Bounds
    prev_top = po if po > pc else pc
    prev_bot = po if po < pc else pc

    curr_top = co if co > cc else cc
    curr_bot = co if co < cc else cc

    # Inside Bar Condition
    is_inside = (curr_top < prev_top) & (curr_bot > prev_bot)

    # Direction
    # Bullish Harami: Prev is Bearish (pc < po)
    # Bearish Harami: Prev is Bullish (pc > po)
    # We assign 100 if Prev Bearish, -100 if Prev Bullish.

    # Using branchless sign logic
    # val = (is_prev_bear * 100) - (is_prev_bull * 100)
    # Only apply if is_inside is True.

    is_prev_bear = pc < po
    is_prev_bull = pc >= po  # or just > ?

    # Logic:
    # out = is_inside * ( (is_prev_bear * 100) + (is_prev_bull * -100) )

    val = (is_prev_bear * 100) - (is_prev_bull * 100)
    out[i] = is_inside * val

  return out
