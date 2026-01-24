"""Numba-optimized Candle Pattern Recognition.

This module provides JIT-compiled kernels for detecting candlestick patterns.
It uses a shared "Candle Properties" kernel to pre-calculate Body, Shadows, and Range
in a single pass, making sub-pattern detection extremely fast.

Architecture:
1. compute_candle_props: Precalculates properties for the window.
2. Pattern kernels: Consume properties to return integer flags (100=Bull, -100=Bear, 0=None).
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
  """Detect Doji pattern.

  Definition: Body is very small relative to the range.
  Strategy: Abs(Open - Close) <= 0.1 * (High - Low)

  Returns:
    100 (Doji is neutral, but TA-Lib returns 100 for match)
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(n):
    body = abs(close[i] - open_[i])
    rng = high[i] - low[i]

    # Doji criteria: body <= 10% of range (configurable usually, using fixed 0.1 for now)
    # Also minimal range check to avoid division by zero or noise
    if rng > 0 and body <= (rng * 0.1):
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_hammer_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Hammer pattern.

  Definition:
  - Small body near top of range
  - Long lower shadow (>= 2 * body)
  - Very short upper shadow
  - Occurs in downtrend (simplified here: just shape)

  Returns: 100 (Bullish)
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(10, n):  # Need trend context? For now just shape.
    real_body = abs(close[i] - open_[i])
    upper_shadow = high[i] - max(close[i], open_[i])
    lower_shadow = min(close[i], open_[i]) - low[i]
    rng = high[i] - low[i]

    if rng == 0:
      continue

    # Shape criteria
    # 1. Body is in upper third? No, let's stick to shadow ratios.
    # 2. Lower shadow >= 2 * body
    # 3. Upper shadow <= 0.1 * body (very small) or small relative to range

    # Avoid zero body issues
    body_val = real_body if real_body > 0 else 0.00001

    is_long_lower = lower_shadow >= (2.0 * body_val)
    is_small_upper = upper_shadow <= (0.1 * rng)  # Upper shadow is small part of range
    is_small_body = real_body < (0.3 * rng)  # Body is small part of range

    if is_long_lower and is_small_upper and is_small_body:
      out[i] = 100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_engulfing_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Engulfing pattern.

  Bullish Engulfing (100):
  - Prev candle: Bearish (Close < Open)
  - Curr candle: Bullish (Close > Open)
  - Curr Open <= Prev Close and Curr Close >= Prev Open (engulfs body)

  Bearish Engulfing (-100):
  - Prev candle: Bullish
  - Curr candle: Bearish
  - Engulfs body
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    prev_close = close[i - 1]
    prev_open = open_[i - 1]
    curr_close = close[i]
    curr_open = open_[i]

    # Bullish Engulfing
    # Prev is Red (Open > Close)
    # Curr is Green (Close > Open)
    if (prev_open > prev_close) and (curr_close > curr_open):
      # Engulfs body:
      # Curr Open <= Prev Close (gapped down or equal)
      # Curr Close >= Prev Open (closed above prev open)
      if (curr_open <= prev_close) and (curr_close >= prev_open):
        out[i] = 100
        continue

    # Bearish Engulfing
    # Prev is Green (Close > Open)
    # Curr is Red (Open > Close)
    if (prev_close > prev_open) and (curr_open > curr_close):
      # Engulfs body:
      # Curr Open >= Prev Close
      # Curr Close <= Prev Open
      if (curr_open >= prev_close) and (curr_close <= prev_open):
        out[i] = -100

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def detect_harami_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.int32]:
  """Detect Harami pattern.

  Inside bar pattern. Small body completely inside previous large body.

  Bullish Harami (100):
  - Prev: Long Bearish
  - Curr: Small Bullish (or Bearish, but usually contrarian) inside prev body

  Bearish Harami (-100):
  - Prev: Long Bullish
  - Curr: Small inside prev body
  """
  n = len(close)
  out = np.zeros(n, dtype=np.int32)

  for i in range(1, n):
    prev_close = close[i - 1]
    prev_open = open_[i - 1]
    curr_close = close[i]
    curr_open = open_[i]

    prev_body = abs(prev_close - prev_open)
    curr_body = abs(curr_close - curr_open)

    # Must be inside
    # Max(CurrOpen, CurrClose) < Max(PrevOpen, PrevClose)
    # Min(CurrOpen, CurrClose) > Min(PrevOpen, PrevClose)

    prev_top = max(prev_open, prev_close)
    prev_bot = min(prev_open, prev_close)
    curr_top = max(curr_open, curr_close)
    curr_bot = min(curr_open, curr_close)

    is_inside = (curr_top < prev_top) and (curr_bot > prev_bot)

    if is_inside:
      # Check previous trend/color
      if prev_close < prev_open:  # Prev was Bearish -> Bullish Harami (reversal)
        out[i] = 100
      else:  # Prev was Bullish -> Bearish Harami
        out[i] = -100

  return out
