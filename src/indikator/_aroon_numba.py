"""Numba-optimized AROON indicator calculation.

This module contains JIT-compiled functions for AROON calculation.
Uses hybrid approach: simple loop for period < 25, deque for period >= 25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Threshold where deque becomes faster than simple loop
# Benchmarks show Deque (O(N)) is faster than Simple (O(N*P)) even at P=25 (37ms vs 96ms).
# Crossover likely around P=10-15.
DEQUE_THRESHOLD = 15


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def _aroon_simple(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """AROON using TA-lib's lazy rescan algorithm - amortized O(n)."""
  n = len(high)

  if n < period + 1:
    return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

  aroon_up = np.empty(n)
  aroon_down = np.empty(n)
  aroon_osc = np.empty(n)
  aroon_up[:period] = np.nan
  aroon_down[:period] = np.nan
  aroon_osc[:period] = np.nan

  # Initialize trackers - precompute first window
  # Note: AROON uses period+1 elements (indices 0..period) for first output at index period
  highest_idx = 0
  highest = high[0]
  lowest_idx = 0
  lowest = low[0]

  for i in range(1, period + 1):
    if high[i] >= highest:
      highest_idx = i
      highest = high[i]
    if low[i] <= lowest:
      lowest_idx = i
      lowest = low[i]

  # Calculate first AROON values at index 'period'
  inv_period = 100.0 / period
  aroon_up[period] = (period - (period - highest_idx)) * inv_period
  aroon_down[period] = (period - (period - lowest_idx)) * inv_period
  aroon_osc[period] = aroon_up[period] - aroon_down[period]

  # Main loop from period+1 onwards
  for today in range(period + 1, n):
    trailing_idx = today - period

    # Update lowest low - lazy rescan
    tmp_low = low[today]
    if lowest_idx < trailing_idx:
      lowest_idx = trailing_idx
      lowest = low[trailing_idx]
      for i in range(trailing_idx + 1, today + 1):
        if low[i] <= lowest:
          lowest_idx = i
          lowest = low[i]
    elif tmp_low <= lowest:
      lowest_idx = today
      lowest = tmp_low

    # Update highest high - lazy rescan
    tmp_high = high[today]
    if highest_idx < trailing_idx:
      highest_idx = trailing_idx
      highest = high[trailing_idx]
      for i in range(trailing_idx + 1, today + 1):
        if high[i] >= highest:
          highest_idx = i
          highest = high[i]
    elif tmp_high >= highest:
      highest_idx = today
      highest = tmp_high

    # Calculate AROON values
    periods_since_high = today - highest_idx
    periods_since_low = today - lowest_idx

    aroon_up[today] = (period - periods_since_high) * inv_period
    aroon_down[today] = (period - periods_since_low) * inv_period
    aroon_osc[today] = aroon_up[today] - aroon_down[today]

  return aroon_up, aroon_down, aroon_osc


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _aroon_deque(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """AROON using O(n) monotonic deque - optimized inline implementation."""
  n = len(high)

  if n < period + 1:
    return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

  aroon_up = np.empty(n)
  aroon_down = np.empty(n)
  aroon_osc = np.empty(n)
  aroon_up[:period] = np.nan
  aroon_down[:period] = np.nan
  aroon_osc[:period] = np.nan

  # Use power of 2 capacity for bitwise masking (avoid Modulo)
  # Capacity needs to be > period.
  cap_needed = period + 1
  # Find next power of 2
  capacity = 1
  while capacity < cap_needed:
    capacity <<= 1

  mask = capacity - 1

  dq_high = np.empty(capacity, dtype=np.int64)
  dq_low = np.empty(capacity, dtype=np.int64)

  h_head, h_tail = 0, 0
  l_head, l_tail = 0, 0

  for i in range(n):
    min_valid_idx = i - period

    # --- High Deque (Max) ---
    # 1. Expire
    while h_head != h_tail and dq_high[h_head & mask] <= min_valid_idx:
      h_head += 1

    # 2. Push
    curr_h = high[i]
    while h_head != h_tail:
      prev_idx = dq_high[(h_tail - 1) & mask]
      if high[prev_idx] <= curr_h:
        h_tail -= 1
      else:
        break
    dq_high[h_tail & mask] = i
    h_tail += 1

    # --- Low Deque (Min) ---
    # 1. Expire
    while l_head != l_tail and dq_low[l_head & mask] <= min_valid_idx:
      l_head += 1

    # 2. Push
    curr_l = low[i]
    while l_head != l_tail:
      prev_idx = dq_low[(l_tail - 1) & mask]
      if low[prev_idx] >= curr_l:
        l_tail -= 1
      else:
        break
    dq_low[l_tail & mask] = i
    l_tail += 1

    # --- Calculate ---
    if i >= period:
      highest_idx = dq_high[h_head & mask]
      lowest_idx = dq_low[l_head & mask]

      # Optimization: precompute floats?
      # aroon_up = 100 * (period - (i - highest_idx)) / period
      #          = 100 * (1 - (i - highest_idx)/period)
      #          = 100 - (100/period)*(i - highest_idx)
      # 100/period is constant.

      dist_h = i - highest_idx
      dist_l = i - lowest_idx

      # Using standard formula calculation for now.
      # Division by period is fast enough (or precomputed reciprocal)
      # But let's stick to simple logic first.

      up_val = 100.0 * (period - dist_h) / period
      down_val = 100.0 * (period - dist_l) / period

      aroon_up[i] = up_val
      aroon_down[i] = down_val
      aroon_osc[i] = up_val - down_val

  return aroon_up, aroon_down, aroon_osc


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_aroon_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled AROON using TA-lib's lazy rescan algorithm.

  Uses amortized O(n) lazy rescan approach: tracks the index of current
  min/max and only rescans when that index falls outside the sliding window.

  Args:
    high: Array of high prices
    low: Array of low prices
    period: Lookback period (typically 25)

  Returns:
    Tuple of (aroon_up, aroon_down, aroon_osc) arrays
  """
  return _aroon_simple(high, low, period)
