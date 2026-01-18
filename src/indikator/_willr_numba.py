"""Numba-optimized Williams %R calculation.

This module contains JIT-compiled functions for Williams %R calculation.
Uses hybrid approach: simple loop for period < 25, deque for period >= 25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

from indikator._deque_numba import (
  deque_expire,
  deque_front,
  deque_push_max,
  deque_push_min,
)

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value
DEQUE_THRESHOLD = 25  # Use deque for period >= this


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def _willr_simple(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Williams %R using simple O(n*period) approach - faster for small periods."""
  n = len(close)
  willr = np.full(n, np.nan)

  if n < period:
    return willr

  for i in range(period - 1, n):
    highest_high = high[i - period + 1]
    lowest_low = low[i - period + 1]
    for j in range(i - period + 2, i + 1):
      highest_high = max(highest_high, high[j])
      lowest_low = min(lowest_low, low[j])

    range_hl = highest_high - lowest_low
    if range_hl > EPSILON:
      willr[i] = -100.0 * (highest_high - close[i]) / range_hl
    else:
      willr[i] = -50.0  # Neutral if no range

  return willr


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def _willr_deque(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Williams %R using O(n) monotonic deque - faster for large periods."""
  n = len(close)
  willr = np.full(n, np.nan)

  if n < period:
    return willr

  # Allocate deque buffers
  capacity = period + 1
  dq_high = np.zeros(capacity, dtype=np.int64)
  dq_low = np.zeros(capacity, dtype=np.int64)
  h_head, h_tail = 0, 0
  l_head, l_tail = 0, 0

  for i in range(n):
    min_valid_idx = i - period + 1

    # Update high deque (max)
    h_head = deque_expire(dq_high, h_head, h_tail, capacity, min_valid_idx)
    h_head, h_tail = deque_push_max(dq_high, h_head, h_tail, capacity, high, i)

    # Update low deque (min)
    l_head = deque_expire(dq_low, l_head, l_tail, capacity, min_valid_idx)
    l_head, l_tail = deque_push_min(dq_low, l_head, l_tail, capacity, low, i)

    if i >= period - 1:
      highest_high = high[deque_front(dq_high, h_head, capacity)]
      lowest_low = low[deque_front(dq_low, l_head, capacity)]

      range_hl = highest_high - lowest_low
      if range_hl > EPSILON:
        willr[i] = -100.0 * (highest_high - close[i]) / range_hl
      else:
        willr[i] = -50.0

  return willr


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_willr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled Williams %R with hybrid optimization.

  Uses simple loop for period < 25 (faster for small windows).
  Uses monotonic deque for period >= 25 (O(n) instead of O(n*period)).

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of closing prices
    period: Lookback period (typically 14)

  Returns:
    Array of Williams %R values
  """
  if period < DEQUE_THRESHOLD:
    return _willr_simple(high, low, close, period)
  return _willr_deque(high, low, close, period)
