"""Numba-optimized AROON indicator calculation.

This module contains JIT-compiled functions for AROON calculation.
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

# Threshold where deque becomes faster than simple loop
DEQUE_THRESHOLD = 25


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def _aroon_simple(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """AROON using simple O(n*period) approach - faster for small periods."""
  n = len(high)
  aroon_up = np.full(n, np.nan)
  aroon_down = np.full(n, np.nan)
  aroon_osc = np.full(n, np.nan)

  if n < period + 1:
    return aroon_up, aroon_down, aroon_osc

  for i in range(period, n):
    # Find position of highest high in lookback period
    highest_idx = i - period
    highest_val = high[i - period]
    for j in range(i - period + 1, i + 1):
      if high[j] >= highest_val:
        highest_val = high[j]
        highest_idx = j

    # Find position of lowest low in lookback period
    lowest_idx = i - period
    lowest_val = low[i - period]
    for j in range(i - period + 1, i + 1):
      if low[j] <= lowest_val:
        lowest_val = low[j]
        lowest_idx = j

    # Calculate AROON values
    periods_since_high = i - highest_idx
    periods_since_low = i - lowest_idx

    aroon_up[i] = 100.0 * (period - periods_since_high) / period
    aroon_down[i] = 100.0 * (period - periods_since_low) / period
    aroon_osc[i] = aroon_up[i] - aroon_down[i]

  return aroon_up, aroon_down, aroon_osc


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def _aroon_deque(  # noqa: PLR0914
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """AROON using O(n) monotonic deque - faster for large periods."""
  n = len(high)
  aroon_up = np.full(n, np.nan)
  aroon_down = np.full(n, np.nan)
  aroon_osc = np.full(n, np.nan)

  if n < period + 1:
    return aroon_up, aroon_down, aroon_osc

  # Allocate deque buffers
  capacity = period + 2
  dq_high = np.zeros(capacity, dtype=np.int64)
  dq_low = np.zeros(capacity, dtype=np.int64)
  h_head, h_tail = 0, 0
  l_head, l_tail = 0, 0

  for i in range(n):
    min_valid_idx = i - period

    # Update high deque (max)
    h_head = deque_expire(dq_high, h_head, h_tail, capacity, min_valid_idx)
    h_head, h_tail = deque_push_max(dq_high, h_head, h_tail, capacity, high, i)

    # Update low deque (min)
    l_head = deque_expire(dq_low, l_head, l_tail, capacity, min_valid_idx)
    l_head, l_tail = deque_push_min(dq_low, l_head, l_tail, capacity, low, i)

    if i >= period:
      highest_idx = deque_front(dq_high, h_head, capacity)
      lowest_idx = deque_front(dq_low, l_head, capacity)

      periods_since_high = i - highest_idx
      periods_since_low = i - lowest_idx

      aroon_up[i] = 100.0 * (period - periods_since_high) / period
      aroon_down[i] = 100.0 * (period - periods_since_low) / period
      aroon_osc[i] = aroon_up[i] - aroon_down[i]

  return aroon_up, aroon_down, aroon_osc


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_aroon_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled AROON calculation with hybrid optimization.

  Uses simple loop for period < 25 (faster for small windows).
  Uses monotonic deque for period >= 25 (O(n) instead of O(n*period)).

  Args:
    high: Array of high prices
    low: Array of low prices
    period: Lookback period (typically 25)

  Returns:
    Tuple of (aroon_up, aroon_down, aroon_osc) arrays
  """
  if period < DEQUE_THRESHOLD:
    return _aroon_simple(high, low, period)
  return _aroon_deque(high, low, period)
