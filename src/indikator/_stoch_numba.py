"""Numba-optimized Stochastic Oscillator calculation.

This module contains JIT-compiled functions for Stochastic calculation.
Uses hybrid approach: simple loop for k_period < 25, deque for k_period >= 25.
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
DEQUE_THRESHOLD = 25  # Use deque for k_period >= this


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def _stoch_simple(  # noqa: C901, PLR0913, PLR0917
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  k_period: int,
  k_slowing: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Stochastic using simple O(n*period) approach - faster for small periods."""
  n = len(close)
  stoch_k = np.full(n, np.nan)
  stoch_d = np.full(n, np.nan)

  if n < k_period:
    return stoch_k, stoch_d

  # First, calculate raw stochastic (before slowing)
  raw_stoch = np.full(n, np.nan)

  for i in range(k_period - 1, n):
    highest_high = high[i - k_period + 1]
    lowest_low = low[i - k_period + 1]
    for j in range(i - k_period + 2, i + 1):
      highest_high = max(highest_high, high[j])
      lowest_low = min(lowest_low, low[j])

    range_hl = highest_high - lowest_low
    if range_hl > EPSILON:
      raw_stoch[i] = 100.0 * (close[i] - lowest_low) / range_hl
    else:
      raw_stoch[i] = 50.0  # Neutral if no range

  # Apply slowing (SMA of raw stochastic)
  if n < k_period + k_slowing - 1:
    return stoch_k, stoch_d

  for i in range(k_period + k_slowing - 2, n):
    k_sum = 0.0
    for j in range(i - k_slowing + 1, i + 1):
      k_sum += raw_stoch[j]
    stoch_k[i] = k_sum / k_slowing

  # Calculate %D (SMA of %K)
  if n < k_period + k_slowing + d_period - 2:
    return stoch_k, stoch_d

  for i in range(k_period + k_slowing + d_period - 3, n):
    d_sum = 0.0
    for j in range(i - d_period + 1, i + 1):
      d_sum += stoch_k[j]
    stoch_d[i] = d_sum / d_period

  return stoch_k, stoch_d


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def _stoch_deque(  # noqa: C901, PLR0913, PLR0914, PLR0917
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  k_period: int,
  k_slowing: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Stochastic using O(n) monotonic deque - faster for large periods."""
  n = len(close)
  stoch_k = np.full(n, np.nan)
  stoch_d = np.full(n, np.nan)

  if n < k_period:
    return stoch_k, stoch_d

  # Allocate deque buffers
  capacity = k_period + 1
  dq_high = np.zeros(capacity, dtype=np.int64)
  dq_low = np.zeros(capacity, dtype=np.int64)
  h_head, h_tail = 0, 0
  l_head, l_tail = 0, 0

  # Calculate raw stochastic using deques
  raw_stoch = np.full(n, np.nan)

  for i in range(n):
    min_valid_idx = i - k_period + 1

    # Update high deque (max)
    h_head = deque_expire(dq_high, h_head, h_tail, capacity, min_valid_idx)
    h_head, h_tail = deque_push_max(dq_high, h_head, h_tail, capacity, high, i)

    # Update low deque (min)
    l_head = deque_expire(dq_low, l_head, l_tail, capacity, min_valid_idx)
    l_head, l_tail = deque_push_min(dq_low, l_head, l_tail, capacity, low, i)

    if i >= k_period - 1:
      highest_high = high[deque_front(dq_high, h_head, capacity)]
      lowest_low = low[deque_front(dq_low, l_head, capacity)]

      range_hl = highest_high - lowest_low
      if range_hl > EPSILON:
        raw_stoch[i] = 100.0 * (close[i] - lowest_low) / range_hl
      else:
        raw_stoch[i] = 50.0

  # Apply slowing (SMA of raw stochastic) - use rolling sum
  if n < k_period + k_slowing - 1:
    return stoch_k, stoch_d

  k_sum = 0.0
  for j in range(k_period - 1, k_period + k_slowing - 1):
    k_sum += raw_stoch[j]
  stoch_k[k_period + k_slowing - 2] = k_sum / k_slowing

  for i in range(k_period + k_slowing - 1, n):
    k_sum = k_sum - raw_stoch[i - k_slowing] + raw_stoch[i]
    stoch_k[i] = k_sum / k_slowing

  # Calculate %D (SMA of %K) - use rolling sum
  if n < k_period + k_slowing + d_period - 2:
    return stoch_k, stoch_d

  first_d_idx = k_period + k_slowing + d_period - 3
  d_sum = 0.0
  for j in range(first_d_idx - d_period + 1, first_d_idx + 1):
    d_sum += stoch_k[j]
  stoch_d[first_d_idx] = d_sum / d_period

  for i in range(first_d_idx + 1, n):
    d_sum = d_sum - stoch_k[i - d_period] + stoch_k[i]
    stoch_d[i] = d_sum / d_period

  return stoch_k, stoch_d


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_stoch_numba(  # noqa: PLR0913, PLR0917
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  k_period: int,
  k_slowing: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled Stochastic Oscillator with hybrid optimization.

  Uses simple loop for k_period < 25 (faster for small windows).
  Uses monotonic deque for k_period >= 25 (O(n) instead of O(n*period)).

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of closing prices
    k_period: Period for highest high / lowest low (typically 14)
    k_slowing: Slowing period for %K (typically 3)
    d_period: Period for %D smoothing (typically 3)

  Returns:
    Tuple of (%K values, %D values)
  """
  if k_period < DEQUE_THRESHOLD:
    return _stoch_simple(high, low, close, k_period, k_slowing, d_period)
  return _stoch_deque(high, low, close, k_period, k_slowing, d_period)
