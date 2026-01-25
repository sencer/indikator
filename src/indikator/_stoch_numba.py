"""Numba-optimized Stochastic Oscillator calculation.

This module contains JIT-compiled functions for Stochastic calculation.
Uses monotonic deque for raw calculation followed by sequential SMA smoothing.
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

EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_stoch_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  k_period: int,
  k_slowing: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled Stochastic Oscillator (Sequential Optimized)."""
  n = len(close)

  if n < k_period:
    return np.full(n, np.nan), np.full(n, np.nan)

  # 1. Raw Stochastic K
  raw_stoch = np.full(n, np.nan, dtype=np.float64)

  # Allocate deque buffers
  capacity = k_period + 2
  dq_high = np.zeros(capacity, dtype=np.int64)
  dq_low = np.zeros(capacity, dtype=np.int64)
  h_head, h_tail = 0, 0
  l_head, l_tail = 0, 0

  for i in range(n):
    min_idx = i - k_period + 1
    h_head = deque_expire(dq_high, h_head, h_tail, capacity, min_idx)
    h_head, h_tail = deque_push_max(dq_high, h_head, h_tail, capacity, high, i)
    l_head = deque_expire(dq_low, l_head, l_tail, capacity, min_idx)
    l_head, l_tail = deque_push_min(dq_low, l_head, l_tail, capacity, low, i)

    if i >= k_period - 1:
      hh = high[deque_front(dq_high, h_head, capacity)]
      ll = low[deque_front(dq_low, l_head, capacity)]
      div = hh - ll
      if div > EPSILON:
        raw_stoch[i] = 100.0 * (close[i] - ll) / div
      else:
        raw_stoch[i] = 50.0

  # 2. %K = SMA(raw_stoch, k_slowing)
  stoch_k = np.full(n, np.nan, dtype=np.float64)
  start_k = k_period - 1 + k_slowing - 1
  if n > start_k:
    current_sum = 0.0
    for i in range(k_period - 1, k_period - 1 + k_slowing):
      current_sum += raw_stoch[i]

    inv_k = 1.0 / k_slowing
    stoch_k[start_k] = current_sum * inv_k

    for i in range(start_k + 1, n):
      current_sum = current_sum + raw_stoch[i] - raw_stoch[i - k_slowing]
      stoch_k[i] = current_sum * inv_k

  # 3. %D = SMA(stoch_k, d_period)
  stoch_d = np.full(n, np.nan, dtype=np.float64)
  start_d = start_k + d_period - 1
  if n > start_d:
    current_sum = 0.0
    for i in range(start_k, start_k + d_period):
      current_sum += stoch_k[i]

    inv_d = 1.0 / d_period
    stoch_d[start_d] = current_sum * inv_d

    for i in range(start_d + 1, n):
      current_sum = current_sum + stoch_k[i] - stoch_k[i - d_period]
      stoch_d[i] = current_sum * inv_d

  return stoch_k, stoch_d
