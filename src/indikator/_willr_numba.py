"""Numba-optimized Williams %R calculation.

This module contains JIT-compiled functions for Williams %R calculation.
Uses parallel chunked lazy rescan for optimal performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_willr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled Williams %R using sequential lazy rescan.

  This algorithm is O(N) on average and faster than monotonic queues for
  typical window sizes due to lower constant overhead.
  """
  n = len(close)

  if n < period:
    return np.full(n, np.nan)

  willr = np.empty(n)
  willr[: period - 1] = np.nan

  start = period - 1
  end = n

  # Scan initial window
  h_idx = -1
  h_val = -np.inf
  l_idx = -1
  l_val = np.inf

  first_idx = start - period + 1
  for k in range(first_idx, start + 1):
    if high[k] >= h_val:
      h_val = high[k]
      h_idx = k
    if low[k] <= l_val:
      l_val = low[k]
      l_idx = k

  for today in range(start, end):
    trailing_idx = today - period + 1

    # Check lazily
    # Update High
    tmp_h = high[today]
    if h_idx < trailing_idx:
      # Rescan needed
      h_idx = trailing_idx
      h_val = high[trailing_idx]
      for k in range(trailing_idx + 1, today + 1):
        if high[k] >= h_val:
          h_val = high[k]
          h_idx = k
    elif tmp_h >= h_val:
      h_val = tmp_h
      h_idx = today

    # Update Low
    tmp_l = low[today]
    if l_idx < trailing_idx:
      # Rescan needed
      l_idx = trailing_idx
      l_val = low[trailing_idx]
      for k in range(trailing_idx + 1, today + 1):
        if low[k] <= l_val:
          l_val = low[k]
          l_idx = k
    elif tmp_l <= l_val:
      l_val = tmp_l
      l_idx = today

    # Calc
    div = h_val - l_val
    if div > EPSILON:
      willr[today] = -100.0 * (h_val - close[today]) / div
    else:
      willr[today] = 0.0

  return willr
