"""Numba-optimized Williams %R calculation.

This module contains JIT-compiled functions for Williams %R calculation.
Uses parallel chunked lazy rescan for optimal performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_willr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled Williams %R using parallel chunked lazy rescan.

  This algorithm splits the data into chunks to utilize multi-core processing
  while maintaining the O(N) efficiency of the lazy rescan algorithm.
  """
  n = len(close)
  willr = np.full(n, np.nan)

  if n < period:
    return willr

  # Use parallel chunks logic
  # Start valid index: period - 1
  # Items before period-1 are NaN (already initialized)
  start_v = period - 1
  total_len = n - start_v

  num_chunks = 16

  # Adaptive parallelism: avoid overhead for small arrays
  if total_len < 1024:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = start_v + (c + 1) * chunk_size

    if c == num_chunks:
      idx_end = n

    if idx_start >= n:
      continue

    # Init trackers for this chunk
    h_idx, l_idx = -1, -1
    h_val, l_val = -np.inf, np.inf

    # Initialize window min/max at start of chunk
    # Pre-scan the full window ending at idx_start to establish state
    for k in range(idx_start - period + 1, idx_start + 1):
      if high[k] >= h_val:
        h_val = high[k]
        h_idx = k
      if low[k] <= l_val:
        l_val = low[k]
        l_idx = k

    # Process chunk
    for i in range(idx_start, idx_end):
      trailing = i - period + 1

      # High update
      if h_idx < trailing:
        # Max is out of window, must rescan
        h_idx, h_val = trailing, high[trailing]
        for k in range(trailing + 1, i + 1):
          if high[k] >= h_val:
            h_val = high[k]
            h_idx = k
      elif high[i] >= h_val:
        # New high is higher than current max
        h_val = high[i]
        h_idx = i

      # Low update
      if l_idx < trailing:
        # Min is out of window, must rescan
        l_idx, l_val = trailing, low[trailing]
        for k in range(trailing + 1, i + 1):
          if low[k] <= l_val:
            l_val = low[k]
            l_idx = k
      elif low[i] <= l_val:
        # New low is lower than current min
        l_val = low[i]
        l_idx = i

      div = h_val - l_val
      if div > EPSILON:
        willr[i] = -100.0 * (h_val - close[i]) / div
      else:
        willr[i] = 0.0

  return willr
