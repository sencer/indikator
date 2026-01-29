"""Numba-optimized Stochastic Oscillator calculation.

This module contains JIT-compiled functions for Stochastic calculation.
Uses Parallel Chunked Lazy Rescan for optimal performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_stoch_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  k_period: int,
  k_slowing: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled Stochastic Oscillator (Ratio of Sums)."""
  n = len(close)

  stoch_k = np.empty(n, dtype=np.float64)
  stoch_d = np.empty(n, dtype=np.float64)
  stoch_k[:] = np.nan
  stoch_d[:] = np.nan

  if n < k_period:
    return stoch_k, stoch_d

  # 1. Raw Stoch with Parallel Chunked Lazy Rescan
  raw_stoch = np.empty(n, dtype=np.float64)
  raw_stoch[:] = np.nan

  start_v = k_period - 1
  total_len = n - start_v

  num_chunks = 16 if total_len >= 4096 else 1
  chunk_size = total_len // num_chunks if num_chunks > 1 else total_len

  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = min(start_v + (c + 1) * chunk_size, n)
    if idx_start >= n:
      continue

    # Initial scan for the chunk
    h_idx, l_idx = -1, -1
    h_val, l_val = -np.inf, np.inf
    scan_start = idx_start - k_period + 1
    for k in range(scan_start, idx_start + 1):
      if high[k] >= h_val:
        h_val, h_idx = high[k], k
      if low[k] <= l_val:
        l_val, l_idx = low[k], k

    for i in range(idx_start, idx_end):
      trailing = i - k_period + 1

      if h_idx < trailing:
        h_idx, h_val = trailing, high[trailing]
        for k in range(trailing + 1, i + 1):
          if high[k] >= h_val:
            h_val, h_idx = high[k], k
      elif high[i] >= h_val:
        h_val, h_idx = high[i], i

      if l_idx < trailing:
        l_idx, l_val = trailing, low[trailing]
        for k in range(trailing + 1, i + 1):
          if low[k] <= l_val:
            l_val, l_idx = low[k], k
      elif low[i] <= l_val:
        l_val, l_idx = low[i], i

      div = h_val - l_val
      if div > EPSILON:
        raw_stoch[i] = 100.0 * (close[i] - l_val) / div
      else:
        raw_stoch[i] = 0.0

  # 2. %K = SMA(raw_stoch, k_slowing) - Parallel Chunked
  start_k_calc = start_v + k_slowing - 1

  if n > start_k_calc:
    len_k = n - start_k_calc
    num_chunks_k = 16 if len_k >= 4096 else 1
    chunk_size_k = len_k // num_chunks_k if num_chunks_k > 1 else len_k

    inv_k = 1.0 / k_slowing

    for c in prange(num_chunks_k + 1):
      idx_start = start_k_calc + c * chunk_size_k
      idx_end = min(start_k_calc + (c + 1) * chunk_size_k, n)
      if idx_start >= n:
        continue

      curr_sum = 0.0
      for k in range(idx_start - k_slowing + 1, idx_start + 1):
        curr_sum += raw_stoch[k]

      stoch_k[idx_start] = curr_sum * inv_k

      for i in range(idx_start + 1, idx_end):
        curr_sum = curr_sum + raw_stoch[i] - raw_stoch[i - k_slowing]
        stoch_k[i] = curr_sum * inv_k

  # 3. %D = SMA(stoch_k, d_period) - Parallel Chunked
  start_d_calc = start_k_calc + d_period - 1

  if n > start_d_calc:
    len_d = n - start_d_calc
    num_chunks_d = 16 if len_d >= 4096 else 1
    chunk_size_d = len_d // num_chunks_d if num_chunks_d > 1 else len_d

    inv_d = 1.0 / d_period

    for c in prange(num_chunks_d + 1):
      idx_start = start_d_calc + c * chunk_size_d
      idx_end = min(start_d_calc + (c + 1) * chunk_size_d, n)
      if idx_start >= n:
        continue

      curr_sum = 0.0
      for k in range(idx_start - d_period + 1, idx_start + 1):
        curr_sum += stoch_k[k]

      stoch_d[idx_start] = curr_sum * inv_d

      for i in range(idx_start + 1, idx_end):
        curr_sum = curr_sum + stoch_k[i] - stoch_k[i - d_period]
        stoch_d[i] = curr_sum * inv_d

  # 4. Final alignment of NaNs to match TA-Lib behavior
  # Both outputs should be NaN until the total lookback is reached.
  max_lookback = start_d_calc
  if max_lookback < n:
    stoch_k[:max_lookback] = np.nan
    stoch_d[:max_lookback] = np.nan
  else:
    stoch_k[:] = np.nan
    stoch_d[:] = np.nan

  return stoch_k, stoch_d
