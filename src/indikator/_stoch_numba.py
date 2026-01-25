"""Numba-optimized Stochastic Oscillator calculation.

This module contains JIT-compiled functions for Stochastic calculation.
Uses Parallel Chunked Lazy Rescan for optimal performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange  # type: ignore[import-untyped]
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
  """Numba JIT-compiled Stochastic Oscillator (Parallel Chunked)."""
  n = len(close)

  stoch_k = np.empty(n, dtype=np.float64)
  stoch_d = np.empty(n, dtype=np.float64)

  if n < k_period:
    stoch_k[:] = np.nan
    stoch_d[:] = np.nan
    return stoch_k, stoch_d

  # 1. Raw Stoch with Parallel Chunked Lazy Rescan
  raw_stoch = np.empty(n, dtype=np.float64)
  raw_stoch[:] = np.nan

  start_v = k_period - 1
  total_len = n - start_v

  num_chunks = 16
  if total_len < 4096:
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
            l_val = low[k]
            l_idx = k
      elif low[i] <= l_val:
        l_val, l_idx = low[i], i

      div = h_val - l_val
      if div > EPSILON:
        raw_stoch[i] = 100.0 * (close[i] - l_val) / div
      else:
        raw_stoch[i] = 50.0

  # 2. %K = SMA(raw_stoch, k_slowing) - Parallel Chunked
  stoch_k[:] = np.nan
  start_k_calc = k_period - 1 + k_slowing - 1

  if n > start_k_calc:
    len_k = n - start_k_calc
    num_chunks_k = 16
    if len_k < 4096:
      num_chunks_k = 1

    chunk_size_k = len_k // num_chunks_k
    if chunk_size_k < 1:
      chunk_size_k = len_k
      num_chunks_k = 1

    inv_k = 1.0 / k_slowing

    for c in prange(num_chunks_k + 1):
      idx_start = start_k_calc + c * chunk_size_k
      idx_end = start_k_calc + (c + 1) * chunk_size_k
      if c == num_chunks_k:
        idx_end = n
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
  stoch_d[:] = np.nan
  start_d_calc = start_k_calc + d_period - 1

  if n > start_d_calc:
    len_d = n - start_d_calc
    num_chunks_d = 16
    if len_d < 4096:
      num_chunks_d = 1

    chunk_size_d = len_d // num_chunks_d
    if chunk_size_d < 1:
      chunk_size_d = len_d
      num_chunks_d = 1

    inv_d = 1.0 / d_period

    for c in prange(num_chunks_d + 1):
      idx_start = start_d_calc + c * chunk_size_d
      idx_end = start_d_calc + (c + 1) * chunk_size_d
      if c == num_chunks_d:
        idx_end = n
      if idx_start >= n:
        continue

      curr_sum = 0.0
      for k in range(idx_start - d_period + 1, idx_start + 1):
        curr_sum += stoch_k[k]

      stoch_d[idx_start] = curr_sum * inv_d

      for i in range(idx_start + 1, idx_end):
        curr_sum = curr_sum + stoch_k[i] - stoch_k[i - d_period]
        stoch_d[i] = curr_sum * inv_d

  return stoch_k, stoch_d
