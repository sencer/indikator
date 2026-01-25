"""Numba-optimized AROON indicator calculation.

This module contains JIT-compiled functions for AROON calculation.
Uses parallel chunked lazy rescan for optimal performance (O(N) work, O(N/C) latency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Constants
EPSILON = 1e-14


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_aroon_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled AROON using parallel chunked lazy rescan.

  Splits the loop into chunks to utilize multi-core CPUs.
  """
  n = len(high)

  aroon_up = np.full(n, np.nan, dtype=np.float64)
  aroon_down = np.full(n, np.nan, dtype=np.float64)
  aroon_osc = np.full(n, np.nan, dtype=np.float64)

  if n < period + 1:
    return aroon_up, aroon_down, aroon_osc

  start_v = period
  total_len = n - start_v

  num_chunks = 16
  if total_len < 2048:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  inv_period = 100.0 / period

  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = start_v + (c + 1) * chunk_size

    if c == num_chunks:
      idx_end = n

    if idx_start >= n:
      continue

    # Pre-scan [idx_start - period, idx_start - 1]
    scan_start = idx_start - period
    highest_idx = -1
    highest_val = -np.inf
    lowest_idx = -1
    lowest_val = np.inf

    for k in range(scan_start, idx_start):
      if high[k] >= highest_val:
        highest_val = high[k]
        highest_idx = k
      if low[k] <= lowest_val:
        lowest_val = low[k]
        lowest_idx = k

    for i in range(idx_start, idx_end):
      trailing_idx = i - period

      # High update
      tmp_h = high[i]
      if highest_idx < trailing_idx:
        highest_val = tmp_h
        highest_idx = i
        for k in range(trailing_idx, i):
          if high[k] >= highest_val:
            highest_val = high[k]
            highest_idx = k
      elif tmp_h >= highest_val:
        highest_val = tmp_h
        highest_idx = i

      # Low update
      tmp_l = low[i]
      if lowest_idx < trailing_idx:
        lowest_val = tmp_l
        lowest_idx = i
        for k in range(trailing_idx, i):
          if low[k] <= lowest_val:
            lowest_val = low[k]
            lowest_idx = k
      elif tmp_l <= lowest_val:
        lowest_val = tmp_l
        lowest_idx = i

      aroon_up[i] = (period - (i - highest_idx)) * inv_period
      aroon_down[i] = (period - (i - lowest_idx)) * inv_period
      aroon_osc[i] = aroon_up[i] - aroon_down[i]

  return aroon_up, aroon_down, aroon_osc


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_aroonosc_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled AROONOSC using parallel chunked lazy rescan.

  Specialized to compute only the oscillator with a single allocation.
  """
  n = len(high)
  aroon_osc = np.full(n, np.nan, dtype=np.float64)

  if n < period + 1:
    return aroon_osc

  start_v = period
  total_len = n - start_v

  num_chunks = 16
  if total_len < 2048:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  inv_period = 100.0 / period

  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = start_v + (c + 1) * chunk_size

    if c == num_chunks:
      idx_end = n

    if idx_start >= n:
      continue

    # Pre-scan [idx_start - period, idx_start - 1]
    scan_start = idx_start - period
    highest_idx = -1
    highest_val = -np.inf
    lowest_idx = -1
    lowest_val = np.inf

    for k in range(scan_start, idx_start):
      if high[k] >= highest_val:
        highest_val = high[k]
        highest_idx = k
      if low[k] <= lowest_val:
        lowest_val = low[k]
        lowest_idx = k

    for i in range(idx_start, idx_end):
      trailing_idx = i - period

      # High update
      tmp_h = high[i]
      if highest_idx < trailing_idx:
        highest_val = tmp_h
        highest_idx = i
        for k in range(trailing_idx, i):
          if high[k] >= highest_val:
            highest_val = high[k]
            highest_idx = k
      elif tmp_h >= highest_val:
        highest_val = tmp_h
        highest_idx = i

      # Low update
      tmp_l = low[i]
      if lowest_idx < trailing_idx:
        lowest_val = tmp_l
        lowest_idx = i
        for k in range(trailing_idx, i):
          if low[k] <= lowest_val:
            lowest_val = low[k]
            lowest_idx = k
      elif tmp_l <= lowest_val:
        lowest_val = tmp_l
        lowest_idx = i

      # Only compute OSC to save work and memory bandwidth
      up = (period - (i - highest_idx)) * inv_period
      down = (period - (i - lowest_idx)) * inv_period
      aroon_osc[i] = up - down

  return aroon_osc
