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
  Each chunk independently maintains its own max/min trackers using the
  lazy rescan algorithm.
  """
  n = len(high)

  # Prepare output arrays
  aroon_up = np.full(n, np.nan, dtype=np.float64)
  aroon_down = np.full(n, np.nan, dtype=np.float64)
  aroon_osc = np.full(n, np.nan, dtype=np.float64)

  if n < period + 1:
    return aroon_up, aroon_down, aroon_osc

  # AROON logic:
  # First valid index is at `period`. (Requires `period + 1` window: 0..period inclusive)
  # So start_v = period.
  start_v = period
  total_len = n - start_v

  num_chunks = 16
  # Adaptive parallelism
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

    # Init trackers
    # We need to scan the window ending at idx_start: [idx_start - period, idx_start]
    # Length is period + 1
    # Example: period=14. idx=14. Window [0, 14]. Len 15.

    scan_start = idx_start - period
    # Sanity check
    scan_start = max(scan_start, 0)

    highest_idx = -1
    highest_val = -np.inf
    lowest_idx = -1
    lowest_val = np.inf

    # Initial scan for this chunk's start state
    # We scan UP TO idx_start (exclusive of the loop below, but inclusive of the window state)
    # The first iteration of the calculation loop is `i = idx_start`.
    # At `i`, we need the max/min of [i - period, i].
    # So we simply initialize with the state just BEFORE `i`, then let the loop handle `i`.
    # Actually, easier to pre-scan the FULL window for `idx_start` and then proceed.

    # Pre-scan for i = idx_start
    for k in range(scan_start, idx_start + 1):
      if high[k] >= highest_val:
        highest_val = high[k]
        highest_idx = k
      if low[k] <= lowest_val:
        lowest_val = low[k]
        lowest_idx = k

    # Now we enter the loop.
    # CAUTION: The pre-scan already covered `idx_start`.
    # So for `i = idx_start`, we already have the answer?
    # Yes, highest_idx/lowest_idx are correct for `idx_start`.
    # But the loop below usually updates first.
    # Let's adjust: Pre-scan window [idx_start - period, idx_start].
    # Then loop i from idx_start to idx_end.
    # Inside the loop:
    # 1. Update with new value at i? No, the window ending at i *includes* i.
    # So if we pre-scanned up to idx_start, then at i=idx_start, we already processed it.
    # Standard pattern:
    #   Pre-scan [start-period, start-1]
    #   Loop i from start to end:
    #     Update with val[i]
    #     Check expiry

    # Let's adopt the standard pattern.
    # Pre-scan [idx_start - period, idx_start - 1]
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

      # Update High (Lazy Rescan)
      tmp_h = high[i]
      # Check expiry FIRST? Or AFTER?
      # Window is [i-period, i].
      # Expiry: if old max index < i-period, it is gone.

      if highest_idx < trailing_idx:
        # Expired, full rescan
        # Scan from trailing_idx to i-1 (since we haven't checked i yet)
        # Actually, simpler to scan trailing_idx to i inclusive
        highest_val = tmp_h
        highest_idx = i
        for k in range(trailing_idx, i):
          if high[k] >= highest_val:
            highest_val = high[k]
            highest_idx = k
      elif tmp_h >= highest_val:
        highest_val = tmp_h
        highest_idx = i

      # Update Low
      tmp_l = low[i]
      if lowest_idx < trailing_idx:
        # Expired
        lowest_val = tmp_l
        lowest_idx = i
        for k in range(trailing_idx, i):
          if low[k] <= lowest_val:
            lowest_val = low[k]
            lowest_idx = k
      elif tmp_l <= lowest_val:
        lowest_val = tmp_l
        lowest_idx = i

      # Calculate
      # Aroon Up = ((period - (Days Since High)) / period) * 100
      # Days Since High = i - highest_idx
      # => ((period - (i - highest_idx)) / period) * 100

      aroon_up[i] = (period - (i - highest_idx)) * inv_period
      aroon_down[i] = (period - (i - lowest_idx)) * inv_period
      aroon_osc[i] = aroon_up[i] - aroon_down[i]

  return aroon_up, aroon_down, aroon_osc
