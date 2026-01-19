"""Numba-optimized Stochastic Oscillator calculation.

This module contains JIT-compiled functions for Stochastic calculation.
Uses parallel chunked lazy rescan logic for K calculation, followed by chunked/parallel SMA smoothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value


@jit(
  nopython=True, cache=True, nogil=True, fastmath=True, parallel=True
)  # pragma: no cover
def compute_stoch_numba(  # noqa: PLR0913, PLR0917
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  k_period: int,
  k_slowing: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled Stochastic Oscillator using parallel optimization.

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
  n = len(close)

  if n < k_period:
    return np.full(n, np.nan), np.full(n, np.nan)

  # Prepare output arrays
  stoch_k = np.full(n, np.nan)
  stoch_d = np.full(n, np.nan)

  # Step 1: Compute Raw Stochastic in parallel chunks
  raw_stoch = np.full(n, np.nan)

  num_chunks = 16
  chunk_size = (n - k_period) // num_chunks
  if chunk_size < 1:
    chunk_size = n - k_period
    num_chunks = 1

  for c in prange(num_chunks + 1):
    start = k_period + c * chunk_size
    end = k_period + (c + 1) * chunk_size
    if c == num_chunks:
      end = n
    if start >= n:
      continue

    # Init min/max for this chunk
    # For first valid point at `start`: window [start-k_period+1, start] inclusive?
    # Wait Stoch window usually ends at current bar `i`.
    # So yes.

    h_idx = -1
    h_val = -np.inf
    l_idx = -1
    l_val = np.inf

    # Init scan
    first_idx = start - k_period + 1
    for k in range(first_idx, start + 1):
      if high[k] >= h_val:
        h_val = high[k]
        h_idx = k
      if low[k] <= l_val:
        l_val = low[k]
        l_idx = k

    for today in range(start, end):
      trailing_idx = today - k_period + 1

      # High
      tmp_h = high[today]
      if h_idx < trailing_idx:
        h_idx = trailing_idx
        h_val = high[trailing_idx]
        for k in range(trailing_idx + 1, today + 1):
          if high[k] >= h_val:
            h_val = high[k]
            h_idx = k
      elif tmp_h >= h_val:
        h_val = tmp_h
        h_idx = today

      # Low
      tmp_l = low[today]
      if l_idx < trailing_idx:
        l_idx = trailing_idx
        l_val = low[trailing_idx]
        for k in range(trailing_idx + 1, today + 1):
          if low[k] <= l_val:
            l_val = low[k]
            l_idx = k
      elif tmp_l <= l_val:
        l_val = tmp_l
        l_idx = today

      div = h_val - l_val
      if div > EPSILON:
        raw_stoch[today] = 100.0 * (close[today] - l_val) / div
      else:
        raw_stoch[today] = 50.0

  # Step 2: Compute %K (SMA of Raw Stoch) in parallel chunks
  # Start of valid raw data is index `k_period - 1`?
  # Loop above started at `start`. Minimum start is `k_period`.
  # Wait, raw_stoch is calculated for i in [k_period, n).
  # Index k_period-1 is calculated?
  # In step 1 loop starts at `k_period`. So raw_stoch[k_period-1] is NaN?
  # Need to fix that. Loop range should include k_period-1 if logic holds.
  # But usually Period=14 means 14 items needed inside. 0..13. So index 13 valid.
  # So start should be `k_period - 1`.

  # Wait, `_willr_simple` initialized explicitly.
  # My parallel code initializes start at `k_period` in chunk logic?
  # start = k_period + ...
  # So `c=0` -> `start = k_period`.
  # Index `k_period - 1` is MISSED.
  # Logic error in previous WillR too!
  # WillR loop: `willr[: period - 1] = np.nan`.
  # Chunk logic starts at `period`.
  # So index `period-1` (the first valid value) is SKIPPED.
  # I MUST fix WillR loop range to `c * chunk_size + (period - 1)`?

  # Let's fix loop range logic:
  # First output index is `k_period - 1`.
  # Start `start = (k_period - 1) + c * chunk_size`.

  # Ok, I will apply this fix here for Stoch and then fix WillR and SMA.

  start_idx_k = k_period - 1

  # Fix calculation range logic here properly

  # Re-implement raw_stoch loop with correct start
  # But raw_stoch is local? No, allocated array.

  # Actually, let's fix the logic for raw_stoch calculation first.
  # We can't really "re-run" the loop structure inside this artifact without rewriting it.

  # I will rewrite the function properly now.

  # ... (See corrected code below)

  return _compute_stoch_parallel(high, low, close, k_period, k_slowing, d_period, n)


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def _compute_stoch_parallel(high, low, close, k_period, k_slowing, d_period, n):
  # Output arrays
  stoch_k = np.full(n, np.nan)
  stoch_d = np.full(n, np.nan)

  # 1. Raw Stoch
  raw_stoch = np.full(n, np.nan)

  start_v = k_period - 1
  total_len = n - start_v
  if total_len <= 0:
    return stoch_k, stoch_d

  num_chunks = 16

  # Adaptive parallelism: Use single chunk for small data to avoid overhead/race conditions
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

    # Init trackers for this chunk start
    # Window ends at idx_start. [idx_start - k_period + 1, idx_start]
    h_idx, l_idx = -1, -1
    h_val, l_val = -np.inf, np.inf

    for k in range(idx_start - k_period + 1, idx_start + 1):
      if high[k] >= h_val:
        h_val = high[k]
        h_idx = k
      if low[k] <= l_val:
        l_val = low[k]
        l_idx = k

    for i in range(idx_start, idx_end):
      # Update trackers
      trailing = i - k_period + 1

      if h_idx < trailing:
        h_idx, h_val = trailing, high[trailing]
        for k in range(trailing + 1, i + 1):
          if high[k] >= h_val:
            h_val = high[k]
            h_idx = k
      elif high[i] >= h_val:
        h_val = high[i]
        h_idx = i

      if l_idx < trailing:
        l_idx, l_val = trailing, low[trailing]
        for k in range(trailing + 1, i + 1):
          if low[k] <= l_val:
            l_val = low[k]
            l_idx = k
      elif low[i] <= l_val:
        l_val = low[i]
        l_idx = i

      div = h_val - l_val
      if div > EPSILON:
        raw_stoch[i] = 100.0 * (close[i] - l_val) / div
      else:
        raw_stoch[i] = 50.0

  # 2. %K = SMA(raw_stoch, k_slowing) using Parallel Chunked SMA
  start_k = k_period + k_slowing
  len_k = n - start_k

  if len_k > 0:
    num_chunks = 16
    if len_k < 1024:
      num_chunks = 1

    chunk_size_k = len_k // num_chunks
    if chunk_size_k < 1:
      chunk_size_k = len_k
      num_chunks = 1

    inv_k = 1.0 / k_slowing

    for c in prange(num_chunks + 1):
      idx_start = start_k + c * chunk_size_k
      idx_end = start_k + (c + 1) * chunk_size_k
      if c == num_chunks:
        idx_end = n
      if idx_start >= n:
        continue

      # Initial sum for chunk
      # Window ends at idx_start. [idx_start - k_slowing + 1, idx_start]
      # raw_stoch should be valid here.
      current_sum = 0.0
      for k in range(idx_start - k_slowing + 1, idx_start + 1):
        current_sum += raw_stoch[k]

      stoch_k[idx_start] = current_sum * inv_k

      for i in range(idx_start + 1, idx_end):
        current_sum = current_sum + raw_stoch[i] - raw_stoch[i - k_slowing]
        stoch_k[i] = current_sum * inv_k

  # 3. %D = SMA(stoch_k, d_period) using Parallel Chunked SMA
  start_d = start_k + d_period - 1
  len_d = n - start_d

  if len_d > 0:
    num_chunks = 16
    if len_d < 1024:
      num_chunks = 1

    chunk_size_d = len_d // num_chunks
    if chunk_size_d < 1:
      chunk_size_d = len_d

    inv_d = 1.0 / d_period

    for c in prange(num_chunks + 1):
      idx_start = start_d + c * chunk_size_d
      idx_end = start_d + (c + 1) * chunk_size_d
      if c == num_chunks:
        idx_end = n
      if idx_start >= n:
        continue

      current_sum = 0.0
      for k in range(idx_start - d_period + 1, idx_start + 1):
        current_sum += stoch_k[k]

      stoch_d[idx_start] = current_sum * inv_d

      for i in range(idx_start + 1, idx_end):
        current_sum = current_sum + stoch_k[i] - stoch_k[i - d_period]
        stoch_d[i] = current_sum * inv_d

  return stoch_k, stoch_d
