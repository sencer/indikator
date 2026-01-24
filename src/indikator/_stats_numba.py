"""Numba-optimized statistical calculations (STDDEV, VAR)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_stddev_numba(
  data: NDArray[np.float64],
  period: int,
  nbdev: float,
) -> NDArray[np.float64]:
  """Calculate rolling parallel standard deviation.

  Uses Parallel Chunked Welford's algorithm.
  """
  n = len(data)
  out = np.full(n, np.nan, dtype=np.float64)

  if n < period or period < 2:
    return out

  start_v = period - 1
  total_len = n - start_v

  num_chunks = 16
  if total_len < 2048:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  # Parallel execution
  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = start_v + (c + 1) * chunk_size
    if c == num_chunks:
      idx_end = n
    if idx_start >= n:
      continue

    # Initialize Welford stats for window ending at idx_start - 1
    # Range: [idx_start - period, idx_start)
    mean = 0.0
    m2 = 0.0

    # We must scan the initializing window sequentially to build Welford state
    # O(Period) overhead per chunk

    # First element of the window
    start_lookback = idx_start - period

    # Welford initialization loop
    for k_idx in range(start_lookback, idx_start):
      val = data[k_idx]
      # Welford step (incremental)
      # k is 1-based count in this window
      count = k_idx - start_lookback + 1
      delta = val - mean
      mean += delta / count
      delta2 = val - mean
      m2 += delta * delta2

    # Now roll through the chunk
    for i in range(idx_start, idx_end):
      old_val = data[i - period]
      new_val = data[i]

      old_mean = mean
      mean = old_mean + (new_val - old_val) / period
      m2 = (
        m2
        - (old_val - old_mean) * (old_val - mean)
        + (new_val - old_mean) * (new_val - mean)
      )

      # FP correction
      if m2 < 0:
        m2 = 0.0

      variance = m2 / period
      out[i] = np.sqrt(variance) * nbdev

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_var_numba(
  data: NDArray[np.float64],
  period: int,
  nbdev: float,
) -> NDArray[np.float64]:
  """Calculate rolling parallel variance.

  Uses Parallel Chunked Welford's algorithm.
  """
  n = len(data)
  out = np.full(n, np.nan, dtype=np.float64)

  if n < period or period < 2:
    return out

  start_v = period - 1
  total_len = n - start_v

  num_chunks = 16
  if total_len < 2048:
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

    # Initialize Welford stats
    mean = 0.0
    m2 = 0.0
    start_lookback = idx_start - period

    for k_idx in range(start_lookback, idx_start):
      val = data[k_idx]
      count = k_idx - start_lookback + 1
      delta = val - mean
      mean += delta / count
      delta2 = val - mean
      m2 += delta * delta2

    # Roll
    for i in range(idx_start, idx_end):
      old_val = data[i - period]
      new_val = data[i]

      old_mean = mean
      mean = old_mean + (new_val - old_val) / period
      m2 = (
        m2
        - (old_val - old_mean) * (old_val - mean)
        + (new_val - old_mean) * (new_val - mean)
      )

      if m2 < 0:
        m2 = 0.0

      variance = m2 / period
      # nbdev acts as multiplier for variance too? (Usually for STDDEV)
      # TA-Lib VAR output: real = variance.
      # Indicator signature: def var(data, period, nbdev=1).
      # If nbdev is passed, we multiply? TA-Lib has no nbdev for VAR.
      # But Indikator.VAR signature seems to have it. Let's support it.
      out[i] = variance * nbdev

  return out
