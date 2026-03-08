"""Numba-optimized SMA (Simple Moving Average) calculation.

This module contains JIT-compiled functions for SMA calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, parallel=True)  # pragma: no cover
def compute_sma_numba(
  prices: NDArray[np.float64],
  period: int,
  scale: float = 1.0,
  shifted: bool = False,
) -> NDArray[np.float64]:
  """Numba JIT-compiled SMA calculation.

  Args:
    prices: Input array
    period: Moving average period
    scale: Multiplier for the result (default 1.0)
    shifted: If True, shift result by 1 (out[i+1] = sma[i])
  """
  n = len(prices)
  out_len = n + 1 if shifted else n
  # If shifted, we might need output of size n (dropping last?) or n (shifting out first?)
  # _rolling_sma_prev expects output size N, where out[i] = sma[i-1].
  # So out[0] is NaN.

  if n < period:
    return np.full(n, np.nan)

  sma = np.full(n, np.nan)

  # Adaptive parallelism
  total_len = n - period
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  inv_period = (1.0 / period) * scale

  for c in prange(num_chunks + 1):
    start = (period - 1) + c * chunk_size
    end = (period - 1) + (c + 1) * chunk_size
    if c == num_chunks:
      end = n

    if start >= n:
      continue

    # Initial sum for chunk
    chunk_sum = 0.0
    for k in range(start - period + 1, start + 1):
      chunk_sum += prices[k]

    current_sum = chunk_sum

    # Branching outside the loop
    if shifted:
      # Initial write
      val = chunk_sum * inv_period
      if start + 1 < n:
        sma[start + 1] = val

      for i in range(start + 1, end):
        current_sum = current_sum + prices[i] - prices[i - period]

        if not np.isnan(current_sum):
          if i + 1 < n:
            sma[i + 1] = current_sum * inv_period
        else:
          # Recover
          valid_sum = 0.0
          is_valid = True
          for k in range(i - period + 1, i + 1):
            v = prices[k]
            if np.isnan(v):
              is_valid = False
              break
            valid_sum += v

          if is_valid:
            current_sum = valid_sum
            if i + 1 < n:
              sma[i + 1] = current_sum * inv_period
          else:
            if i + 1 < n:
              sma[i + 1] = np.nan
    else:
      # Non-shifted loop (Original optimized path)
      val = chunk_sum * inv_period
      sma[start] = val

      for i in range(start + 1, end):
        current_sum = current_sum + prices[i] - prices[i - period]

        if not np.isnan(current_sum):
          sma[i] = current_sum * inv_period
        else:
          # Recover
          valid_sum = 0.0
          is_valid = True
          for k in range(i - period + 1, i + 1):
            v = prices[k]
            if np.isnan(v):
              is_valid = False
              break
            valid_sum += v

          if is_valid:
            current_sum = valid_sum
            sma[i] = current_sum * inv_period
          else:
            sma[i] = np.nan

  return sma
