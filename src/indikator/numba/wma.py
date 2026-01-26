"""Numba-optimized WMA (Weighted Moving Average) calculation.

Uses parallel chunked O(1) rolling update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_wma_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled WMA with Parallel Chunked Rolling Update.

  WMA = sum(price[i] * weight[i]) / sum(weights)
  """
  n = len(prices)
  out = np.full(n, np.nan, dtype=np.float64)

  if n < period:
    return out

  # Constants
  weight_sum = period * (period + 1) / 2.0
  inv_weight_sum = 1.0 / weight_sum

  start_v = period - 1
  total_len = n - start_v

  num_chunks = 16
  # Adaptive
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

    # Initialize rolling state for window ending at idx_start - 1
    # Range: [idx_start - period, idx_start)
    weighted_sum = 0.0
    unweighted_sum = 0.0

    start_lookback = idx_start - period

    for k_idx in range(period):
      # We need to construct WMA state ending at idx_start - 1
      # Indices in prices: start_lookback ... idx_start - 1
      # Weights: 1 ... period

      val = prices[start_lookback + k_idx]
      weight = k_idx + 1
      weighted_sum += val * weight
      unweighted_sum += val

    # Rolling Loop
    for i in range(idx_start, idx_end):
      leaving_price = prices[i - period]
      entering_price = prices[i]

      # O(1) Update
      # New WeightedSum = OldWeightedSum - OldUnweightedSum + NewPrice * Period
      # New UnweightedSum = OldUnweightedSum - LeavingPrice + EnteringPrice

      weighted_sum = weighted_sum - unweighted_sum + entering_price * period
      unweighted_sum = unweighted_sum - leaving_price + entering_price

      out[i] = weighted_sum * inv_weight_sum

  return out
