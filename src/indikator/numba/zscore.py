"""Numba-optimized Z-Score calculation.

This module contains JIT-compiled functions for Z-Score calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_zscore_numba(
  values: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled Z-Score calculation with O(1) rolling update.

  Z = (Price - Mean) / StdDev

  Uses O(1) rolling sum/sum_sq updates instead of O(period) recalculation.

  Args:
    values: Array of values
    period: Rolling window size

  Returns:
    Array of Z-Score values
  """
  n = len(values)
  zscore = np.zeros(n, dtype=np.float64)

  if n < period:
    return zscore

  inv_period = 1.0 / period

  # Initialize sums for first window [0, period-1]
  current_sum = 0.0
  current_sq_sum = 0.0

  for i in range(period - 1):
    val = values[i]
    current_sum += val
    current_sq_sum += val * val
    zscore[i] = 0.0  # Not enough data

  # First valid zscore at index period-1
  val = values[period - 1]
  current_sum += val
  current_sq_sum += val * val

  mean = current_sum * inv_period
  variance = current_sq_sum * inv_period - mean * mean
  if variance <= EPSILON:
    zscore[period - 1] = 0.0
  else:
    zscore[period - 1] = (val - mean) / np.sqrt(variance)

  # Main loop with O(1) rolling update
  for i in range(period, n):
    old_val = values[i - period]
    new_val = values[i]

    # O(1) update: subtract old, add new
    current_sum = current_sum - old_val + new_val
    current_sq_sum = current_sq_sum - old_val * old_val + new_val * new_val

    mean = current_sum * inv_period
    variance = current_sq_sum * inv_period - mean * mean

    if variance <= EPSILON:
      zscore[i] = 0.0
    else:
      zscore[i] = (new_val - mean) / np.sqrt(variance)

  return zscore
