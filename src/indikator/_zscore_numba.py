"""Numba-optimized Z-Score calculation.

This module contains JIT-compiled functions for Z-Score calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_zscore_numba(
  values: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled Z-Score calculation.

  Z = (Price - Mean) / StdDev

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

  # We can reuse compute_sma / std logic or implement rolling loop
  # Rolling scan

  # Initialize circular buffer
  circ_buffer = np.zeros(period, dtype=np.float64)
  idx = 0

  # Fill initial and compute
  # Standard deviation requires sum_sq and sum

  # Using Welford's algorithm or simple sum/sq_sum for window
  # Simple is faster for small windows, but precision issues?
  # For period=20, simple sum is fine.

  # Actually, straightforward implementation:
  # Maintain sum and sum_sq in window.

  current_sum = 0.0
  current_sq_sum = 0.0

  for i in range(period - 1):
    val = values[i]
    circ_buffer[i] = val
    current_sum += val
    current_sq_sum += val * val
    zscore[i] = 0.0  # Not enough data

  idx = period - 1  # Next index to overwrite in buffer (circular)
  # But circular buffer usually indexes 0..period-1.
  # Let's say buffer[idx] is oldest.
  # We fill 0..period-2.
  # Next i=period-1.

  # Let's use simple loop for implementation safety
  for i in range(period - 1, n):
    val = values[i]

    # We need to add val to sums, and remove oldest (if i >= period)
    # But first iteration i=period-1: we have period-1 items. Add 1. window total is period.
    # Wait, circular buffer logic:
    # We store 'period' items.

    # Let's restart logic.
    # sum of window [i-period+1 : i+1]

    # Just calculating slice sum is O(period) per point.
    # O(N*period). For N=1M, period=20 -> 20M ops. Fast enough.
    # Rolling update is O(N).

    # Numba loop with slice is fast? Numba doesn't like slices in loops sometimes.
    # Manual sliding window sum.

    # Re-calc sum for stability
    window_sum = 0.0
    window_sq_sum = 0.0
    for j in range(i - period + 1, i + 1):
      v = values[j]
      window_sum += v
      window_sq_sum += v * v

    mean = window_sum / period
    variance = (window_sq_sum / period) - (mean * mean)

    if variance <= EPSILON:
      std = 0.0
      z = 0.0
    else:
      std = np.sqrt(variance)
      z = (val - mean) / std

    zscore[i] = z

  return zscore
