"""Numba-optimized Stochastic Oscillator calculation.

This module contains JIT-compiled functions for Stochastic calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_stoch_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  k_period: int,
  k_slowing: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled Stochastic Oscillator calculation.

  %K = 100 * SMA((Close - Lowest Low) / (Highest High - Lowest Low), k_slowing)
  %D = SMA(%K, d_period)

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
  stoch_k = np.full(n, np.nan)
  stoch_d = np.full(n, np.nan)

  if n < k_period:
    return stoch_k, stoch_d

  # First, calculate raw stochastic (before slowing)
  raw_stoch = np.full(n, np.nan)

  for i in range(k_period - 1, n):
    highest_high = high[i - k_period + 1]
    lowest_low = low[i - k_period + 1]
    for j in range(i - k_period + 2, i + 1):
      highest_high = max(highest_high, high[j])
      lowest_low = min(lowest_low, low[j])

    range_hl = highest_high - lowest_low
    if range_hl > 1e-10:
      raw_stoch[i] = 100.0 * (close[i] - lowest_low) / range_hl
    else:
      raw_stoch[i] = 50.0  # Neutral if no range

  # Apply slowing (SMA of raw stochastic)
  if n < k_period + k_slowing - 1:
    return stoch_k, stoch_d

  for i in range(k_period + k_slowing - 2, n):
    k_sum = 0.0
    for j in range(i - k_slowing + 1, i + 1):
      k_sum += raw_stoch[j]
    stoch_k[i] = k_sum / k_slowing

  # Calculate %D (SMA of %K)
  if n < k_period + k_slowing + d_period - 2:
    return stoch_k, stoch_d

  for i in range(k_period + k_slowing + d_period - 3, n):
    d_sum = 0.0
    for j in range(i - d_period + 1, i + 1):
      d_sum += stoch_k[j]
    stoch_d[i] = d_sum / d_period

  return stoch_k, stoch_d
