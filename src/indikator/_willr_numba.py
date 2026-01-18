"""Numba-optimized Williams %R calculation.

This module contains JIT-compiled functions for Williams %R calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_willr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled Williams %R calculation.

  %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)

  Range: -100 to 0

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of closing prices
    period: Lookback period (typically 14)

  Returns:
    Array of Williams %R values
  """
  n = len(close)
  willr = np.full(n, np.nan)

  if n < period:
    return willr

  for i in range(period - 1, n):
    highest_high = high[i - period + 1]
    lowest_low = low[i - period + 1]
    for j in range(i - period + 2, i + 1):
      highest_high = max(highest_high, high[j])
      lowest_low = min(lowest_low, low[j])

    range_hl = highest_high - lowest_low
    if range_hl > EPSILON:
      willr[i] = -100.0 * (highest_high - close[i]) / range_hl
    else:
      willr[i] = -50.0  # Neutral if no range

  return willr
