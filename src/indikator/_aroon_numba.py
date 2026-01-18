"""Numba-optimized AROON indicator calculation.

This module contains JIT-compiled functions for AROON calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_aroon_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled AROON calculation.

  AROON measures trend strength by tracking how many periods since
  the highest high and lowest low.

  Formulas:
  Aroon Up = 100 * (period - periods_since_high) / period
  Aroon Down = 100 * (period - periods_since_low) / period
  Aroon Oscillator = Aroon Up - Aroon Down

  Args:
    high: Array of high prices
    low: Array of low prices
    period: Lookback period (typically 25)

  Returns:
    Tuple of (aroon_up, aroon_down, aroon_osc) arrays
  """
  n = len(high)
  aroon_up = np.full(n, np.nan)
  aroon_down = np.full(n, np.nan)
  aroon_osc = np.full(n, np.nan)

  if n < period + 1:
    return aroon_up, aroon_down, aroon_osc

  for i in range(period, n):
    # Find position of highest high in lookback period
    highest_idx = i - period
    highest_val = high[i - period]
    for j in range(i - period + 1, i + 1):
      if high[j] >= highest_val:
        highest_val = high[j]
        highest_idx = j

    # Find position of lowest low in lookback period
    lowest_idx = i - period
    lowest_val = low[i - period]
    for j in range(i - period + 1, i + 1):
      if low[j] <= lowest_val:
        lowest_val = low[j]
        lowest_idx = j

    # Calculate AROON values
    periods_since_high = i - highest_idx
    periods_since_low = i - lowest_idx

    aroon_up[i] = 100.0 * (period - periods_since_high) / period
    aroon_down[i] = 100.0 * (period - periods_since_low) / period
    aroon_osc[i] = aroon_up[i] - aroon_down[i]

  return aroon_up, aroon_down, aroon_osc
