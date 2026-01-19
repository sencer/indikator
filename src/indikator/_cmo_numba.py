"""Numba-optimized CMO (Chande Momentum Oscillator) calculation.

This module contains JIT-compiled functions for CMO calculation.
CMO uses Wilder's smoothing like RSI, not simple sliding window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_cmo_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled CMO calculation using Wilder's smoothing.

  Optimized with loop fusion to remove intermediate array allocations.
  """
  n = len(prices)
  cmo = np.full(n, np.nan)

  if n < period + 1:
    return cmo

  # Initialize accumulators
  sum_gains = 0.0
  sum_losses = 0.0

  # 1. Initialization (Sum of first 'period' changes)
  # Matches TA-lib: simple sum for initial value
  # Note: indices 1 to period (inclusive)
  for i in range(1, period + 1):
    change = prices[i] - prices[i - 1]
    if change > 0:
      sum_gains += change
    else:
      # change < 0, so -change > 0
      sum_losses += -change

  avg_gain = sum_gains / period
  avg_loss = sum_losses / period

  # First CMO at index 'period'
  total = avg_gain + avg_loss
  if total > EPSILON:
    cmo[period] = 100.0 * (avg_gain - avg_loss) / total
  else:
    cmo[period] = 0.0

  # 2. Main Loop (Wilder's Smoothing)
  for i in range(period + 1, n):
    change = prices[i] - prices[i - 1]

    # Current gain/loss
    curr_gain = 0.0
    curr_loss = 0.0
    if change > 0:
      curr_gain = change
    else:
      curr_loss = -change

    # Wilder's Smoothing: (Previous * (n-1) + Current) / n
    avg_gain = (avg_gain * (period - 1) + curr_gain) / period
    avg_loss = (avg_loss * (period - 1) + curr_loss) / period

    total = avg_gain + avg_loss
    if total > EPSILON:
      cmo[i] = 100.0 * (avg_gain - avg_loss) / total
    else:
      cmo[i] = 0.0

  return cmo
