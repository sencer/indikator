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


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_cmo_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled CMO calculation using Wilder's smoothing.

  CMO measures momentum as the difference between smoothed gains and losses
  divided by their sum, oscillating between -100 and +100.

  Uses Wilder's smoothing (same as RSI) for the gain/loss averages.

  Formula:
  CMO = 100 * (smoothed_gains - smoothed_losses) / (smoothed_gains + smoothed_losses)

  Args:
    prices: Array of prices (typically closing prices)
    period: Lookback period (typically 14)

  Returns:
    Array of CMO values (-100 to +100 range)
  """
  n = len(prices)
  cmo = np.full(n, np.nan)

  if n < period + 1:
    return cmo

  # Calculate price changes
  gains = np.zeros(n)
  losses = np.zeros(n)

  for i in range(1, n):
    change = prices[i] - prices[i - 1]
    if change > 0:
      gains[i] = change
    else:
      losses[i] = -change

  # Calculate initial averages (simple average of first period)
  sum_gains = 0.0
  sum_losses = 0.0

  for i in range(1, period + 1):
    sum_gains += gains[i]
    sum_losses += losses[i]

  avg_gain = sum_gains / period
  avg_loss = sum_losses / period

  # First CMO
  total = avg_gain + avg_loss
  if total > 1e-10:
    cmo[period] = 100.0 * (avg_gain - avg_loss) / total
  else:
    cmo[period] = 0.0

  # Subsequent CMO values using Wilder's smoothing
  for i in range(period + 1, n):
    avg_gain = (avg_gain * (period - 1) + gains[i]) / period
    avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    total = avg_gain + avg_loss
    if total > 1e-10:
      cmo[i] = 100.0 * (avg_gain - avg_loss) / total
    else:
      cmo[i] = 0.0

  return cmo
