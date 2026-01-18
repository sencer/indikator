"""Numba-optimized TRIX (Triple Exponential Average) calculation.

This module contains JIT-compiled functions for TRIX calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_trix_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled TRIX calculation.

  TRIX is the percentage rate of change of a triple exponentially smoothed
  moving average. It filters out short-term noise to identify longer-term
  trend direction and reversals.

  Steps:
  1. EMA1 = EMA(prices, period)
  2. EMA2 = EMA(EMA1, period)
  3. EMA3 = EMA(EMA2, period)
  4. TRIX = (EMA3[today] - EMA3[yesterday]) / EMA3[yesterday] * 100

  Args:
    prices: Array of prices (typically closing prices)
    period: EMA period (typically 14)

  Returns:
    Array of TRIX values (percentage)
  """
  n = len(prices)
  trix = np.full(n, np.nan)

  # Need at least 3 * period - 2 bars for first valid TRIX
  min_bars = 3 * period - 2
  if n < min_bars + 1:
    return trix

  # EMA smoothing factor
  k = 2.0 / (period + 1)

  # First EMA: seeded with SMA
  ema1 = np.full(n, np.nan)
  sma1 = 0.0
  for i in range(period):
    sma1 += prices[i]
  ema1[period - 1] = sma1 / period

  for i in range(period, n):
    ema1[i] = prices[i] * k + ema1[i - 1] * (1 - k)

  # Second EMA: seeded with SMA of EMA1
  ema2 = np.full(n, np.nan)
  ema2_start = 2 * period - 2
  sma2 = 0.0
  for i in range(period - 1, 2 * period - 1):
    sma2 += ema1[i]
  ema2[ema2_start] = sma2 / period

  for i in range(ema2_start + 1, n):
    ema2[i] = ema1[i] * k + ema2[i - 1] * (1 - k)

  # Third EMA: seeded with SMA of EMA2
  ema3 = np.full(n, np.nan)
  ema3_start = 3 * period - 3
  sma3 = 0.0
  for i in range(ema2_start, 3 * period - 2):
    sma3 += ema2[i]
  ema3[ema3_start] = sma3 / period

  for i in range(ema3_start + 1, n):
    ema3[i] = ema2[i] * k + ema3[i - 1] * (1 - k)

  # TRIX: percentage rate of change of EMA3
  epsilon = 1e-10  # Minimum denominator value
  for i in range(ema3_start + 1, n):
    prev_ema3 = ema3[i - 1]
    if abs(prev_ema3) > epsilon:
      trix[i] = ((ema3[i] - prev_ema3) / prev_ema3) * 100.0
    else:
      trix[i] = 0.0

  return trix
