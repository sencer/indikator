"""Numba-optimized EMA (Exponential Moving Average) calculation.

This module contains JIT-compiled functions for EMA calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_ema_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled EMA calculation.

  EMA = Price(t) * k + EMA(t-1) * (1-k)
  where k = 2 / (period + 1)

  The first EMA value is the SMA of the first 'period' values.

  Note: This function assumes clean input (no NaN). Input validation
  is handled by the public API using datawarden Finite validator.

  Args:
    prices: Array of prices (must not contain NaN)
    period: Lookback period

  Returns:
    Array of EMA values (NaN for initial bars where period not satisfied)
  """
  n = len(prices)

  if n < period:
    return np.full(n, np.nan)

  ema = np.empty(n)

  # Multiplier for weighting
  k = 2.0 / (period + 1)
  k1 = 1.0 - k

  # First EMA is SMA of first 'period' values
  sma_sum = 0.0
  for i in range(period):
    sma_sum += prices[i]
    ema[i] = np.nan

  # Initialize with SMA
  prev_ema = sma_sum / period
  ema[period - 1] = prev_ema

  # Calculate subsequent EMA values using local variable (register optimized)
  for i in range(period, n):
    prev_ema = prices[i] * k + prev_ema * k1
    ema[i] = prev_ema

  return ema
