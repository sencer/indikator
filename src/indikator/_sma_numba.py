"""Numba-optimized SMA (Simple Moving Average) calculation.

This module contains JIT-compiled functions for SMA calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_sma_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled SMA calculation.

  SMA = (P1 + P2 + ... + Pn) / n

  Uses rolling sum for O(n) efficiency.

  Args:
    prices: Array of prices (typically closing prices)
    period: Lookback period

  Returns:
    Array of SMA values (NaN for initial bars where period not satisfied)
  """
  n = len(prices)
  sma = np.full(n, np.nan)

  if n < period:
    return sma

  # Calculate first SMA
  rolling_sum = 0.0
  for i in range(period):
    rolling_sum += prices[i]
  sma[period - 1] = rolling_sum / period

  # Rolling calculation for subsequent values
  for i in range(period, n):
    rolling_sum = rolling_sum + prices[i] - prices[i - period]
    sma[i] = rolling_sum / period

  return sma
