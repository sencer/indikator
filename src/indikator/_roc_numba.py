"""Numba-optimized ROC (Rate of Change) calculation.

This module contains JIT-compiled functions for ROC calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_roc_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROC calculation.

  ROC = ((Price - Price_n_periods_ago) / Price_n_periods_ago) * 100

  Args:
    prices: Array of prices (typically closing prices)
    period: Lookback period (typically 10)

  Returns:
    Array of ROC values (percentage)
  """
  n = len(prices)
  roc = np.full(n, np.nan)

  if n <= period:
    return roc

  for i in range(period, n):
    prev_price = prices[i - period]
    if abs(prev_price) > 1e-10:
      roc[i] = ((prices[i] - prev_price) / prev_price) * 100.0
    else:
      roc[i] = 0.0

  return roc
