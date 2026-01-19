"""Numba-optimized ROC (Rate of Change) calculation.

This module contains JIT-compiled functions for ROC calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
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

  if n <= period:
    return np.full(n, np.nan)

  roc = np.empty(n, dtype=np.float64)

  # Fill NaN using loop (slightly faster than slice assignment in Numba)
  for i in range(period):
    roc[i] = np.nan

  for i in range(period, n):
    prev_price = prices[i - period]
    # Use simple zero check - branch predictor handles this well
    if prev_price != 0.0:
      roc[i] = ((prices[i] - prev_price) / prev_price) * 100.0
    else:
      roc[i] = 0.0

  return roc


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_rocp_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCP calculation.

  ROCP = (Price - Price_n_periods_ago) / Price_n_periods_ago
  """
  n = len(prices)

  if n <= period:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(period):
    out[i] = np.nan

  for i in range(period, n):
    prev_price = prices[i - period]
    if prev_price != 0.0:
      out[i] = (prices[i] - prev_price) / prev_price
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_rocr_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCR calculation.

  ROCR = Price / Price_n_periods_ago
  """
  n = len(prices)

  if n <= period:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(period):
    out[i] = np.nan

  for i in range(period, n):
    prev_price = prices[i - period]
    if prev_price != 0.0:
      out[i] = prices[i] / prev_price
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_rocr100_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCR100 calculation.

  ROCR100 = (Price / Price_n_periods_ago) * 100
  """
  n = len(prices)

  if n <= period:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(period):
    out[i] = np.nan

  for i in range(period, n):
    prev_price = prices[i - period]
    if prev_price != 0.0:
      out[i] = (prices[i] / prev_price) * 100.0
    else:
      out[i] = 0.0

  return out
