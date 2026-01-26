"""Numba-optimized ROC (Rate of Change) calculation.

This module contains JIT-compiled functions for ROC calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_roc_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROC calculation (Sequential).

  ROC = ((Price - Price_n_periods_ago) / Price_n_periods_ago) * 100
  """
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  # NaN for warmup
  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    if prev != 0.0:
      out[i] = ((prices[i] - prev) / prev) * 100.0
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocp_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCP calculation (Sequential)."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    if prev != 0.0:
      out[i] = (prices[i] - prev) / prev
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocr_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCR calculation (Sequential)."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    if prev != 0.0:
      out[i] = prices[i] / prev
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocr100_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCR100 calculation (Sequential)."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    if prev != 0.0:
      out[i] = (prices[i] / prev) * 100.0
    else:
      out[i] = 0.0

  return out
