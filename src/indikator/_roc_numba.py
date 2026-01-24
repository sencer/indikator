"""Numba-optimized ROC (Rate of Change) calculation.

This module contains JIT-compiled functions for ROC calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_roc_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROC calculation (Parallel)."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  # NaN for warmup
  out[:period] = np.nan

  # Parallel computation
  for i in range(period, n):
    prev_price = prices[i - period]
    if abs(prev_price) > 1e-12:
      out[i] = ((prices[i] - prev_price) / prev_price) * 100.0
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocp_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCP calculation (Parallel)."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  out[:period] = np.nan

  for i in range(period, n):
    prev_price = prices[i - period]
    if abs(prev_price) > 1e-12:
      out[i] = (prices[i] - prev_price) / prev_price
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocr_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCR calculation (Parallel)."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  out[:period] = np.nan

  for i in range(period, n):
    prev_price = prices[i - period]
    if abs(prev_price) > 1e-12:
      out[i] = prices[i] / prev_price
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocr100_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ROCR100 calculation (Parallel)."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  out[:period] = np.nan

  for i in range(period, n):
    prev_price = prices[i - period]
    if abs(prev_price) > 1e-12:
      out[i] = (prices[i] / prev_price) * 100.0
    else:
      out[i] = 0.0

  return out
