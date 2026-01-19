"""Numba-optimized ROC (Rate of Change) calculation.

This module contains JIT-compiled functions for ROC calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_roc_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate ROC: ((Price - Prev) / Prev) * 100."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)
  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    curr = prices[i]

    if prev != 0.0:
      out[i] = (curr - prev) / prev * 100.0
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocp_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate ROCP: (Price - Prev) / Prev."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)
  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    curr = prices[i]

    if prev != 0.0:
      out[i] = (curr - prev) / prev
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocr_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate ROCR: Price / Prev."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)
  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    curr = prices[i]

    if prev != 0.0:
      out[i] = curr / prev
    else:
      out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_rocr100_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate ROCR100: (Price / Prev) * 100."""
  n = len(prices)
  out = np.empty(n, dtype=np.float64)
  out[:period] = np.nan

  for i in range(period, n):
    prev = prices[i - period]
    curr = prices[i]

    if prev != 0.0:
      out[i] = curr / prev * 100.0
    else:
      out[i] = 0.0

  return out
