"""Numba-optimized Momentum calculation.

Simple price difference indicator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_mom_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Momentum calculation using high-performance SIMD-optimized Numba core.

  MOM = price[i] - price[i - period]

  Complies with optimization_checks.md:
  - Sequential O(n) strategy for light arithmetic.
  - @jit(fastmath=True) for SIMD vectorization.
  - Pre-allocation with np.empty() and manual NaN assignment.
  - Zero-copy input handling.
  - Optimized for SIMD by using local contiguous views.
  """
  n = prices.shape[0]
  out = np.empty(n, dtype=np.float64)

  if n < period + 1:
    out[:] = np.nan
    return out

  # Set NaNs for the lookback window
  out[:period] = np.nan

  # SIMD optimized loop using contiguous views
  # This pattern (C = A - B) is easily vectorized by Numba
  p_high = prices[period:]
  p_low = prices[:-period]
  o = out[period:]

  for i in range(n - period):
    o[i] = p_high[i] - p_low[i]

  return out
