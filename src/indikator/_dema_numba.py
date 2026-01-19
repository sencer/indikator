"""Numba-optimized DEMA (Double EMA) calculation.

Uses loop fusion to compute EMA1 and EMA2 in a single pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_dema_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled DEMA with fused EMA computation.

  DEMA = 2 * EMA1 - EMA2

  Uses loop fusion: computes both EMA passes in a single iteration
  after initialization, keeping values in registers.

  Args:
    prices: Array of prices
    period: EMA period

  Returns:
    Array of DEMA values
  """
  n = len(prices)

  if n < period:
    return np.full(n, np.nan)

  dema = np.empty(n, dtype=np.float64)

  # EMA multiplier
  k = 2.0 / (period + 1)
  k1 = 1.0 - k

  # Phase 1: Seed EMA1 with SMA of first 'period' values
  sma_sum = 0.0
  for i in range(period):
    sma_sum += prices[i]
    dema[i] = np.nan

  ema1 = sma_sum / period

  # Phase 2: Build up EMA2 seed (need 'period' EMA1 values)
  # First EMA1 value is at index period-1
  # We need to accumulate period more EMA1 values for EMA2 SMA seed

  ema2_sum = ema1  # First EMA1 value contributes to EMA2 seed
  ema1_count = 1

  # Continue computing EMA1, accumulating for EMA2 seed
  for i in range(period, 2 * period - 1):
    if i >= n:
      break
    ema1 = prices[i] * k + ema1 * k1
    ema2_sum += ema1
    ema1_count += 1
    dema[i] = np.nan

  if n < 2 * period - 1:
    return dema

  # Seed EMA2
  ema2 = ema2_sum / period

  # First valid DEMA at index 2*period - 2
  first_idx = 2 * period - 2
  dema[first_idx] = 2.0 * ema1 - ema2

  # Phase 3: Main fused loop - compute EMA1, EMA2, DEMA in one pass
  for i in range(first_idx + 1, n):
    ema1 = prices[i] * k + ema1 * k1
    ema2 = ema1 * k + ema2 * k1
    dema[i] = 2.0 * ema1 - ema2

  return dema
