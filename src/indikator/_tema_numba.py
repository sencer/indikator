"""Numba-optimized TEMA (Triple EMA) calculation.

Uses loop fusion to compute EMA1, EMA2, EMA3 in a single pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_tema_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled TEMA with fused triple-EMA computation.

  TEMA = 3 * EMA1 - 3 * EMA2 + EMA3

  Uses loop fusion: after initialization, computes all three EMA
  stages in a single loop, keeping values in registers.

  Args:
    prices: Array of prices
    period: EMA period

  Returns:
    Array of TEMA values
  """
  n = len(prices)
  lookback = 3 * period - 3

  if n <= lookback:
    return np.full(n, np.nan)

  tema = np.empty(n, dtype=np.float64)

  # EMA multiplier
  k = 2.0 / (period + 1)
  k1 = 1.0 - k

  # Initialize all early values to NaN
  for i in range(lookback):
    tema[i] = np.nan

  # Phase 1: Seed EMA1 with SMA
  sma1_sum = 0.0
  for i in range(period):
    sma1_sum += prices[i]
  ema1 = sma1_sum / period

  # Phase 2: Build EMA1 values and seed EMA2
  sma2_sum = ema1
  for i in range(period, 2 * period - 1):
    ema1 = prices[i] * k + ema1 * k1
    sma2_sum += ema1
  ema2 = sma2_sum / period

  # Phase 3: Build EMA2 values and seed EMA3
  sma3_sum = ema2
  for i in range(2 * period - 1, 3 * period - 2):
    ema1 = prices[i] * k + ema1 * k1
    ema2 = ema1 * k + ema2 * k1
    sma3_sum += ema2
  ema3 = sma3_sum / period

  # First valid TEMA
  tema[lookback] = 3.0 * ema1 - 3.0 * ema2 + ema3

  # Phase 4: Main fused loop
  for i in range(lookback + 1, n):
    ema1 = prices[i] * k + ema1 * k1
    ema2 = ema1 * k + ema2 * k1
    ema3 = ema2 * k + ema3 * k1
    tema[i] = 3.0 * ema1 - 3.0 * ema2 + ema3

  return tema
