"""Numba-optimized CCI (Commodity Channel Index) calculation.

This module contains JIT-compiled functions for CCI calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_cci_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled CCI calculation.

  CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)

  Where:
  - Typical Price = (High + Low + Close) / 3
  - Mean Deviation = mean of absolute deviations from SMA

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of closing prices
    period: Lookback period (typically 20)

  Returns:
    Array of CCI values (unbounded, typically -100 to +100)
  """
  n = len(close)
  cci = np.full(n, np.nan)

  if n < period:
    return cci

  # Calculate typical price
  tp = (high + low + close) / 3.0

  # Initialize rolling sum for SMA
  rolling_sum = 0.0
  for j in range(period):
    rolling_sum += tp[j]

  for i in range(period - 1, n):
    if i > period - 1:
      # Slide window: remove oldest, add newest
      rolling_sum = rolling_sum - tp[i - period] + tp[i]

    sma = rolling_sum / period

    # Mean deviation still requires full loop (can't be optimized with rolling sum
    # because abs() is non-linear)
    md_sum = 0.0
    for j in range(i - period + 1, i + 1):
      md_sum += abs(tp[j] - sma)
    mean_deviation = md_sum / period

    # Calculate CCI
    if mean_deviation > 1e-10:
      cci[i] = (tp[i] - sma) / (0.015 * mean_deviation)
    else:
      cci[i] = 0.0

  return cci
