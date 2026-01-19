"""Numba-optimized CCI (Commodity Channel Index) calculation.

This module contains JIT-compiled functions for CCI calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_cci_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
  constant: float,
) -> NDArray[np.float64]:
  """Numba JIT-compiled CCI calculation with optimized rolling sum.

  CCI = (Typical Price - SMA(Typical Price)) / (constant * Mean Deviation)

  Where:
  - Typical Price = (High + Low + Close) / 3
  - Mean Deviation = mean of absolute deviations from SMA

  Optimizations:
  - O(1) rolling sum for mean calculation (vs O(period) in naive approach)
  - Circular buffer to avoid array shifting
  - Precomputed constants

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    period: Lookback period (typically 14)
    constant: Scaling constant (typically 0.015)

  Returns:
    Array of CCI values
  """
  n = len(high)

  if n < period:
    return np.full(n, np.nan)

  cci = np.empty(n)
  cci[: period - 1] = np.nan

  # Circular buffer for typical prices
  circ_buffer = np.empty(period, dtype=np.float64)

  # Precompute constants
  inv_period = 1.0 / period
  if constant == 0:
    inv_const = 0.0
  else:
    inv_const = 1.0 / constant

  # Running sum for O(1) mean calculation
  running_sum = 0.0
  circ_idx = 0

  # Fill initial buffer (first period-1 values)
  for j in range(period - 1):
    tp = (high[j] + low[j] + close[j]) / 3.0
    circ_buffer[circ_idx] = tp
    running_sum += tp
    circ_idx = (circ_idx + 1) % period

  # Main loop
  for i in range(period - 1, n):
    tp = (high[i] + low[i] + close[i]) / 3.0

    # For i >= period, subtract old value from running sum
    if i >= period:
      old_tp = circ_buffer[circ_idx]
      running_sum -= old_tp

    # Add current value to buffer and running sum
    circ_buffer[circ_idx] = tp
    running_sum += tp

    # O(1) mean calculation using running sum
    the_average = running_sum * inv_period

    # Mean deviation (O(period) - unavoidable for MAD)
    md_sum = 0.0
    for j in range(period):
      md_sum += abs(circ_buffer[j] - the_average)

    # Calculate CCI
    temp_real = tp - the_average

    if abs(temp_real) > EPSILON and md_sum > EPSILON:
      cci[i] = temp_real * inv_const / (md_sum * inv_period)
    else:
      cci[i] = 0.0

    # Advance circular buffer index
    circ_idx = (circ_idx + 1) % period

  return cci
