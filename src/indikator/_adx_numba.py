"""Numba-optimized ADX (Average Directional Index) calculation.

This module contains JIT-compiled functions for ADX calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_adx_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled ADX calculation.

  ADX measures trend strength (not direction).

  Steps:
  1. Calculate +DM and -DM (Directional Movement)
  2. Calculate TR (True Range)
  3. Smooth +DM, -DM, and TR using Wilder's smoothing
  4. Calculate +DI and -DI
  5. Calculate DX = |+DI - -DI| / (+DI + -DI) * 100
  6. ADX = Wilder's smoothed DX

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of closing prices
    period: Lookback period (typically 14)

  Returns:
    Tuple of (ADX, +DI, -DI) arrays
  """
  n = len(close)
  adx = np.full(n, np.nan)
  plus_di = np.full(n, np.nan)
  minus_di = np.full(n, np.nan)

  if n < period * 2:
    return adx, plus_di, minus_di

  # Arrays for directional movements and true range
  plus_dm = np.zeros(n)
  minus_dm = np.zeros(n)
  tr = np.zeros(n)

  # Calculate directional movements and true range
  for i in range(1, n):
    up_move = high[i] - high[i - 1]
    down_move = low[i - 1] - low[i]

    if up_move > down_move and up_move > 0:
      plus_dm[i] = up_move
    else:
      plus_dm[i] = 0.0

    if down_move > up_move and down_move > 0:
      minus_dm[i] = down_move
    else:
      minus_dm[i] = 0.0

    # True Range
    hl = high[i] - low[i]
    hc = abs(high[i] - close[i - 1])
    lc = abs(low[i] - close[i - 1])
    tr[i] = max(hl, hc, lc)

  # Calculate first smoothed values (sum of first 'period' values)
  smoothed_plus_dm = 0.0
  smoothed_minus_dm = 0.0
  smoothed_tr = 0.0

  for i in range(1, period + 1):
    smoothed_plus_dm += plus_dm[i]
    smoothed_minus_dm += minus_dm[i]
    smoothed_tr += tr[i]

  # Calculate first DI values at index 'period'
  if smoothed_tr > 1e-10:
    plus_di[period] = 100.0 * smoothed_plus_dm / smoothed_tr
    minus_di[period] = 100.0 * smoothed_minus_dm / smoothed_tr
  else:
    plus_di[period] = 0.0
    minus_di[period] = 0.0

  # Continue smoothing using Wilder's method
  for i in range(period + 1, n):
    smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
    smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]
    smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr[i]

    if smoothed_tr > 1e-10:
      plus_di[i] = 100.0 * smoothed_plus_dm / smoothed_tr
      minus_di[i] = 100.0 * smoothed_minus_dm / smoothed_tr
    else:
      plus_di[i] = 0.0
      minus_di[i] = 0.0

  # Calculate DX and ADX
  dx = np.zeros(n)
  for i in range(period, n):
    di_sum = plus_di[i] + minus_di[i]
    if di_sum > 1e-10:
      dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    else:
      dx[i] = 0.0

  # First ADX is average of first 'period' DX values
  if n >= period * 2:
    adx_sum = 0.0
    for i in range(period, period * 2):
      adx_sum += dx[i]
    adx[period * 2 - 1] = adx_sum / period

    # Subsequent ADX values using Wilder's smoothing
    for i in range(period * 2, n):
      adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

  return adx, plus_di, minus_di
