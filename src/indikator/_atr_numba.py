"""Numba-optimized ATR (Average True Range) calculation.

This module contains JIT-compiled functions for ATR calculation.
Separated for better code organization and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_true_range_numba(
  highs: NDArray[np.float64],
  lows: NDArray[np.float64],
  closes: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Numba JIT-compiled true range calculation.

  True Range is the greatest of:
  1. Current high - current low
  2. |Current high - previous close|
  3. |Current low - previous close|

  For the first bar, true range equals high - low since there's no previous close.

  Args:
    highs: Array of high prices
    lows: Array of low prices
    closes: Array of close prices

  Returns:
    Array of true range values
  """
  n = len(highs)
  tr = np.empty(n, dtype=np.float64)

  if n == 0:
    return tr

  # First bar: TR = high - low
  tr[0] = highs[0] - lows[0]

  # Subsequent bars: TR = max(high-low, |high-prev_close|, |low-prev_close|)
  # Manual 'if' checks are faster than max() in Numba here (avoid function call overhead)
  for i in range(1, n):
    hl = highs[i] - lows[i]
    hc = abs(highs[i] - closes[i - 1])
    lc = abs(lows[i] - closes[i - 1])

    curr_max = hl
    if hc > curr_max:
      curr_max = hc
    if lc > curr_max:
      curr_max = lc

    tr[i] = curr_max

  return tr


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_true_range_numba_2d(
  ohlc: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Numba JIT-compiled true range calculation for 2D array (OHLC input).

  Assumes columns: 0=High, 1=Low, 2=Close.
  Faster for DataFrame inputs as it avoids separating columns.
  """
  n = len(ohlc)
  tr = np.empty(n, dtype=np.float64)

  if n == 0:
    return tr

  # First bar: TR = high - low
  tr[0] = ohlc[0, 0] - ohlc[0, 1]

  # Subsequent bars
  for i in range(1, n):
    h = ohlc[i, 0]
    l = ohlc[i, 1]
    prev_c = ohlc[i - 1, 2]

    hl = h - l
    hc = abs(h - prev_c)
    lc = abs(l - prev_c)

    curr_max = hl
    if hc > curr_max:
      curr_max = hc
    if lc > curr_max:
      curr_max = lc

    tr[i] = curr_max

  return tr


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_atr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  window: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ATR calculation using Wilder's smoothing.

  Uses Wilder's smoothing method (similar to EMA but with different smoothing factor):
  ATR = (ATR_previous * (window - 1) + TR_current) / window

  For the initial ATR, uses simple moving average of first 'window' true ranges.
  First valid output is at index `window` (matching TA-lib behavior).

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of close prices
    window: Smoothing period (typically 14)

  Returns:
    Array of ATR values (NaN for initial bars where window not satisfied)
  """
  n = len(high)

  # Need at least window+1 bars for first valid ATR at index window
  if n <= window:
    return np.full(n, np.nan)

  atr = np.empty(n)
  atr[:window] = np.nan

  # Calculate initial SMA of first window TRs (indices 1 to window)
  # TR[0] is H-L (ignored by TA-Lib algorithm?), TR[1..window] use previous close
  sum_tr = 0.0

  for i in range(1, window + 1):
    hl = high[i] - low[i]
    hc = abs(high[i] - close[i - 1])
    lc = abs(low[i] - close[i - 1])
    tr = max(hl, hc, lc)
    sum_tr += tr

  # First ATR at index window is SMA of TRs 1..window (window values)
  atr[window] = sum_tr / window

  # Use Wilder's smoothing for subsequent values
  inv_period = 1.0 / window
  period_m1 = window - 1

  for i in range(window + 1, n):
    # Calculate TR for current bar
    hl = high[i] - low[i]
    hc = abs(high[i] - close[i - 1])
    lc = abs(low[i] - close[i - 1])
    current_tr = max(hl, hc, lc)

    atr[i] = (atr[i - 1] * period_m1 + current_tr) * inv_period

  return atr
