"""Numba-optimized ADX (Average Directional Index) calculation.

This module contains JIT-compiled functions for ADX calculation.
Uses branchless arithmetic for maximum performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-10  # Minimum denominator value


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_adx_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled ADX calculation with branchless DM logic.

  Uses multiplied boolean masks instead of if/else for ~2x speedup.
  Returns ADX, +DI, -DI.
  """
  n = len(close)

  if n < period * 2:
    return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

  adx = np.empty(n)
  plus_di = np.empty(n)
  minus_di = np.empty(n)
  adx[: period * 2 - 1] = np.nan
  plus_di[:period] = np.nan
  minus_di[:period] = np.nan

  # Precompute constants
  inv_period = 1.0 / period
  k1 = 1.0 - inv_period

  # State variables
  smoothed_plus_dm = 0.0
  smoothed_minus_dm = 0.0
  smoothed_tr = 0.0

  # Register variables (previous prices)
  prev_high = high[0]
  prev_low = low[0]
  prev_close = close[0]

  # 1. Initialization Phase (Accumulate sums for 1..period)
  for i in range(1, period + 1):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    # Branchless DM calculation using multiplied masks
    up_is_max = 1.0 if up_move > down_move else 0.0
    down_is_max = 1.0 - up_is_max
    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    p_dm = up_move * up_is_max * up_pos
    m_dm = down_move * down_is_max * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(curr_low - prev_close)
    curr_tr = max(hl, max(hc, lc))

    smoothed_plus_dm += p_dm
    smoothed_minus_dm += m_dm
    smoothed_tr += curr_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # Calculate first DI/DX at index `period`
  inv_tr = 1.0 / (smoothed_tr + EPSILON)
  p_di = 100.0 * smoothed_plus_dm * inv_tr
  m_di = 100.0 * smoothed_minus_dm * inv_tr

  plus_di[period] = p_di
  minus_di[period] = m_di

  di_diff = abs(p_di - m_di)
  di_sum = p_di + m_di + EPSILON
  dx_sum = 100.0 * di_diff / di_sum

  # 2. DI/DX Phase (period+1 to 2*period - 1)
  for i in range(period + 1, period * 2):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_is_max = 1.0 if up_move > down_move else 0.0
    down_is_max = 1.0 - up_is_max
    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    p_dm = up_move * up_is_max * up_pos
    m_dm = down_move * down_is_max * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(curr_low - prev_close)
    curr_tr = max(hl, max(hc, lc))

    # Wilder's smoothing
    smoothed_plus_dm = smoothed_plus_dm * k1 + p_dm
    smoothed_minus_dm = smoothed_minus_dm * k1 + m_dm
    smoothed_tr = smoothed_tr * k1 + curr_tr

    inv_tr = 1.0 / (smoothed_tr + EPSILON)
    p_di = 100.0 * smoothed_plus_dm * inv_tr
    m_di = 100.0 * smoothed_minus_dm * inv_tr

    plus_di[i] = p_di
    minus_di[i] = m_di

    di_diff = abs(p_di - m_di)
    di_sum = p_di + m_di + EPSILON
    dx_sum += 100.0 * di_diff / di_sum

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # Initialize ADX
  current_adx = dx_sum * inv_period
  adx[period * 2 - 1] = current_adx

  # 3. Main Phase (2*period to n)
  for i in range(period * 2, n):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_is_max = 1.0 if up_move > down_move else 0.0
    down_is_max = 1.0 - up_is_max
    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    p_dm = up_move * up_is_max * up_pos
    m_dm = down_move * down_is_max * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(curr_low - prev_close)
    curr_tr = max(hl, max(hc, lc))

    smoothed_plus_dm = smoothed_plus_dm * k1 + p_dm
    smoothed_minus_dm = smoothed_minus_dm * k1 + m_dm
    smoothed_tr = smoothed_tr * k1 + curr_tr

    inv_tr = 1.0 / (smoothed_tr + EPSILON)
    p_di = 100.0 * smoothed_plus_dm * inv_tr
    m_di = 100.0 * smoothed_minus_dm * inv_tr

    plus_di[i] = p_di
    minus_di[i] = m_di

    di_diff = abs(p_di - m_di)
    di_sum = p_di + m_di + EPSILON
    dx_val = 100.0 * di_diff / di_sum

    current_adx = current_adx * k1 + dx_val * inv_period
    adx[i] = current_adx

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  return adx, plus_di, minus_di


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_adx_numba_pure(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ADX calculation (Returns ADX only).

  Optimized version using branchless DM calculation for maximum performance.
  Skips allocation and writing of DI arrays.
  """
  n = len(close)

  if n < period * 2:
    return np.full(n, np.nan)

  adx = np.empty(n)
  for i in range(period * 2 - 1):
    adx[i] = np.nan

  # Precompute constants
  inv_period = 1.0 / period
  k1 = 1.0 - inv_period

  # State variables
  smoothed_plus_dm = 0.0
  smoothed_minus_dm = 0.0
  smoothed_tr = 0.0

  # Register variables
  prev_high = high[0]
  prev_low = low[0]
  prev_close = close[0]

  # 1. Initialization Phase
  for i in range(1, period + 1):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    # Branchless DM using multiplied masks
    up_is_max = 1.0 if up_move > down_move else 0.0
    down_is_max = 1.0 - up_is_max
    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    p_dm = up_move * up_is_max * up_pos
    m_dm = down_move * down_is_max * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(curr_low - prev_close)
    curr_tr = max(hl, max(hc, lc))

    smoothed_plus_dm += p_dm
    smoothed_minus_dm += m_dm
    smoothed_tr += curr_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # First DI/DX at index `period`
  inv_tr = 1.0 / (smoothed_tr + EPSILON)
  p_di = 100.0 * smoothed_plus_dm * inv_tr
  m_di = 100.0 * smoothed_minus_dm * inv_tr

  di_diff = abs(p_di - m_di)
  di_sum = p_di + m_di + EPSILON
  dx_sum = 100.0 * di_diff / di_sum

  # 2. DI/DX Phase
  for i in range(period + 1, period * 2):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_is_max = 1.0 if up_move > down_move else 0.0
    down_is_max = 1.0 - up_is_max
    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    p_dm = up_move * up_is_max * up_pos
    m_dm = down_move * down_is_max * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(curr_low - prev_close)
    curr_tr = max(hl, max(hc, lc))

    smoothed_plus_dm = smoothed_plus_dm * k1 + p_dm
    smoothed_minus_dm = smoothed_minus_dm * k1 + m_dm
    smoothed_tr = smoothed_tr * k1 + curr_tr

    inv_tr = 1.0 / (smoothed_tr + EPSILON)
    p_di = 100.0 * smoothed_plus_dm * inv_tr
    m_di = 100.0 * smoothed_minus_dm * inv_tr

    di_diff = abs(p_di - m_di)
    di_sum = p_di + m_di + EPSILON
    dx_sum += 100.0 * di_diff / di_sum

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # Initialize ADX
  current_adx = dx_sum * inv_period
  adx[period * 2 - 1] = current_adx

  # 3. Main Phase
  for i in range(period * 2, n):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_is_max = 1.0 if up_move > down_move else 0.0
    down_is_max = 1.0 - up_is_max
    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    p_dm = up_move * up_is_max * up_pos
    m_dm = down_move * down_is_max * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(curr_low - prev_close)
    curr_tr = max(hl, max(hc, lc))

    smoothed_plus_dm = smoothed_plus_dm * k1 + p_dm
    smoothed_minus_dm = smoothed_minus_dm * k1 + m_dm
    smoothed_tr = smoothed_tr * k1 + curr_tr

    inv_tr = 1.0 / (smoothed_tr + EPSILON)
    p_di = 100.0 * smoothed_plus_dm * inv_tr
    m_di = 100.0 * smoothed_minus_dm * inv_tr

    di_diff = abs(p_di - m_di)
    di_sum = p_di + m_di + EPSILON
    dx_val = 100.0 * di_diff / di_sum

    current_adx = current_adx * k1 + dx_val * inv_period
    adx[i] = current_adx

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  return adx
