"""Numba-optimized ADX (Average Directional Index) calculation.

This module contains JIT-compiled functions for ADX calculation.
Uses branchless arithmetic for maximum performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
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

  States aligned with TA-Lib:
  - DM/DI/DX valid starting at index `period-1` (e.g., 13 for p=14).
  - ADX valid starting at index `2*period-1` (e.g., 27 for p=14).
    TA-Lib ADX uses a window of DX values starting from the 2nd valid DX (index `period`),
    skipping the first valid DX at `period-1`.
  """
  n = len(close)

  if n < period * 2:
    return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

  adx = np.empty(n)
  plus_di = np.empty(n)
  minus_di = np.empty(n)

  # Initial fills
  adx[: period * 2 - 1] = np.nan
  plus_di[: period - 1] = np.nan
  minus_di[: period - 1] = np.nan

  inv_period = 1.0 / period
  k1 = 1.0 - inv_period

  smoothed_plus_dm = 0.0
  smoothed_minus_dm = 0.0
  smoothed_tr = 0.0

  prev_high = high[0]
  prev_low = low[0]
  prev_close = close[0]

  # 1. Initialization Phase (Accumulate sums for 1..period-1)
  # This builds the first Valid Smoothed DM/TR at index `period-1`.
  for i in range(1, period):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    # Branchless DM calculation using multiplied masks
    # Use strict inequality. If up == down, both are 0.
    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_plus_dm += p_dm
    smoothed_minus_dm += m_dm
    smoothed_tr += curr_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # Calculate first DI/DX at index `period-1`
  inv_tr = 1.0 / (smoothed_tr + EPSILON)
  p_di = 100.0 * smoothed_plus_dm * inv_tr
  m_di = 100.0 * smoothed_minus_dm * inv_tr

  plus_di[period - 1] = p_di
  minus_di[period - 1] = m_di

  # DX at period-1 is technically valid, but TA-Lib ADX seems to start smoothing
  # from the NEXT one. So we don't start dx_sum accumulator here.
  dx_sum = 0.0

  # 2. DI/DX Phase (period to 2*period - 1)
  # Accumulate DX values for ADX SMA window (starting from index `period` to `2*period - 1`)
  # Range is [period, 2*period). Total `period` items.
  for i in range(period, period * 2):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

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

  # Initialize ADX at index 2*period - 1
  current_adx = dx_sum * inv_period
  adx[period * 2 - 1] = current_adx

  # 3. Main Phase (2*period to n)
  for i in range(period * 2, n):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

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
  """Numba JIT-compiled ADX calculation (Returns ADX only)."""
  n = len(close)

  if n < period * 2:
    return np.full(n, np.nan)

  adx = np.empty(n)
  adx[: period * 2 - 1] = np.nan

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
  for i in range(1, period):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_plus_dm += p_dm
    smoothed_minus_dm += m_dm
    smoothed_tr += curr_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # First DI/DX at `period-1` (Skipped for ADX SUM)
  inv_tr = 1.0 / (smoothed_tr + EPSILON)
  # (Calculations skipped)

  dx_sum = 0.0

  # 2. DI/DX Phase
  for i in range(period, period * 2):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

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

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

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


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_dms_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled calculation for Smoothed +DM, -DM, and TR."""
  n = len(close)
  if n < period:
    nan = np.full(n, np.nan)
    return nan, nan, nan

  plus_dm = np.empty(n)
  minus_dm = np.empty(n)
  tr = np.empty(n)

  # Invalid until index period-1
  plus_dm[: period - 1] = np.nan
  minus_dm[: period - 1] = np.nan
  tr[: period - 1] = np.nan

  inv_period = 1.0 / period
  k1 = 1.0 - inv_period

  smoothed_p_dm = 0.0
  smoothed_m_dm = 0.0
  smoothed_tr = 0.0

  prev_high = high[0]
  prev_low = low[0]
  prev_close = close[0]

  # Initialization
  for i in range(1, period):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_p_dm += p_dm
    smoothed_m_dm += m_dm
    smoothed_tr += curr_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # Output starting at index period-1
  plus_dm[period - 1] = smoothed_p_dm
  minus_dm[period - 1] = smoothed_m_dm
  tr[period - 1] = smoothed_tr

  for i in range(period, n):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_p_dm = smoothed_p_dm * k1 + p_dm
    smoothed_m_dm = smoothed_m_dm * k1 + m_dm
    smoothed_tr = smoothed_tr * k1 + curr_tr

    plus_dm[i] = smoothed_p_dm
    minus_dm[i] = smoothed_m_dm
    tr[i] = smoothed_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  return plus_dm, minus_dm, tr


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_dx_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled DX calculation (Fused Kernel).

  Computes smoothed DM/TR internally in a single pass to avoid
  allocating intermediate arrays.
  """
  n = len(close)
  if n < period:
    return np.full(n, np.nan)

  dx = np.empty(n)
  dx[: period - 1] = np.nan

  inv_period = 1.0 / period
  k1 = 1.0 - inv_period

  # State variables for smoothing
  smoothed_p_dm = 0.0
  smoothed_m_dm = 0.0
  smoothed_tr = 0.0

  prev_high = high[0]
  prev_low = low[0]
  prev_close = close[0]

  # 1. Initialization Phase (1 to period-1)
  for i in range(1, period):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_p_dm += p_dm
    smoothed_m_dm += m_dm
    smoothed_tr += curr_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # 2. Main Loop (period to n)
  # Note: The first DX value is at index `period-1`?
  # TA-Lib behavior: DX is validated at `period-1`?
  # Let's check typical TA-Lib output.
  # TA-Lib ADX has lookback 2*period-1. DX has lookback period.
  # So DX[period-1] should be valid.

  # Calculate for index `period-1`
  inv_tr = 1.0 / (smoothed_tr + EPSILON)
  p_di = 100.0 * smoothed_p_dm * inv_tr
  m_di = 100.0 * smoothed_m_dm * inv_tr

  di_sum = p_di + m_di + EPSILON
  dx[period - 1] = 100.0 * abs(p_di - m_di) / di_sum

  for i in range(period, n):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_p_dm = smoothed_p_dm * k1 + p_dm
    smoothed_m_dm = smoothed_m_dm * k1 + m_dm
    smoothed_tr = smoothed_tr * k1 + curr_tr

    inv_tr = 1.0 / (smoothed_tr + EPSILON)
    p_di = 100.0 * smoothed_p_dm * inv_tr
    m_di = 100.0 * smoothed_m_dm * inv_tr

    di_sum = p_di + m_di + EPSILON
    dx[i] = 100.0 * abs(p_di - m_di) / di_sum

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  return dx


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_di_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled calculation for +DI and -DI."""
  n = len(close)
  if n < period:
    nan = np.full(n, np.nan)
    return nan, nan

  # Reuse DM calculation logic
  # Optimization: We could inline this to avoid tuple unpacking/allocation if needed,
  # but calling the jit function is usually efficient enough in Numba.
  # However, to avoid allocating 'plus_dm', 'minus_dm', 'tr' arrays just to read them once,
  # we should inline the loop. But for now, let's trust Numba's inlining or just use the function.
  # Actually, 'compute_dms_numba' allocates arrays. We want to avoid that.
  # So we replicate the loop logic.

  plus_di = np.empty(n)
  minus_di = np.empty(n)

  # Invalid until index period-1
  plus_di[: period - 1] = np.nan
  minus_di[: period - 1] = np.nan

  inv_period = 1.0 / period
  k1 = 1.0 - inv_period

  smoothed_p_dm = 0.0
  smoothed_m_dm = 0.0
  smoothed_tr = 0.0

  prev_high = high[0]
  prev_low = low[0]
  prev_close = close[0]

  # Initialization
  for i in range(1, period):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_p_dm += p_dm
    smoothed_m_dm += m_dm
    smoothed_tr += curr_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  # First DI at period-1
  inv_tr = 1.0 / (smoothed_tr + EPSILON)
  plus_di[period - 1] = 100.0 * smoothed_p_dm * inv_tr
  minus_di[period - 1] = 100.0 * smoothed_m_dm * inv_tr

  for i in range(period, n):
    curr_high = high[i]
    curr_low = low[i]
    curr_close = close[i]

    up_move = curr_high - prev_high
    down_move = prev_low - curr_low

    up_pos = 1.0 if up_move > 0 else 0.0
    down_pos = 1.0 if down_move > 0 else 0.0

    up_gt_down = 1.0 if up_move > down_move else 0.0
    down_gt_up = 1.0 if down_move > up_move else 0.0

    p_dm = up_move * up_gt_down * up_pos
    m_dm = down_move * down_gt_up * down_pos

    hl = curr_high - curr_low
    hc = abs(curr_high - prev_close)
    lc = abs(low[i] - prev_close)
    curr_tr = max(hl, hc, lc)

    smoothed_p_dm = smoothed_p_dm * k1 + p_dm
    smoothed_m_dm = smoothed_m_dm * k1 + m_dm
    smoothed_tr = smoothed_tr * k1 + curr_tr

    inv_tr = 1.0 / (smoothed_tr + EPSILON)
    # Write directly to output arrays
    plus_di[i] = 100.0 * smoothed_p_dm * inv_tr
    minus_di[i] = 100.0 * smoothed_m_dm * inv_tr

    prev_high = curr_high
    prev_low = curr_low
    prev_close = curr_close

  return plus_di, minus_di


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_adxr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled ADXR calculation.

  Computes ADX internally and then applies the ADXR smoothing:
  ADXR = (ADX + ADX[i - period]) / 2
  """
  n = len(close)
  # ADXR needs ADX valid at i and i-period.
  # ADX is valid starting at 2*period - 1.
  # So ADXR is valid starting at (2*period - 1) + period = 3*period - 1.

  # First, compute ADX (pure)
  adx = compute_adx_numba_pure(high, low, close, period)

  adxr = np.full(n, np.nan)

  start_idx = period * 3 - 1
  if n <= start_idx:
    return adxr

  # Vectorized calculation on the array is efficient in Numba
  # ADXR[i] = (ADX[i] + ADX[i-period]) / 2
  for i in range(start_idx, n):
    adxr[i] = (adx[i] + adx[i - period]) / 2.0

  return adxr
