"""Numba-optimized Ultimate Oscillator calculation.

Multi-timeframe momentum using True Range and Buying Pressure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_ultosc_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period1: int,
  period2: int,
  period3: int,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled Ultimate Oscillator.

  ULTOSC uses weighted average of Buying Pressure/True Range ratios
  across three timeframes.

  BP = Close - min(Low, Prior Close)
  TR = max(High, Prior Close) - min(Low, Prior Close)
  ULTOSC = 100 * (4*Avg1 + 2*Avg2 + 1*Avg3) / 7

  Optimized: direct array access with O(1) rolling sum updates.

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    period1: Short period (typically 7)
    period2: Medium period (typically 14)
    period3: Long period (typically 28)

  Returns:
    Array of ULTOSC values (0-100)
  """
  n = len(close)
  lookback = period3

  if n <= lookback:
    return np.full(n, np.nan)

  ultosc = np.empty(n, dtype=np.float64)

  # Pre-compute all BP and TR values - avoids recomputation
  bp = np.empty(n, dtype=np.float64)
  tr = np.empty(n, dtype=np.float64)

  bp[0] = 0.0
  tr[0] = high[0] - low[0]

  for i in range(1, n):
    prior_close = close[i - 1]
    true_low = low[i] if low[i] < prior_close else prior_close
    true_high = high[i] if high[i] > prior_close else prior_close
    bp[i] = close[i] - true_low
    tr[i] = true_high - true_low

  # Initialize NaN for lookback
  for i in range(lookback):
    ultosc[i] = np.nan

  # Initialize sums for first valid point (at index period3)
  bp_sum1, bp_sum2, bp_sum3 = 0.0, 0.0, 0.0
  tr_sum1, tr_sum2, tr_sum3 = 0.0, 0.0, 0.0

  # Sum for period3 window [1, period3]
  for i in range(1, period3 + 1):
    bp_sum3 += bp[i]
    tr_sum3 += tr[i]

  # Sum for period2 window [period3 - period2 + 1, period3]
  for i in range(period3 - period2 + 1, period3 + 1):
    bp_sum2 += bp[i]
    tr_sum2 += tr[i]

  # Sum for period1 window [period3 - period1 + 1, period3]
  for i in range(period3 - period1 + 1, period3 + 1):
    bp_sum1 += bp[i]
    tr_sum1 += tr[i]

  # First ULTOSC
  avg1 = bp_sum1 / tr_sum1 if tr_sum1 > 1e-10 else 0.0
  avg2 = bp_sum2 / tr_sum2 if tr_sum2 > 1e-10 else 0.0
  avg3 = bp_sum3 / tr_sum3 if tr_sum3 > 1e-10 else 0.0
  ultosc[lookback] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0

  # Main loop with O(1) updates
  for i in range(lookback + 1, n):
    # Update period3 sum: remove [i - period3], add [i]
    bp_sum3 = bp_sum3 - bp[i - period3] + bp[i]
    tr_sum3 = tr_sum3 - tr[i - period3] + tr[i]

    # Update period2 sum: remove [i - period2], add [i]
    bp_sum2 = bp_sum2 - bp[i - period2] + bp[i]
    tr_sum2 = tr_sum2 - tr[i - period2] + tr[i]

    # Update period1 sum: remove [i - period1], add [i]
    bp_sum1 = bp_sum1 - bp[i - period1] + bp[i]
    tr_sum1 = tr_sum1 - tr[i - period1] + tr[i]

    # Calculate ULTOSC
    avg1 = bp_sum1 / tr_sum1 if tr_sum1 > 1e-10 else 0.0
    avg2 = bp_sum2 / tr_sum2 if tr_sum2 > 1e-10 else 0.0
    avg3 = bp_sum3 / tr_sum3 if tr_sum3 > 1e-10 else 0.0
    ultosc[i] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0

  return ultosc
