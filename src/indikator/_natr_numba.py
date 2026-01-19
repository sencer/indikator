"""Numba-optimized NATR (Normalized ATR) calculation.

Fused ATR + normalization in single pass. Matches TA-Lib exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_natr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled NATR with fused ATR + normalization.

  NATR = (ATR / Close) * 100

  Matches TA-Lib seeding: First ATR = SMA of TR[1:period+1], output at index 'period'.

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    period: ATR period

  Returns:
    Array of NATR values (percentage)
  """
  n = len(high)

  if n <= period:
    return np.full(n, np.nan)

  natr = np.empty(n, dtype=np.float64)
  natr[:period] = np.nan

  # TA-Lib seeds ATR with SMA of TR[1] to TR[period] (using prior close)
  # This gives 'period' TR values for the first ATR at index 'period'
  sum_tr = 0.0
  for i in range(1, period + 1):
    hl = high[i] - low[i]
    hc = abs(high[i] - close[i - 1])
    lc = abs(low[i] - close[i - 1])
    tr = max(hl, max(hc, lc))
    sum_tr += tr

  # First ATR at index period
  atr = sum_tr / period

  # First NATR
  if close[period] > 0:
    natr[period] = (atr / close[period]) * 100.0
  else:
    natr[period] = np.nan

  # Wilder's smoothing constants
  inv_period = 1.0 / period
  period_m1 = period - 1

  # Main fused loop: TR -> ATR -> NATR
  for i in range(period + 1, n):
    hl = high[i] - low[i]
    hc = abs(high[i] - close[i - 1])
    lc = abs(low[i] - close[i - 1])
    tr = max(hl, max(hc, lc))

    # Wilder's smoothing
    atr = (atr * period_m1 + tr) * inv_period

    # Normalize
    if close[i] > 0:
      natr[i] = (atr / close[i]) * 100.0
    else:
      natr[i] = np.nan

  return natr
