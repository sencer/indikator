"""Numba-optimized KAMA (Kaufman Adaptive Moving Average) calculation.

Uses Efficiency Ratio to adapt smoothing constant. Matches TA-Lib exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_kama_numba(
  prices: NDArray[np.float64],
  period: int,
  fast_period: int,
  slow_period: int,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled KAMA with O(1) rolling volatility update.

  Matches TA-Lib: First KAMA at index 'period', seeded with price[period-1].

  ER = abs(change over period) / sum(abs(daily changes))
  SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
  KAMA = KAMA_prev + SC * (price - KAMA_prev)

  Args:
    prices: Array of prices
    period: Efficiency ratio period (typically 10)
    fast_period: Fast EMA period (typically 2)
    slow_period: Slow EMA period (typically 30)

  Returns:
    Array of KAMA values
  """
  n = len(prices)

  if n <= period:
    return np.full(n, np.nan)

  kama = np.empty(n, dtype=np.float64)

  # Smoothing constants
  fast_sc = 2.0 / (fast_period + 1)
  slow_sc = 2.0 / (slow_period + 1)
  sc_diff = fast_sc - slow_sc

  # TA-Lib: First valid KAMA is at index 'period', not 'period-1'
  for i in range(period):
    kama[i] = np.nan

  # Compute initial volatility: sum of abs changes from index 1 to period
  # This captures 'period' changes [1-0, 2-1, ..., period-(period-1)]
  volatility = 0.0
  for i in range(1, period + 1):
    volatility += abs(prices[i] - prices[i - 1])

  # Seed KAMA with price[period-1] (TA-Lib behavior)
  current_kama = prices[period - 1]

  # First KAMA at index 'period'
  # Direction from prices[0] to prices[period]
  direction = abs(prices[period] - prices[0])

  if volatility > 1e-10:
    er = direction / volatility
  else:
    er = 0.0

  sc = (er * sc_diff + slow_sc) ** 2
  current_kama = current_kama + sc * (prices[period] - current_kama)
  kama[period] = current_kama

  # Main loop from index period+1 onwards
  for i in range(period + 1, n):
    # Update volatility: remove oldest change, add newest
    old_change = abs(prices[i - period] - prices[i - period - 1])
    new_change = abs(prices[i] - prices[i - 1])
    volatility = volatility - old_change + new_change

    # Direction from prices[i-period] to prices[i]
    direction = abs(prices[i] - prices[i - period])

    # Efficiency Ratio
    if volatility > 1e-10:
      er = direction / volatility
    else:
      er = 0.0

    # Smoothing Constant
    sc = (er * sc_diff + slow_sc) ** 2

    # Update KAMA
    current_kama = current_kama + sc * (prices[i] - current_kama)
    kama[i] = current_kama

  return kama
