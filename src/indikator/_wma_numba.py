"""Numba-optimized WMA (Weighted Moving Average) calculation.

Uses O(1) rolling update similar to slope calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_wma_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled WMA with O(1) rolling update.

  WMA = sum(price[i] * weight[i]) / sum(weights)
  where weight[i] = i + 1 (most recent has highest weight)

  Optimization: Instead of O(period) per step, uses rolling update:
  - Maintain weighted_sum and unweighted_sum
  - When sliding: subtract unweighted_sum, add new_price * period
  - This works because shifting weights down by 1 = subtracting sum of prices

  Args:
    prices: Array of prices
    period: Lookback period

  Returns:
    Array of WMA values
  """
  n = len(prices)

  if n < period:
    return np.full(n, np.nan)

  wma = np.empty(n, dtype=np.float64)

  # Fill NaN for warmup
  for i in range(period - 1):
    wma[i] = np.nan

  # Weight sum: 1 + 2 + ... + period = period * (period + 1) / 2
  weight_sum = period * (period + 1) / 2
  inv_weight_sum = 1.0 / weight_sum

  # Initial weighted sum: price[0]*1 + price[1]*2 + ... + price[period-1]*period
  weighted_sum = 0.0
  unweighted_sum = 0.0

  for j in range(period):
    weight = j + 1
    weighted_sum += prices[j] * weight
    unweighted_sum += prices[j]

  wma[period - 1] = weighted_sum * inv_weight_sum

  # Rolling update loop
  # When sliding from [i-period+1, i] to [i-period+2, i+1]:
  # - Remove: prices[i-period+1] had weight 1
  # - Shift: all other prices' weights decrease by 1 (subtract unweighted_sum - leaving_price)
  # - Add: new price gets weight = period
  #
  # New weighted_sum = old_weighted_sum - unweighted_sum + new_price * period
  # New unweighted_sum = old_unweighted_sum - leaving_price + new_price

  for i in range(period, n):
    leaving_price = prices[i - period]
    entering_price = prices[i]

    # Update: shift all weights down by 1, remove old, add new
    weighted_sum = weighted_sum - unweighted_sum + entering_price * period
    unweighted_sum = unweighted_sum - leaving_price + entering_price

    wma[i] = weighted_sum * inv_weight_sum

  return wma
