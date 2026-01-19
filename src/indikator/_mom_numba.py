"""Numba-optimized Momentum calculation.

Simple price difference indicator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


def compute_mom_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Momentum calculation using optimized NumPy.

  MOM = price[i] - price[i - period]

  Uses np.subtract with 'out' parameter to avoid intermediate allocations.
  This is faster than Numba for simple vectorized operations due to lower overhead.

  Args:
    prices: Array of prices (typically closing prices)
    period: Lookback period

  Returns:
    Array of momentum values (NaN for initial bars)
  """
  n = len(prices)

  if n <= period:
    return np.full(n, np.nan)

  mom = np.empty(n, dtype=np.float64)
  mom[:period] = np.nan

  # Calculate directly into the output array slice
  np.subtract(prices[period:], prices[:-period], out=mom[period:])

  return mom
