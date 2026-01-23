"""Numba-optimized Momentum calculation.

Simple price difference indicator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


# Optimized: use Numpy directly as it outperforms Numba for simple subtract
def compute_mom_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Momentum calculation using optimized NumPy.

  MOM = price[i] - price[i - period]
  """
  n = len(prices)

  if n <= period:
    return np.full(n, np.nan)

  mom = np.empty(n, dtype=np.float64)
  mom[:period] = np.nan

  # Calculate directly into the output array slice
  np.subtract(prices[period:], prices[:-period], out=mom[period:])

  return mom
