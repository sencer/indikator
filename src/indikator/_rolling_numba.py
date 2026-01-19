"""Numba-optimized rolling min/max calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_midprice_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPRICE: (highest high + lowest low) / 2 over period."""
  n = len(high)
  out = np.empty(n, dtype=np.float64)

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  for i in range(period - 1, n):
    # Find max high and min low in window [i - period + 1, i]
    max_high = high[i - period + 1]
    min_low = low[i - period + 1]

    for j in range(i - period + 2, i + 1):
      if high[j] > max_high:
        max_high = high[j]
      if low[j] < min_low:
        min_low = low[j]

    out[i] = (max_high + min_low) / 2.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_midpoint_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPOINT: (highest + lowest) / 2 over period."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  for i in range(period - 1, n):
    # Find max and min in window [i - period + 1, i]
    max_val = data[i - period + 1]
    min_val = data[i - period + 1]

    for j in range(i - period + 2, i + 1):
      if data[j] > max_val:
        max_val = data[j]
      if data[j] < min_val:
        min_val = data[j]

    out[i] = (max_val + min_val) / 2.0

  return out
