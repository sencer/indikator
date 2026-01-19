"""Numba-optimized Balance of Power (BOP) calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_bop_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Calculate Balance of Power (BOP).

  BOP = (Close - Open) / (High - Low)
  """
  n = len(close)
  out = np.empty(n, dtype=np.float64)

  for i in range(n):
    h = high[i]
    l = low[i]
    rng = h - l

    if rng <= 0.0:
      out[i] = 0.0
    else:
      out[i] = (close[i] - open_[i]) / rng

  return out
