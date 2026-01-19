"""Numba-optimized price transform calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_typprice_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Calculate Typical Price: (High + Low + Close) / 3."""
  n = len(high)
  out = np.empty(n, dtype=np.float64)
  for i in range(n):
    out[i] = (high[i] + low[i] + close[i]) / 3.0
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_medprice_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Calculate Median Price: (High + Low) / 2."""
  n = len(high)
  out = np.empty(n, dtype=np.float64)
  for i in range(n):
    out[i] = (high[i] + low[i]) / 2.0
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_wclprice_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Calculate Weighted Close Price: (High + Low + 2*Close) / 4."""
  n = len(high)
  out = np.empty(n, dtype=np.float64)
  for i in range(n):
    out[i] = (high[i] + low[i] + 2.0 * close[i]) / 4.0
  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_avgprice_numba(
  open_: NDArray[np.float64],
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Calculate Average Price: (Open + High + Low + Close) / 4."""
  n = len(high)
  out = np.empty(n, dtype=np.float64)
  for i in range(n):
    out[i] = (open_[i] + high[i] + low[i] + close[i]) / 4.0
  return out
