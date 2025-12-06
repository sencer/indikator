"""Numba-optimized OBV calculation.

This module contains JIT-compiled functions for OBV calculation.
Separated for better code organization and testability.
"""
# pyright: reportAny=false

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator]  # pragma: no cover
def compute_obv_numba(
  closes: NDArray[np.float64],
  volumes: NDArray[np.float64],
) -> NDArray[np.float64]:
  """Numba JIT-compiled OBV calculation.

  On-Balance Volume (OBV) is a cumulative indicator that adds volume on up
  days and subtracts volume on down days.

  Formula:
  - If close > prev_close: OBV = OBV_prev + volume
  - If close < prev_close: OBV = OBV_prev - volume
  - If close == prev_close: OBV = OBV_prev

  Args:
    closes: Array of closing prices
    volumes: Array of volumes

  Returns:
    Array of OBV values
  """
  n = len(closes)
  obv = np.zeros(n, dtype=np.float64)

  if n == 0:
    return obv

  # First bar: OBV = volume (no previous price to compare)
  obv[0] = volumes[0]

  # Subsequent bars: add/subtract based on price direction
  for i in range(1, n):
    if closes[i] > closes[i - 1]:
      obv[i] = obv[i - 1] + volumes[i]
    elif closes[i] < closes[i - 1]:
      obv[i] = obv[i - 1] - volumes[i]
    else:
      obv[i] = obv[i - 1]

  return obv
