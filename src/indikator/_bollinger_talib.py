"""Numba-optimized Bollinger Bands calculation - TA-lib compatible version.

This version uses population std (ddof=0) to match TA-lib exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_bollinger_talib(
  data: NDArray[np.float64], window: int, num_std: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Calculate Bollinger Bands using TA-lib compatible algorithm.

  Uses population standard deviation (ddof=0) to match TA-lib.
  Only computes the three bands (upper, middle, lower) for maximum speed.

  Args:
    data: Input price array
    window: Rolling window size (default: 20)
    num_std: Number of standard deviations for bands (default: 2.0)

  Returns:
    Tuple of (upper, middle, lower) arrays
  """
  n = len(data)

  if n < window:
    nans = np.full(n, np.nan, dtype=np.float64)
    return nans, nans.copy(), nans.copy()

  # Allocate arrays
  middle = np.empty(n, dtype=np.float64)
  upper = np.empty(n, dtype=np.float64)
  lower = np.empty(n, dtype=np.float64)

  # Fill initial NaNs
  for i in range(window - 1):
    middle[i] = np.nan
    upper[i] = np.nan
    lower[i] = np.nan

  # Initialize first window
  current_sum = 0.0
  current_sum_sq = 0.0

  for i in range(window):
    val = data[i]
    current_sum += val
    current_sum_sq += val * val

  # Constants
  inv_window = 1.0 / window

  # First result (at index window-1)
  idx = window - 1
  mean = current_sum * inv_window
  # Population variance: var = E[X^2] - E[X]^2
  var_num = current_sum_sq * inv_window - mean * mean
  var_num = max(var_num, 0.0)
  std_dev = np.sqrt(var_num)

  middle[idx] = mean
  band_offset = std_dev * num_std
  upper[idx] = mean + band_offset
  lower[idx] = mean - band_offset

  # Rolling Loop - optimized
  for i in range(window, n):
    out_val = data[i - window]
    in_val = data[i]

    current_sum += in_val - out_val
    current_sum_sq += in_val * in_val - out_val * out_val

    mean = current_sum * inv_window
    # Population variance
    var_num = current_sum_sq * inv_window - mean * mean
    var_num = max(var_num, 0.0)
    std_dev = np.sqrt(var_num)

    middle[i] = mean
    band_offset = std_dev * num_std
    upper[i] = mean + band_offset
    lower[i] = mean - band_offset

  return upper, middle, lower
