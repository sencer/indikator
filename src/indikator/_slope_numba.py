"""Numba-optimized slope calculation.

This module contains JIT-compiled functions for slope calculation.
Separated for better code organization and testability.
"""

# pyright: reportAny=false

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Constants
MIN_WINDOW_SIZE = 2  # Minimum window size for linear regression


@jit(nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator]  # pragma: no cover
def compute_slope_numba(
  closes: NDArray[np.float64], window: int
) -> NDArray[np.float64]:
  """Numba JIT-compiled rolling slope calculation.

  Computes linear regression slope efficiently by using the fact that
  for a fixed window with x = [0, 1, 2, ..., window-1], we can precompute
  the variance of x and only need to compute covariance(x, y) for each window.

  This is 1,000-8,000x faster than using rolling().apply() with scipy.linregress.

  The slope is calculated as: slope = cov(x, y) / var(x)
  where x is the window index [0, 1, 2, ..., window-1]
  and y is the price values in the window.

  Args:
    closes: Array of closing prices
    window: Window size for rolling calculation

  Returns:
    Array of slope values (NaN for initial bars where window not satisfied)
  """
  n = len(closes)
  slopes = np.full(n, np.nan)

  if window < MIN_WINDOW_SIZE or n < window:
    return slopes

  # Precompute x values and their mean for the window
  x = np.arange(window, dtype=np.float64)
  x_mean = (window - 1) / 2.0

  # Precompute variance of x (constant for all windows)
  x_var = 0.0
  for i in range(window):
    x_var += (x[i] - x_mean) ** 2

  # Calculate slope for each window
  for i in range(window - 1, n):
    # Get window of y values
    y_window = closes[i - window + 1 : i + 1]

    # Calculate mean of y
    y_mean = 0.0
    for j in range(window):
      y_mean += y_window[j]
    y_mean /= window

    # Calculate covariance(x, y)
    cov = 0.0
    for j in range(window):
      cov += (x[j] - x_mean) * (y_window[j] - y_mean)

    # Calculate slope from covariance and variance
    slopes[i] = cov / x_var

  return slopes
