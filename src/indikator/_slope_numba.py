"""Numba-optimized linear regression calculations.

This module contains JIT-compiled functions for linear regression indicators:
- SLOPE (existing)
- LINEARREG
- LINEARREG_INTERCEPT
- LINEARREG_ANGLE
- TSF (Time Series Forecast)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Constants
MIN_WINDOW_SIZE = 2  # Minimum window size for linear regression


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
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

  if window < MIN_WINDOW_SIZE or n < window:
    return np.full(n, np.nan)

  slopes = np.empty(n, dtype=np.float64)

  # Fill NaN for warmup
  for i in range(window - 1):
    slopes[i] = np.nan

  # Precompute variance of x (constant for all windows)
  # Sum_{j=0}^{W-1} (j - (W-1)/2)^2 = W(W^2 - 1) / 12
  x_var = (window * (window**2 - 1.0)) / 12.0
  x_mean = (window - 1.0) / 2.0

  # Initial sums for the first window
  sum_y = 0.0
  sum_xy = 0.0
  for i in range(window):
    y = closes[i]
    sum_y += y
    sum_xy += i * y

  # Calculate first slope
  slopes[window - 1] = (sum_xy - x_mean * sum_y) / x_var

  # Slide the window
  for i in range(window, n):
    leaving_y = closes[i - window]
    entering_y = closes[i]

    # Update sum_y
    sum_y = sum_y - leaving_y + entering_y

    # Update sum_xy
    sum_xy = sum_xy - (sum_y - entering_y) + (window - 1.0) * entering_y

    slopes[i] = (sum_xy - x_mean * sum_y) / x_var

  return slopes


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_linearreg_numba(
  closes: NDArray[np.float64], window: int
) -> NDArray[np.float64]:
  """Compute LINEARREG: the linear regression value at the end of the window.

  LINEARREG = intercept + slope * (window - 1)

  Uses O(1) rolling update for efficiency.
  """
  n = len(closes)

  if window < MIN_WINDOW_SIZE or n < window:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(window - 1):
    out[i] = np.nan

  x_var = (window * (window**2 - 1.0)) / 12.0
  x_mean = (window - 1.0) / 2.0
  inv_window = 1.0 / window

  sum_y = 0.0
  sum_xy = 0.0
  for i in range(window):
    y = closes[i]
    sum_y += y
    sum_xy += i * y

  slope = (sum_xy - x_mean * sum_y) / x_var
  y_mean = sum_y * inv_window
  intercept = y_mean - slope * x_mean
  out[window - 1] = intercept + slope * (window - 1.0)

  for i in range(window, n):
    leaving_y = closes[i - window]
    entering_y = closes[i]

    sum_y = sum_y - leaving_y + entering_y
    sum_xy = sum_xy - (sum_y - entering_y) + (window - 1.0) * entering_y

    slope = (sum_xy - x_mean * sum_y) / x_var
    y_mean = sum_y * inv_window
    intercept = y_mean - slope * x_mean
    out[i] = intercept + slope * (window - 1.0)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_linearreg_intercept_numba(
  closes: NDArray[np.float64], window: int
) -> NDArray[np.float64]:
  """Compute LINEARREG_INTERCEPT: the y-intercept of the regression line.

  intercept = mean_y - slope * mean_x

  Uses O(1) rolling update for efficiency.
  """
  n = len(closes)

  if window < MIN_WINDOW_SIZE or n < window:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(window - 1):
    out[i] = np.nan

  x_var = (window * (window**2 - 1.0)) / 12.0
  x_mean = (window - 1.0) / 2.0
  inv_window = 1.0 / window

  sum_y = 0.0
  sum_xy = 0.0
  for i in range(window):
    y = closes[i]
    sum_y += y
    sum_xy += i * y

  slope = (sum_xy - x_mean * sum_y) / x_var
  y_mean = sum_y * inv_window
  out[window - 1] = y_mean - slope * x_mean

  for i in range(window, n):
    leaving_y = closes[i - window]
    entering_y = closes[i]

    sum_y = sum_y - leaving_y + entering_y
    sum_xy = sum_xy - (sum_y - entering_y) + (window - 1.0) * entering_y

    slope = (sum_xy - x_mean * sum_y) / x_var
    y_mean = sum_y * inv_window
    out[i] = y_mean - slope * x_mean

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_linearreg_angle_numba(
  closes: NDArray[np.float64], window: int
) -> NDArray[np.float64]:
  """Compute LINEARREG_ANGLE: the angle of the regression line in degrees.

  angle = atan(slope) * 180 / pi

  Uses O(1) rolling update for efficiency.
  """
  n = len(closes)

  if window < MIN_WINDOW_SIZE or n < window:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(window - 1):
    out[i] = np.nan

  x_var = (window * (window**2 - 1.0)) / 12.0
  x_mean = (window - 1.0) / 2.0
  rad_to_deg = 180.0 / np.pi

  sum_y = 0.0
  sum_xy = 0.0
  for i in range(window):
    y = closes[i]
    sum_y += y
    sum_xy += i * y

  slope = (sum_xy - x_mean * sum_y) / x_var
  out[window - 1] = np.arctan(slope) * rad_to_deg

  for i in range(window, n):
    leaving_y = closes[i - window]
    entering_y = closes[i]

    sum_y = sum_y - leaving_y + entering_y
    sum_xy = sum_xy - (sum_y - entering_y) + (window - 1.0) * entering_y

    slope = (sum_xy - x_mean * sum_y) / x_var
    out[i] = np.arctan(slope) * rad_to_deg

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_tsf_numba(closes: NDArray[np.float64], window: int) -> NDArray[np.float64]:
  """Compute TSF (Time Series Forecast): LINEARREG projected 1 bar forward.

  TSF = intercept + slope * window

  Uses O(1) rolling update for efficiency.
  """
  n = len(closes)

  if window < MIN_WINDOW_SIZE or n < window:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(window - 1):
    out[i] = np.nan

  x_var = (window * (window**2 - 1.0)) / 12.0
  x_mean = (window - 1.0) / 2.0
  inv_window = 1.0 / window

  sum_y = 0.0
  sum_xy = 0.0
  for i in range(window):
    y = closes[i]
    sum_y += y
    sum_xy += i * y

  slope = (sum_xy - x_mean * sum_y) / x_var
  y_mean = sum_y * inv_window
  intercept = y_mean - slope * x_mean
  out[window - 1] = intercept + slope * window  # Project 1 bar forward

  for i in range(window, n):
    leaving_y = closes[i - window]
    entering_y = closes[i]

    sum_y = sum_y - leaving_y + entering_y
    sum_xy = sum_xy - (sum_y - entering_y) + (window - 1.0) * entering_y

    slope = (sum_xy - x_mean * sum_y) / x_var
    y_mean = sum_y * inv_window
    intercept = y_mean - slope * x_mean
    out[i] = intercept + slope * window

  return out
