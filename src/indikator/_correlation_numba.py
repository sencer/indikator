"""Numba-optimized correlation/correlation calculations.

This module contains JIT-compiled functions for:
- BETA: Rolling beta coefficient β = cov(X,Y) / var(X)
- CORREL: Rolling Pearson correlation r = cov(X,Y) / (std_X * std_Y)

All implementations use O(1) rolling updates for optimal performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Constants
EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_beta_numba(
  x: NDArray[np.float64],
  y: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Compute rolling BETA: sensitivity of Y to X.

  BETA = cov(X, Y) / var(X)

  Uses O(1) rolling update by maintaining:
  - sum_x, sum_y (for means)
  - sum_xx (for variance of X)
  - sum_xy (for covariance)

  Args:
    x: Independent variable (e.g., market returns)
    y: Dependent variable (e.g., stock returns)
    period: Rolling window size

  Returns:
    Array of beta values
  """
  n = len(x)

  if n < period or period < 2:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(period - 1):
    out[i] = np.nan

  inv_period = 1.0 / period

  # Initialize sums for first window
  sum_x = 0.0
  sum_y = 0.0
  sum_xx = 0.0
  sum_xy = 0.0

  for i in range(period):
    xi = x[i]
    yi = y[i]
    sum_x += xi
    sum_y += yi
    sum_xx += xi * xi
    sum_xy += xi * yi

  # First beta
  mean_x = sum_x * inv_period
  var_x = sum_xx * inv_period - mean_x * mean_x
  mean_y = sum_y * inv_period
  cov_xy = sum_xy * inv_period - mean_x * mean_y

  if var_x > EPSILON:
    out[period - 1] = cov_xy / var_x
  else:
    out[period - 1] = np.nan

  # Main loop with O(1) update
  for i in range(period, n):
    old_x = x[i - period]
    old_y = y[i - period]
    new_x = x[i]
    new_y = y[i]

    sum_x = sum_x - old_x + new_x
    sum_y = sum_y - old_y + new_y
    sum_xx = sum_xx - old_x * old_x + new_x * new_x
    sum_xy = sum_xy - old_x * old_y + new_x * new_y

    mean_x = sum_x * inv_period
    var_x = sum_xx * inv_period - mean_x * mean_x
    mean_y = sum_y * inv_period
    cov_xy = sum_xy * inv_period - mean_x * mean_y

    if var_x > EPSILON:
      out[i] = cov_xy / var_x
    else:
      out[i] = np.nan

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_correl_numba(
  x: NDArray[np.float64],
  y: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Compute rolling CORREL: Pearson correlation coefficient.

  CORREL = cov(X, Y) / (std_X * std_Y)

  Uses O(1) rolling update by maintaining:
  - sum_x, sum_y (for means)
  - sum_xx, sum_yy (for variances)
  - sum_xy (for covariance)

  Args:
    x: First variable
    y: Second variable
    period: Rolling window size

  Returns:
    Array of correlation values (-1 to 1)
  """
  n = len(x)

  if n < period or period < 2:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  for i in range(period - 1):
    out[i] = np.nan

  inv_period = 1.0 / period

  # Initialize sums for first window
  sum_x = 0.0
  sum_y = 0.0
  sum_xx = 0.0
  sum_yy = 0.0
  sum_xy = 0.0

  for i in range(period):
    xi = x[i]
    yi = y[i]
    sum_x += xi
    sum_y += yi
    sum_xx += xi * xi
    sum_yy += yi * yi
    sum_xy += xi * yi

  # First correlation
  mean_x = sum_x * inv_period
  mean_y = sum_y * inv_period
  var_x = sum_xx * inv_period - mean_x * mean_x
  var_y = sum_yy * inv_period - mean_y * mean_y
  cov_xy = sum_xy * inv_period - mean_x * mean_y

  denom = np.sqrt(var_x * var_y)
  if denom > EPSILON:
    out[period - 1] = cov_xy / denom
  else:
    out[period - 1] = np.nan

  # Main loop with O(1) update
  for i in range(period, n):
    old_x = x[i - period]
    old_y = y[i - period]
    new_x = x[i]
    new_y = y[i]

    sum_x = sum_x - old_x + new_x
    sum_y = sum_y - old_y + new_y
    sum_xx = sum_xx - old_x * old_x + new_x * new_x
    sum_yy = sum_yy - old_y * old_y + new_y * new_y
    sum_xy = sum_xy - old_x * old_y + new_x * new_y

    mean_x = sum_x * inv_period
    mean_y = sum_y * inv_period
    var_x = sum_xx * inv_period - mean_x * mean_x
    var_y = sum_yy * inv_period - mean_y * mean_y
    cov_xy = sum_xy * inv_period - mean_x * mean_y

    denom = np.sqrt(var_x * var_y)
    if denom > EPSILON:
      out[i] = cov_xy / denom
    else:
      out[i] = np.nan

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_beta_fused_rocp_numba(
  x: NDArray[np.float64],
  y: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Compute rolling BETA with FUSED ROCP calculation.

  This kernel combines:
  1. ROCP calculation: (price - prev) / prev
  2. Rolling Beta calculation: cov(rocp_x, rocp_y) / var(rocp_x)

  This avoids allocating intermediate arrays for ROCP, improving performance.

  Note on Lookback:
  - ROCP requires 1 period lookback.
  - Beta requires 'period' window of valid ROCP values.
  - Total lookback = period + 1 prices.
  - The first period + 1 values will be NaN.

  Args:
    x: Independent variable price series
    y: Dependent variable price series
    period: Rolling window size for Beta
  """
  n = len(x)
  total_lookback = period + 1

  if n < total_lookback:
    return np.full(n, np.nan)

  out = np.empty(n, dtype=np.float64)

  # Initialize NaNs
  for i in range(total_lookback - 1):
    out[i] = np.nan

  inv_period = 1.0 / period

  # Initialize sums for first valid window
  # The window starts at index 1 (since index 0 is used for first ROCP at index 1)
  # We iterate from i=1 to i=period.
  sum_x = 0.0
  sum_y = 0.0
  sum_xx = 0.0
  sum_xy = 0.0

  # Pre-calculate first window sums
  # We need ROCP values from index 1 to period (inclusive? no, size is period).
  # Indices of prices: 0, 1, 2, ..., period, period+1, ...
  # ROCP indices: -, 1, 2, ..., period
  # Rolling window 1 covers ROCP indices [1, period]. Total count = period.

  for i in range(1, period + 1):
    # Calculate ROCP on the fly
    prev_x = x[i - 1]
    prev_y = y[i - 1]

    # Avoid div by zero
    if prev_x != 0.0:
      val_x = (x[i] - prev_x) / prev_x
    else:
      val_x = 0.0

    if prev_y != 0.0:
      val_y = (y[i] - prev_y) / prev_y
    else:
      val_y = 0.0

    sum_x += val_x
    sum_y += val_y
    sum_xx += val_x * val_x
    sum_xy += val_x * val_y

  # First beta at index 'period' (total lookback required is period+1 prices,
  # so index 'period' is the (period+1)-th element, which is correct)
  # Wait, if period=5. Total lookback=6. NaNs should be indices 0,1,2,3,4,5?
  # Let's trace:
  # Prices: p0, p1, p2, p3, p4, p5, p6
  # ROCP:   -,  r1, r2, r3, r4, r5, r6
  # Beta(5): -,  -,  -,  -,  -,  b5, b6
  # b5 uses r1..r5.
  # So yes, calculation at index 'period' (which is 5) is correct.

  mean_x = sum_x * inv_period
  mean_y = sum_y * inv_period
  var_x = sum_xx * inv_period - mean_x * mean_x
  cov_xy = sum_xy * inv_period - mean_x * mean_y

  if var_x > EPSILON:
    out[period] = cov_xy / var_x
  else:
    out[period] = np.nan

  # Main loop
  # We start calculating for index period+1
  # Window slides from [1, period] to [2, period+1]
  # Remove index 1, Add index period+1
  for i in range(period + 1, n):
    # Remove old (index i - period)
    # Since window length is 'period'
    # Start of previous window was at index (i-1) - period + 1 = i - period
    remove_idx = i - period

    # Re-calculate ROCP at remove_idx
    prev_x_rem = x[remove_idx - 1]
    prev_y_rem = y[remove_idx - 1]

    if prev_x_rem != 0.0:
      rem_x = (x[remove_idx] - prev_x_rem) / prev_x_rem
    else:
      rem_x = 0.0

    if prev_y_rem != 0.0:
      rem_y = (y[remove_idx] - prev_y_rem) / prev_y_rem
    else:
      rem_y = 0.0

    # Add new (index i)
    # Re-calculate ROCP at index i
    prev_x_new = x[i - 1]
    prev_y_new = y[i - 1]

    if prev_x_new != 0.0:
      add_x = (x[i] - prev_x_new) / prev_x_new
    else:
      add_x = 0.0

    if prev_y_new != 0.0:
      add_y = (y[i] - prev_y_new) / prev_y_new
    else:
      add_y = 0.0

    sum_x = sum_x - rem_x + add_x
    sum_y = sum_y - rem_y + add_y
    sum_xx = sum_xx - rem_x * rem_x + add_x * add_x
    sum_xy = sum_xy - rem_x * rem_y + add_x * add_y

    mean_x = sum_x * inv_period
    mean_y = sum_y * inv_period
    var_x = sum_xx * inv_period - mean_x * mean_x
    cov_xy = sum_xy * inv_period - mean_x * mean_y

    if var_x > EPSILON:
      out[i] = cov_xy / var_x
    else:
      out[i] = np.nan

  return out
