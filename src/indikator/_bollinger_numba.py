"""Numba-optimized Bollinger Bands calculation."""
# ruff: noqa: PLR2004  # Magic values are acceptable in Numba performance code

from __future__ import annotations

from numba import jit, prange
import numpy as np


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_bollinger_numba_fast(
  data: np.ndarray, window: int, num_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Calculate Bollinger Bands using Numba (Parallel Chunked).

  Uses parallel chunks for O(N/cores) throughput.
  """
  n = len(data)

  # Allocate arrays
  middle = np.full(n, np.nan, dtype=np.float64)
  upper = np.full(n, np.nan, dtype=np.float64)
  lower = np.full(n, np.nan, dtype=np.float64)
  bandwidth = np.full(n, np.nan, dtype=np.float64)
  percent_b = np.full(n, np.nan, dtype=np.float64)

  if n < window:
    return middle, upper, lower, bandwidth, percent_b

  # Constants
  inv_window = 1.0 / window
  inv_window_minus_1 = 1.0 / (window - 1) if window > 1 else 0.0

  num_chunks = 16
  chunk_size = (n - window) // num_chunks
  if chunk_size < 1:
    chunk_size = n - window
    num_chunks = 1

  # Parallel Loop
  for c in prange(num_chunks + 1):
    idx_start = window + c * chunk_size
    idx_end = window + (c + 1) * chunk_size
    if c == num_chunks:
      idx_end = n
    if idx_start >= n:
      continue

    # For chunk 0, handle initial calculation at `window - 1`
    if c == 0:
      init_idx = window - 1
      sum_0 = 0.0
      sum_sq_0 = 0.0
      for k in range(window):
        val = data[k]
        sum_0 += val
        sum_sq_0 += val * val

      mean_0 = sum_0 * inv_window
      var_0 = sum_sq_0 - sum_0 * sum_0 * inv_window
      var_0 = max(var_0, 0.0)
      std_0 = np.sqrt(var_0 * inv_window_minus_1)

      middle[init_idx] = mean_0
      upper[init_idx] = mean_0 + std_0 * num_std
      lower[init_idx] = mean_0 - std_0 * num_std
      # Init bandwidth/%b
      up, low = upper[init_idx], lower[init_idx]
      abs_m = abs(mean_0)
      rng = up - low
      bandwidth[init_idx] = rng / abs_m if abs_m > 1e-9 else np.nan
      percent_b[init_idx] = (data[init_idx] - low) / rng if rng > 1e-9 else np.nan

    # Init state for main chunk loop
    current_sum = 0.0
    current_sum_sq = 0.0
    for k in range(idx_start - window + 1, idx_start + 1):
      val = data[k]
      current_sum += val
      current_sum_sq += val * val

    # Calculate at start
    mean = current_sum * inv_window
    var = current_sum_sq - current_sum * current_sum * inv_window
    var = max(var, 0.0)
    std = np.sqrt(var * inv_window_minus_1)

    middle[idx_start] = mean
    up = mean + std * num_std
    low = mean - std * num_std
    upper[idx_start] = up
    lower[idx_start] = low

    abs_mid = abs(mean)
    rng = up - low
    bandwidth[idx_start] = rng / abs_mid if abs_mid > 1e-9 else 0.0
    percent_b[idx_start] = (data[idx_start] - low) / rng if rng > 1e-9 else 0.5

    # Loop
    for i in range(idx_start + 1, idx_end):
      out_val = data[i - window]
      in_val = data[i]

      current_sum += in_val - out_val
      current_sum_sq += in_val * in_val - out_val * out_val

      mean = current_sum * inv_window
      var = current_sum_sq - current_sum * current_sum * inv_window
      var = max(var, 0.0)
      std = np.sqrt(var * inv_window_minus_1)

      middle[i] = mean
      up = mean + std * num_std
      low = mean - std * num_std
      upper[i] = up
      lower[i] = low

      abs_mid = abs(mean)
      rng = up - low
      bandwidth[i] = rng / abs_mid if abs_mid > 1e-9 else 0.0
      percent_b[i] = (in_val - low) / rng if rng > 1e-9 else 0.5

  return middle, upper, lower, bandwidth, percent_b


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_bollinger_basic_numba(
  data: np.ndarray, window: int, num_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Calculate Bollinger Bands (Upper, Middle, Lower only) using Numba (Parallel Chunked).

  Optimized workload matching TA-Lib.
  """
  n = len(data)

  # Allocate 3 arrays
  middle = np.full(n, np.nan, dtype=np.float64)
  upper = np.full(n, np.nan, dtype=np.float64)
  lower = np.full(n, np.nan, dtype=np.float64)

  if n < window:
    return middle, upper, lower

  # Constants - use population std (ddof=0) to match TA-lib
  inv_window = 1.0 / window

  num_chunks = 16
  chunk_size = (n - window) // num_chunks
  if chunk_size < 1:
    chunk_size = n - window
    num_chunks = 1

  # Parallel Loop
  for c in prange(num_chunks + 1):
    idx_start = window + c * chunk_size
    idx_end = window + (c + 1) * chunk_size
    if c == num_chunks:
      idx_end = n
    if idx_start >= n:
      continue

    if c == 0:
      init_idx = window - 1
      sum_0 = 0.0
      sum_sq_0 = 0.0
      for k in range(window):
        val = data[k]
        sum_0 += val
        sum_sq_0 += val * val

      mean_0 = sum_0 * inv_window
      # Population variance: var = E[X^2] - E[X]^2
      var_0 = sum_sq_0 * inv_window - mean_0 * mean_0
      var_0 = max(var_0, 0.0)
      std_0 = np.sqrt(var_0)

      middle[init_idx] = mean_0
      upper[init_idx] = mean_0 + std_0 * num_std
      lower[init_idx] = mean_0 - std_0 * num_std

    # Init state
    current_sum = 0.0
    current_sum_sq = 0.0
    for k in range(idx_start - window + 1, idx_start + 1):
      val = data[k]
      current_sum += val
      current_sum_sq += val * val

    mean = current_sum * inv_window
    # Population variance
    var = current_sum_sq * inv_window - mean * mean
    var = max(var, 0.0)
    std = np.sqrt(var)

    middle[idx_start] = mean
    upper[idx_start] = mean + std * num_std
    lower[idx_start] = mean - std * num_std

    # Loop
    for i in range(idx_start + 1, idx_end):
      out_val = data[i - window]
      in_val = data[i]

      current_sum += in_val - out_val
      current_sum_sq += in_val * in_val - out_val * out_val

      mean = current_sum * inv_window
      # Population variance
      var = current_sum_sq * inv_window - mean * mean
      var = max(var, 0.0)
      std = np.sqrt(var)

      middle[i] = mean
      upper[i] = mean + std * num_std
      lower[i] = mean - std * num_std

  return middle, upper, lower


# Keep Safe Mode as is
@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_bollinger_numba(
  data: np.ndarray, window: int, num_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Calculate Bollinger Bands using Numba (Safe Mode)."""
  # ... Copy of original Safe Mode impl ...
  n = len(data)

  middle = np.full(n, np.nan, dtype=np.float64)
  upper = np.full(n, np.nan, dtype=np.float64)
  lower = np.full(n, np.nan, dtype=np.float64)
  bandwidth = np.full(n, np.nan, dtype=np.float64)
  percent_b = np.full(n, np.nan, dtype=np.float64)

  if n < window:
    return middle, upper, lower, bandwidth, percent_b

  current_sum = 0.0
  current_sum_sq = 0.0
  count = 0

  # 1. Growth Phase
  limit = min(window, n)
  for i in range(limit):
    val_new = data[i]
    if not np.isnan(val_new):
      current_sum += val_new
      current_sum_sq += val_new * val_new
      count += 1

    if count >= 1:
      mean = current_sum / count
      if count > 1:
        variance_numerator = current_sum_sq - (current_sum * current_sum) / count
        if variance_numerator < 0:
          variance_numerator = 0.0
        std_dev = np.sqrt(variance_numerator / (count - 1))
      else:
        std_dev = np.nan

      middle[i] = mean
      if not np.isnan(std_dev):
        upper[i] = mean + (std_dev * num_std)
        lower[i] = mean - (std_dev * num_std)
        if mean != 0:
          bandwidth[i] = (upper[i] - lower[i]) / np.abs(mean)
        rng = upper[i] - lower[i]
        if rng != 0:
          percent_b[i] = (val_new - lower[i]) / rng

  # 2. Steady Phase
  for i in range(window, n):
    val_new = data[i]
    val_out = data[i - window]

    if not np.isnan(val_new):
      current_sum += val_new
      current_sum_sq += val_new * val_new
      count += 1

    if not np.isnan(val_out):
      current_sum -= val_out
      current_sum_sq -= val_out * val_out
      count -= 1

    if count >= 1:
      mean = current_sum / count
      if count > 1:
        variance_numerator = current_sum_sq - (current_sum * current_sum) / count
        if variance_numerator < 0:
          variance_numerator = 0.0
        std_dev = np.sqrt(variance_numerator / (count - 1))
      else:
        std_dev = np.nan

      middle[i] = mean
      if not np.isnan(std_dev):
        up = mean + (std_dev * num_std)
        low = mean - (std_dev * num_std)
        upper[i] = up
        lower[i] = low

        abs_mid = np.abs(mean)
        if abs_mid > 1e-9:
          bandwidth[i] = (up - low) / abs_mid

        band_range = up - low
        if band_range > 1e-9:
          percent_b[i] = (val_new - low) / band_range

  return middle, upper, lower, bandwidth, percent_b
