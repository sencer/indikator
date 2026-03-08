"""Numba-optimized intraday aggregation kernels.

These kernels provide fast time-of-day based aggregations using fixed-size
arrays indexed by time slot for O(1) lookups.
"""

from typing import Any

from numba import jit
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Max time slots per day = 86400 seconds. We'll use a fixed array.
MAX_TIME_SLOTS = 86400


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_intraday_mean_numba(
  values: NDArray[np.float64],
  time_keys: NDArray[np.int64],
  min_samples: int,
) -> NDArray[np.float64]:
  """Compute expanding mean by time-of-day with shift(1).

  For each row, returns the mean of all previous observations with the same
  time-of-day (not including the current observation).

  Args:
    values: Data values to aggregate
    time_keys: Integer time keys (seconds since midnight, 0-86399)
    min_samples: Minimum samples required before returning non-NaN

  Returns:
    Array of expanding means (shifted by 1)
  """
  n = len(values)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  # Fixed-size arrays: index by time_key (seconds since midnight)
  sums = np.zeros(MAX_TIME_SLOTS, dtype=np.float64)
  counts = np.zeros(MAX_TIME_SLOTS, dtype=np.int64)

  for i in range(n):
    t = time_keys[i]
    v = values[i]

    # Get current state for this time slot (before adding this value)
    cnt = counts[t]
    if cnt >= min_samples:
      out[i] = sums[t] / cnt
    # else: out[i] stays NaN

    # Update state with current value (for future lookups)
    if not np.isnan(v):
      sums[t] += v
      counts[t] += 1

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_intraday_mean_std_numba(
  values: NDArray[np.float64],
  time_keys: NDArray[np.int64],
  min_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Compute expanding mean and std by time-of-day with shift(1).

  Uses Welford's online algorithm for numerically stable variance.

  Args:
    values: Data values to aggregate
    time_keys: Integer time keys (seconds since midnight, 0-86399)
    min_samples: Minimum samples required before returning non-NaN

  Returns:
    Tuple of (means, stds) arrays
  """
  n = len(values)
  means_out = np.empty(n, dtype=np.float64)
  means_out[:] = np.nan
  stds_out = np.empty(n, dtype=np.float64)
  stds_out[:] = np.nan

  # Welford's algorithm state: count, mean, M2 (sum of squared differences)
  welf_count = np.zeros(MAX_TIME_SLOTS, dtype=np.int64)
  welf_mean = np.zeros(MAX_TIME_SLOTS, dtype=np.float64)
  welf_m2 = np.zeros(MAX_TIME_SLOTS, dtype=np.float64)

  for i in range(n):
    t = time_keys[i]
    v = values[i]

    # Return state BEFORE adding current value (shift by 1)
    cnt = welf_count[t]
    if cnt >= min_samples:
      means_out[i] = welf_mean[t]
      if cnt > 1:
        stds_out[i] = np.sqrt(welf_m2[t] / (cnt - 1))
      else:
        stds_out[i] = 0.0

    # Update Welford state with current value
    if not np.isnan(v):
      cnt = welf_count[t] + 1
      delta = v - welf_mean[t]
      new_mean = welf_mean[t] + delta / cnt
      delta2 = v - new_mean
      new_m2 = welf_m2[t] + delta * delta2

      welf_count[t] = cnt
      welf_mean[t] = new_mean
      welf_m2[t] = new_m2

  return means_out, stds_out


def time_to_key(dt_index: pd.DatetimeIndex | NDArray[Any]) -> NDArray[np.int64]:
  """Convert DatetimeIndex to integer time keys (seconds since midnight).

  This is a helper that runs in Python (not Numba) to prepare the time keys.
  Uses fast numpy view + modulo instead of property access for speed.
  """

  if isinstance(dt_index, pd.DatetimeIndex):
    # Fast path: use underlying int64 nanoseconds and modulo
    nanos = dt_index.view(np.int64)
    ns_per_day = 86400 * 10**9  # nanoseconds per day
    tod_nanos = nanos % ns_per_day
    return (tod_nanos // 10**9).astype(np.int64)
  # Assume numpy datetime64
  dt_index = np.asarray(dt_index, dtype="datetime64[s]")
  dates = dt_index.astype("datetime64[D]")
  return ((dt_index - dates) / np.timedelta64(1, "s")).astype(np.int64)
