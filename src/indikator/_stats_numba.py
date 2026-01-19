"""Numba-optimized statistical calculations (STDDEV, VAR)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_stddev_numba(
  data: NDArray[np.float64],
  period: int,
  nbdev: float,
) -> NDArray[np.float64]:
  """Calculate rolling standard deviation using Welford's online algorithm.

  Uses population std (ddof=0) to match TA-Lib.
  """
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # Initialize with first window
  mean = 0.0
  m2 = 0.0
  for i in range(period):
    delta = data[i] - mean
    mean += delta / (i + 1)
    delta2 = data[i] - mean
    m2 += delta * delta2

  # First valid output
  variance = m2 / period
  out[period - 1] = np.sqrt(variance) * nbdev

  # Rolling using incremental update
  for i in range(period, n):
    old_val = data[i - period]
    new_val = data[i]

    # Remove old value from stats
    old_mean = mean
    mean = old_mean + (new_val - old_val) / period
    # Update M2:
    # m2_new = m2 - (old - old_mean)(old - mean) + (new - old_mean)(new - mean)
    m2 = (
      m2
      - (old_val - old_mean) * (old_val - mean)
      + (new_val - old_mean) * (new_val - mean)
    )

    variance = m2 / period
    # Handle numerical issues
    if variance < 0.0:
      variance = 0.0
    out[i] = np.sqrt(variance) * nbdev

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_var_numba(
  data: NDArray[np.float64],
  period: int,
  nbdev: float,
) -> NDArray[np.float64]:
  """Calculate rolling variance using Welford's online algorithm.

  Uses population variance (ddof=0) to match TA-Lib.
  """
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # Initialize with first window
  mean = 0.0
  m2 = 0.0
  for i in range(period):
    delta = data[i] - mean
    mean += delta / (i + 1)
    delta2 = data[i] - mean
    m2 += delta * delta2

  # First valid output
  variance = m2 / period
  out[period - 1] = variance * nbdev

  # Rolling using incremental update
  for i in range(period, n):
    old_val = data[i - period]
    new_val = data[i]

    # Remove old value from stats
    old_mean = mean
    mean = old_mean + (new_val - old_val) / period
    m2 = (
      m2
      - (old_val - old_mean) * (old_val - mean)
      + (new_val - old_mean) * (new_val - mean)
    )

    variance = m2 / period
    if variance < 0.0:
      variance = 0.0
    out[i] = variance * nbdev

  return out
