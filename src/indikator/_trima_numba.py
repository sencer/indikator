"""Numba-optimized Triangular Moving Average (TRIMA) calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _sma_seq(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
  """Sequential SMA for internal use."""
  n = len(data)
  out = np.full(n, np.nan)

  if n < period:
    return out

  # Initial sum
  curr_sum = 0.0
  for i in range(period):
    val = data[i]
    if np.isnan(val):
      curr_sum = np.nan
    else:
      curr_sum += val

  out[period - 1] = curr_sum / period

  # Rolling
  for i in range(period, n):
    old = data[i - period]
    new = data[i]

    if np.isnan(curr_sum) or np.isnan(old) or np.isnan(new):
      # Re-sum to be safe or propagate NaN?
      # Standard SMA propagates NaN if any element in window is NaN.
      # If we have NaNs, we must be careful.
      # Re-summing at each step handles NaNs correctly (if window has NaN, sum is NaN).
      # But it is O(N*Period).
      # Optimized rolling handles it if we propagate NaNs logic carefully.
      # Here we can just assume propagation.
      curr_sum += new - old
    else:
      curr_sum += new - old

    out[i] = curr_sum / period

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_trima_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate Triangular Moving Average (TRIMA)."""
  n = len(prices)
  if n == 0 or period <= 0:
    return np.full(n, np.nan)

  # Determine periods
  if period % 2 == 1:
    p1 = (period + 1) // 2
    p2 = p1
  else:
    p1 = period // 2
    p2 = p1 + 1

  # 1. First Pass
  # We use our own seq logic which is safer than calling parallel kernel
  sma1 = _sma_seq(prices, p1)

  # 2. Second Pass
  # Only sma1 values starting from p1-1 are valid (non-NaN).
  # But _sma_seq propagates NaN correctly?
  # If sma1 has NaNs at start, _sma_seq rolling logic:
  # Window 0..p2-1 includes NaNs from sma1. Sum will be NaN.
  # Correct.
  # So we don't need slicing if _sma_seq handles NaN propagation naturally!
  # But rolling sum `curr_sum += new - old` fails if `old` is NaN.
  # A simple `curr_sum` tracking NaNs?
  # If `curr_sum` is NaN, can it recover? No.
  # Once a NaN enters the window, sum is NaN.
  # When NaN leaves the window, sum SHOULD become valid (if rest are valid).
  # But `NaN - NaN` is NaN. So it never recovers.
  # So rolling sum fails to recover from NaNs.

  # We MUST slice to give valid data to the second pass.
  # Or implement a smarter rolling sum that re-calculates if needed.
  # Slicing is faster/easier for this structure.

  valid_start = p1 - 1
  if valid_start >= n:
    return np.full(n, np.nan)

  # Slice valid portion
  valid_sma1 = sma1[valid_start:]
  # Ensure contiguous just in case
  valid_sma1 = np.ascontiguousarray(valid_sma1)

  if len(valid_sma1) < p2:
    return np.full(n, np.nan)

  sma2_part = _sma_seq(valid_sma1, p2)

  # Reconstruct
  trima = np.full(n, np.nan)
  trima[valid_start:] = sma2_part

  return trima
