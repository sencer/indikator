"""Numba-optimized MAVP (Moving Average with Variable Period) calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)
def compute_mavp_sma_numba(
  data: NDArray[np.float64],
  periods: NDArray[np.float64],
  min_period: int,
  max_period: int,
) -> NDArray[np.float64]:
  """Calculate MAVP using SMA logic with Prefix Sum optimization (O(N)).

  Uses a cumulative sum array to calculate the sum of any variable window
  in constant time.

  Args:
    data: Input data array.
    periods: Array of periods for each element.
    min_period: Minimum allowed period.
    max_period: Maximum allowed period.

  Returns:
    Array of MAVP values.
  """
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  # Compute prefix sum (cumsum)
  # prefix_sum[i] = sum(data[0]...data[i]) (inclusive)
  # So sum(i-p+1 ... i) = prefix_sum[i] - prefix_sum[i-p]
  # We pad properly to handle range.

  # Using cumsum from numpy might be fast, but explicit loop is safer for Numba
  # and avoids allocation if we do it carefully?
  # Actually allocating one array for cumsum is fine.

  # prefix_sum[0] = 0
  # prefix_sum[k] = sum(data[0]...data[k-1])
  # Length n+1
  prefix_sum = np.empty(n + 1, dtype=np.float64)
  prefix_sum[0] = 0.0
  current_sum = 0.0

  for i in range(n):
    current_sum += data[i]
    prefix_sum[i + 1] = current_sum

  for i in range(n):
    # Get period
    p_float = periods[i]
    if np.isnan(p_float):
      out[i] = np.nan
      continue

    p = int(p_float)

    # Clamp period
    if p < min_period:
      p = min_period
    elif p > max_period:
      p = max_period

    # Validate window availability
    # We need p elements ending at i.
    # Start index = i - p + 1.
    # Check if Start index >= 0.
    if i - p + 1 < 0:
      out[i] = np.nan
    else:
      # efficient sum using prefix_sum
      # sum(data[Start ... i])
      # prefix_sum[i+1] captures sum(0..i)
      # prefix_sum[Start] captures sum(0..Start-1)
      # Range Sum = prefix_sum[i+1] - prefix_sum[Start]
      # Start = i - p + 1
      start_idx = i - p + 1
      window_sum = prefix_sum[i + 1] - prefix_sum[start_idx]
      out[i] = window_sum / p

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_mavp_general_numba(
  data: NDArray[np.float64],
  periods: NDArray[np.float64],
  min_period: int,
  max_period: int,
  matype: int,
) -> NDArray[np.float64]:
  """General MAVP calculation for recursive and window-based types."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  
  if matype == 1: # EMA
      # Variable Alpha EMA: y_t = alpha_t * x_t + (1 - alpha_t) * y_{t-1}
      curr_ema = data[0]
      out[0] = curr_ema
      for i in range(1, n):
          p = periods[i]
          if np.isnan(p):
              out[i] = np.nan
              continue
          p = max(min_period, min(max_period, int(p)))
          k = 2.0 / (p + 1)
          curr_ema = data[i] * k + curr_ema * (1.0 - k)
          out[i] = curr_ema
      return out

  if matype == 2: # WMA (Window based)
      # Must rescan window of size p_i each time. O(N*P).
      for i in range(n):
          p_float = periods[i]
          if np.isnan(p_float):
              out[i] = np.nan
              continue
          p = max(min_period, min(max_period, int(p_float)))
          if i < p - 1:
              out[i] = np.nan
              continue
          
          # WMA calculation
          w_sum = 0.0
          w_denom = p * (p + 1) * 0.5
          for j in range(p):
              w_sum += data[i - j] * (p - j)
          out[i] = w_sum / w_denom
      return out

  # Default fallback or SMA if type not specifically optimized
  # (Currently only SMA, EMA, WMA implemented for variable period)
  # For others, we might need more complex logic.
  # TA-Lib supports all 9.
  return out # Should handle other types or return SMA
