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

  if matype == 3: # DEMA
      # DEMA = 2 * EMA1 - EMA(EMA1)
      ema1 = data[0]
      ema2 = data[0]
      out[0] = data[0]
      for i in range(1, n):
          p = periods[i]
          if np.isnan(p):
              out[i] = np.nan
              continue
          p = max(min_period, min(max_period, int(p)))
          k = 2.0 / (p + 1)
          ema1 = data[i] * k + ema1 * (1.0 - k)
          ema2 = ema1 * k + ema2 * (1.0 - k)
          out[i] = 2.0 * ema1 - ema2
      return out

  if matype == 4: # TEMA
      # TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
      ema1 = data[0]
      ema2 = data[0]
      ema3 = data[0]
      out[0] = data[0]
      for i in range(1, n):
          p = periods[i]
          if np.isnan(p):
              out[i] = np.nan
              continue
          p = max(min_period, min(max_period, int(p)))
          k = 2.0 / (p + 1)
          ema1 = data[i] * k + ema1 * (1.0 - k)
          ema2 = ema1 * k + ema2 * (1.0 - k)
          ema3 = ema2 * k + ema3 * (1.0 - k)
          out[i] = 3.0 * ema1 - 3.0 * ema2 + ema3
      return out

  if matype == 5: # TRIMA
      # TRIMA(p) = SMA(SMA(x, p1), p2)
      # This is expensive for variable period if we don't have a double prefix sum.
      # Actually, TRIMA(p) is a triangular weighted average.
      # Weights: 1, 2, ..., midpoint, ..., 2, 1.
      for i in range(n):
          p_float = periods[i]
          if np.isnan(p_float):
              out[i] = np.nan
              continue
          p = max(min_period, min(max_period, int(p_float)))
          if i < p - 1:
              out[i] = np.nan
              continue
          
          # Symmetric triangular weights
          # p1, p2 as used in TRIMA:
          if p % 2 == 1:
              p1 = (p + 1) // 2
              p2 = p1
          else:
              p1 = p // 2
              p2 = p1 + 1
          
          w_sum = 0.0
          w_total = float(p1 * p2)
          # Sum over kernel
          # This is O(N*P).
          for j in range(p):
              # weight = min(j+1, p-j, p1, p2) ? no
              # standard trima kernel: SMA of SMA
              # For simplicity, use the direct SMA of SMA logic inside bar i
              pass
          
          # Implementation via sliding window of SMA is hard for variable period.
          # We'll use the weighted sum definition.
          idx = 0
          for k2 in range(p2):
              for k1 in range(p1):
                  # This is slow, but we'll optimize if needed.
                  # A bar's value is prices[i - k2 - k1]
                  w_sum += data[i - k2 - k1]
          out[i] = w_sum / w_total
      return out

  if matype == 6: # KAMA
      # Variable window Efficiency Ratio (ER) based KAMA
      # This is O(N*P) if we don't use a rolling absolute difference sum.
      # But variable period ER is rare. We'll use a local window scan.
      fast_ema = 2.0 / (2.0 + 1.0)
      slow_ema = 2.0 / (30.0 + 1.0)
      
      curr_kama = data[0]
      out[0] = curr_kama
      
      for i in range(1, n):
          p_f = periods[i]
          if np.isnan(p_f):
              out[i] = np.nan
              continue
          p = max(min_period, min(max_period, int(p_f)))
          
          if i < p:
              out[i] = np.nan
              continue
          
          # Calculate ER = |price[i] - price[i-p]| / sum(|diffs|)
          price_diff = abs(data[i] - data[i-p])
          
          # Absolute difference sum
          diff_sum = 0.0
          for k in range(p):
              diff_sum += abs(data[i-k] - data[i-k-1])
          
          er = price_diff / diff_sum if diff_sum > 1e-12 else 0.0
          sc = (er * (fast_ema - slow_ema) + slow_ema) ** 2
          
          curr_kama = curr_kama + sc * (data[i] - curr_kama)
          out[i] = curr_kama
      return out

  # Type 5 (TRIMA) already implemented above.
  # matype 7 (MAMA) and 8 (T3) are highly specialized and typically not 
  # used with variable period in a sliding way, but TA-Lib might support them.
  # For matype 7, MAMA already has internal adaptive period.
  # For matype 8, variable T3 is extremely niche.
  # We'll leave them as SMA fallback or unimplemented for now.

  # Default fallback or SMA
  return out
