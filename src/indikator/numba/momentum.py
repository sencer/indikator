"""Numba-optimized Advanced Momentum Indicators.

This module consolidates kernels for:
- WILLR: Williams %R
- CCI: Commodity Channel Index
- CMO: Chande Momentum Oscillator
- MFI: Money Flow Index

All implementations use Parallel Chunked strategy or optimized O(n) rolling algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Constants
EPSILON = 1e-10


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_willr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Williams %R using Parallel Chunked Lazy Rescan."""
  n = len(close)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    out[:] = np.nan
    return out

  # Adaptive parallelism
  total_len = n - period + 1
  num_chunks = 16
  if total_len < 2048:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1
  out[:warmup_idx] = np.nan

  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initial scan for the chunk [start_v - period + 1, start_v]
    h_val, l_val = -np.inf, np.inf
    h_idx, l_idx = -1, -1
    for k in range(start_v - period + 1, start_v + 1):
      if high[k] >= h_val:
        h_val, h_idx = high[k], k
      if low[k] <= l_val:
        l_val, l_idx = low[k], k

    # Initial calculation
    div = h_val - l_val
    out[start_v] = -100.0 * (h_val - close[start_v]) / div if div > EPSILON else 0.0

    # Rolling loop for chunk
    for i in range(start_v + 1, end_v):
      trailing_idx = i - period + 1

      # Lazy rescan for max
      if h_idx < trailing_idx:
        h_val = high[trailing_idx]
        h_idx = trailing_idx
        for k in range(trailing_idx + 1, i + 1):
          if high[k] >= h_val:
            h_val, h_idx = high[k], k
      elif high[i] >= h_val:
        h_val, h_idx = high[i], i

      # Lazy rescan for min
      if l_idx < trailing_idx:
        l_val = low[trailing_idx]
        l_idx = trailing_idx
        for k in range(trailing_idx + 1, i + 1):
          if low[k] <= l_val:
            l_val, l_idx = low[k], k
      elif low[i] <= l_val:
        l_val, l_idx = low[i], i

      div = h_val - l_val
      out[i] = -100.0 * (h_val - close[i]) / div if div > EPSILON else 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_cci_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
  constant: float = 0.015,
) -> NDArray[np.float64]:
  """Compute Commodity Channel Index (CCI) - Parallel Chunked."""
  n = len(close)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    out[:] = np.nan
    return out

  # Adaptive parallelism
  total_len = n - period + 1
  num_chunks = 16
  if total_len < 2048:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  inv_period = 1.0 / period
  warmup_idx = period - 1
  out[:warmup_idx] = np.nan

  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initial sum and MAD for this chunk
    curr_sum = 0.0
    for k in range(start_v - period + 1, start_v + 1):
      curr_sum += (high[k] + low[k] + close[k]) / 3.0

    # MAD loop for the first element in chunk
    curr_sma = curr_sum * inv_period
    sum_abs_diff = 0.0
    for k in range(start_v - period + 1, start_v + 1):
      tp_k = (high[k] + low[k] + close[k]) / 3.0
      sum_abs_diff += abs(tp_k - curr_sma)

    mean_dev = sum_abs_diff * inv_period
    tp_curr = (high[start_v] + low[start_v] + close[start_v]) / 3.0
    if mean_dev > EPSILON:
      out[start_v] = (tp_curr - curr_sma) / (constant * mean_dev)
    else:
      out[start_v] = 0.0

    # Rolling loop for the rest of the chunk
    for i in range(start_v + 1, end_v):
      tp_prev = (high[i - period] + low[i - period] + close[i - period]) / 3.0
      tp_i = (high[i] + low[i] + close[i]) / 3.0
      curr_sum = curr_sum + tp_i - tp_prev
      curr_sma = curr_sum * inv_period

      sum_abs_diff = 0.0
      for k in range(i - period + 1, i + 1):
        tp_k = (high[k] + low[k] + close[k]) / 3.0
        sum_abs_diff += abs(tp_k - curr_sma)

      mean_dev = sum_abs_diff * inv_period
      if mean_dev > EPSILON:
        out[i] = (tp_i - curr_sma) / (constant * mean_dev)
      else:
        out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_cmo_numba(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
  """Compute Chande Momentum Oscillator (CMO) using Wilder's Smoothing."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  out[:period] = np.nan

  sum_gains = 0.0
  sum_losses = 0.0

  for i in range(1, period + 1):
    diff = data[i] - data[i - 1]
    if diff > 0:
      sum_gains += diff
    elif diff < 0:
      sum_losses += -diff

  avg_gain = sum_gains / period
  avg_loss = sum_losses / period

  total = avg_gain + avg_loss
  out[period] = 100.0 * (avg_gain - avg_loss) / total if total != 0 else 0.0

  for i in range(period + 1, n):
    diff = data[i] - data[i - 1]
    curr_gain = diff if diff > 0 else 0.0
    curr_loss = -diff if diff < 0 else 0.0

    avg_gain = (avg_gain * (period - 1) + curr_gain) / period
    avg_loss = (avg_loss * (period - 1) + curr_loss) / period

    total = avg_gain + avg_loss
    out[i] = 100.0 * (avg_gain - avg_loss) / total if total != 0 else 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_mfi_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  volume: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Compute Money Flow Index (MFI) - Parallel Chunked with Fusion."""
  n = len(close)
  out = np.empty(n, dtype=np.float64)

  if n <= period:
    out[:] = np.nan
    return out

  # Adaptive parallelism
  total_len = n - period
  num_chunks = 16
  if total_len < 2048:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  out[:period] = np.nan

  for c in prange(num_chunks + 1):
    start_v = period + c * chunk_size
    end_v = period + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initial MF sums for this chunk
    curr_pos = 0.0
    curr_neg = 0.0

    for k in range(start_v - period + 1, start_v + 1):
      tp_k = (high[k] + low[k] + close[k]) / 3.0
      tp_prev = (high[k - 1] + low[k - 1] + close[k - 1]) / 3.0
      mf = tp_k * volume[k]
      if tp_k > tp_prev:
        curr_pos += mf
      elif tp_k < tp_prev:
        curr_neg += mf

    total = curr_pos + curr_neg
    out[start_v] = 100.0 * curr_pos / total if total > EPSILON else 0.0

    # Rolling loop for the rest of the chunk
    for i in range(start_v + 1, end_v):
      tp_i = (high[i] + low[i] + close[i]) / 3.0
      tp_prev_i = (high[i - 1] + low[i - 1] + close[i - 1]) / 3.0
      mf_i = tp_i * volume[i]
      if tp_i > tp_prev_i:
        curr_pos += mf_i
      elif tp_i < tp_prev_i:
        curr_neg += mf_i

      idx_trail = i - period
      tp_trail = (high[idx_trail] + low[idx_trail] + close[idx_trail]) / 3.0
      tp_prev_trail = (
        high[idx_trail - 1] + low[idx_trail - 1] + close[idx_trail - 1]
      ) / 3.0
      mf_trail = tp_trail * volume[idx_trail]
      if tp_trail > tp_prev_trail:
        curr_pos -= mf_trail
      elif tp_trail < tp_prev_trail:
        curr_neg -= mf_trail

      total = curr_pos + curr_neg
      out[i] = 100.0 * curr_pos / total if total > EPSILON else 0.0

  return out
