"""Numba-optimized Advanced Momentum Indicators.

This module consolidates kernels for:
- WILLR: Williams %R
- CCI: Commodity Channel Index
- CMO: Chande Momentum Oscillator
- MFI: Money Flow Index

All implementations use O(1) or optimized O(n) rolling window algorithms where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

# Constants
MIN_WINDOW_SIZE = 2
EPSILON = 1e-10  # Minimum denominator value


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_willr_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled Williams %R using parallel chunked lazy rescan.

  This algorithm splits the data into chunks to utilize multi-core processing
  while maintaining the O(N) efficiency of the lazy rescan algorithm.
  """
  n = len(close)
  willr = np.full(n, np.nan, dtype=np.float64)

  if n < period:
    return willr

  # Use parallel chunks logic
  # Start valid index: period - 1
  # Items before period-1 are NaN (already initialized)
  start_v = period - 1
  total_len = n - start_v

  num_chunks = 16

  # Adaptive parallelism: avoid overhead for small arrays
  if total_len < 1024:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = start_v + (c + 1) * chunk_size

    if c == num_chunks:
      idx_end = n

    if idx_start >= n:
      continue

    # Init trackers for this chunk
    h_idx = -1
    l_idx = -1
    h_val = -np.inf
    l_val = np.inf

    # Initialize window min/max at start of chunk
    # Pre-scan the full window ending at idx_start to establish state
    start_lookback = idx_start - period + 1
    for k in range(start_lookback, idx_start + 1):
      if high[k] >= h_val:
        h_val = high[k]
        h_idx = k
      if low[k] <= l_val:
        l_val = low[k]
        l_idx = k

    # Process chunk
    for i in range(idx_start, idx_end):
      trailing = i - period + 1

      # High update
      if h_idx < trailing:
        # Max is out of window, must rescan
        h_idx = trailing
        h_val = high[trailing]
        for k in range(trailing + 1, i + 1):
          if high[k] >= h_val:
            h_val = high[k]
            h_idx = k
      elif high[i] >= h_val:
        # New high is higher than current max
        h_val = high[i]
        h_idx = i

      # Low update
      if l_idx < trailing:
        # Min is out of window, must rescan
        l_idx = trailing
        l_val = low[trailing]
        for k in range(trailing + 1, i + 1):
          if low[k] <= l_val:
            l_val = low[k]
            l_idx = k
      elif low[i] <= l_val:
        # New low is lower than current min
        l_val = low[i]
        l_idx = i

      div = h_val - l_val
      if div > EPSILON:
        willr[i] = -100.0 * (h_val - close[i]) / div
      else:
        willr[i] = 0.0

  return willr


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_cci_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  period: int,
  constant: float = 0.015,
) -> NDArray[np.float64]:
  """Compute Commodity Channel Index (CCI) - Parallel implementation.

  CCI = (TP - SMA_TP) / (0.015 * MeanDeviation)
  """
  n = len(close)
  out = np.full(n, np.nan, dtype=np.float64)

  if n < period:
    return out

  # Precompute TP
  tp = np.empty(n, dtype=np.float64)
  # Standard loop usually vectorized well, explicit parallel loop is safer
  # prange for simple TP calculation
  for i in prange(n):
    tp[i] = (high[i] + low[i] + close[i]) * 0.3333333333333333

  start_v = period - 1
  total_len = n - start_v

  num_chunks = 16
  if total_len < 1024:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  inv_period = 1.0 / period

  # Parallel Loop
  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = start_v + (c + 1) * chunk_size
    if c == num_chunks:
      idx_end = n
    if idx_start >= n:
      continue

    # Initialize rolling state for this chunk
    # 1. Sum of TP for SMA
    # 2. MeanDev calculation requires iteration anyway

    # Calculate initial sum for the window ending at idx_start (inclusive)
    current_sum = 0.0
    for k in range(idx_start - period + 1, idx_start + 1):
      current_sum += tp[k]

    # Pre-calculate first value at idx_start?
    # Or loop from idx_start?
    # Standard: calculate for idx_start inside loop, but means we need sum[idx_start-1]...
    # Better: Calculate complete initial state for window ending at idx_start *before* loop starts?
    # Actually, idx_start is the *first index we compute*.
    # So we need Sum of [idx_start-period+1 : idx_start] to compute output[idx_start].
    # But wait, rolling sum usually adds `entering` and subtracts `leaving`.
    # At `idx_start`, entering is tp[idx_start].

    # Let's adjust:
    # We initialize `current_sum` to sum(tp[idx_start-period : idx_start]) -> window ending at idx_start-1?
    # No, let's just initialize for the *start state* of the loop iteration.

    # Initialize sum up to idx_start-1
    # i ranges from idx_start to idx_end

    # Sum for window ending at idx_start-1:
    # range: [idx_start - period, idx_start - 1]

    running_sum = 0.0
    for k in range(idx_start - period, idx_start):  # Exclusive of idx_start
      running_sum += tp[k]

    for i in range(idx_start, idx_end):
      entering = tp[i]
      leaving = tp[i - period]

      running_sum = running_sum + entering - leaving

      sma = running_sum * inv_period

      # Mean Deviation Loop
      # Sum of abs(tp[j] - sma)
      # This inner loop is O(Period).
      # Total efficiency O(N * Period).
      # Vectorization helps here.

      sum_abs_diff = 0.0
      window_start = i - period + 1
      for k in range(window_start, i + 1):
        diff = tp[k] - sma
        # abs() is intrinsic
        sum_abs_diff += abs(diff)

      mean_dev = sum_abs_diff * inv_period

      if mean_dev > EPSILON:
        out[i] = (entering - sma) / (constant * mean_dev)
      else:
        out[i] = 0.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_cmo_numba(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
  """Compute Chande Momentum Oscillator (CMO).

  Uses Wilder's Smoothing, which is recursive and hard to parallelize efficiently.
  Keeping standard optimized sequential implementation.
  """
  n = len(data)
  out = np.full(n, np.nan, dtype=np.float64)

  if n <= period:
    return out

  sum_gains = 0.0
  sum_losses = 0.0

  # 1. Warmup sum (first period)
  for i in range(1, period + 1):
    diff = data[i] - data[i - 1]
    if diff > 0:
      sum_gains += diff
    elif diff < 0:
      sum_losses += -diff

  # First Avg values (Simple Average for startup)
  avg_gain = sum_gains / period
  avg_loss = sum_losses / period

  total = avg_gain + avg_loss
  if total != 0.0:
    out[period] = 100.0 * (avg_gain - avg_loss) / total
  else:
    out[period] = 0.0

  # 2. Rolling update (Wilder's Smoothing)
  # Avg_t = (Avg_{t-1} * (period-1) + Val_t) / period

  prev_avg_gain = avg_gain
  prev_avg_loss = avg_loss

  for i in range(period + 1, n):
    diff = data[i] - data[i - 1]
    curr_gain = diff if diff > 0 else 0.0
    curr_loss = -diff if diff < 0 else 0.0

    avg_gain = (prev_avg_gain * (period - 1) + curr_gain) / period
    avg_loss = (prev_avg_loss * (period - 1) + curr_loss) / period

    total = avg_gain + avg_loss
    if total != 0.0:
      out[i] = 100.0 * (avg_gain - avg_loss) / total
    else:
      out[i] = 0.0

    prev_avg_gain = avg_gain
    prev_avg_loss = avg_loss

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_mfi_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  volume: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Compute Money Flow Index (MFI) - Parallel implementation.

  MFI requires rolling sum of Positive Flow and Negative Flow.
  """
  n = len(close)
  out = np.full(n, np.nan, dtype=np.float64)

  if n <= period:
    return out

  # Precompute TP
  tp = np.empty(n, dtype=np.float64)
  for i in prange(n):
    tp[i] = (high[i] + low[i] + close[i]) * 0.3333333333333333

  # Precompute Flows
  # Positive Flow at i: TP[i] * Vol[i] if TP[i] > TP[i-1] else 0
  # Negative Flow at i: TP[i] * Vol[i] if TP[i] < TP[i-1] else 0
  pos_flow = np.zeros(n, dtype=np.float64)
  neg_flow = np.zeros(n, dtype=np.float64)

  # Parallel precompute of flows
  for i in prange(1, n):
    curr_tp = tp[i]
    prev_tp = tp[i - 1]
    mf = curr_tp * volume[i]

    if curr_tp > prev_tp:
      pos_flow[i] = mf
    elif curr_tp < prev_tp:
      neg_flow[i] = mf

  # Parallel Rolling Sums
  start_v = period
  total_len = n - start_v

  num_chunks = 16
  if total_len < 2048:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  for c in prange(num_chunks + 1):
    idx_start = start_v + c * chunk_size
    idx_end = start_v + (c + 1) * chunk_size
    if c == num_chunks:
      idx_end = n
    if idx_start >= n:
      continue

    # Initialize sums for chunk start
    # Sum range [idx_start - period + 1 : idx_start + 1] (inclusive of end?)
    # Sum window of length period ending at idx_start.
    # Indices: idx_start-period+1 to idx_start

    # But wait, MFI logic:
    # First output at index 'period'. Sums from i=1 to period.
    # Window length 'period'.

    curr_pos = 0.0
    curr_neg = 0.0

    # Initialize state just before idx_start loop
    # We need sums for window ending at idx_start-1
    for k in range(idx_start - period + 1, idx_start):  # Up to idx_start-1
      curr_pos += pos_flow[k]
      curr_neg += neg_flow[k]

    # Loop
    for i in range(idx_start, idx_end):
      # Add new
      curr_pos += pos_flow[i]
      curr_neg += neg_flow[i]

      # Remove old (index i - period)
      # Flow array at [i-period]
      # Because flow[0] is 0 anyway (no prev), 0 indexing is safe if period >= 1.
      old_idx = i - period
      curr_pos -= pos_flow[old_idx]
      curr_neg -= neg_flow[old_idx]

      # FP correction
      if curr_pos < 0:
        curr_pos = 0.0
      if curr_neg < 0:
        curr_neg = 0.0

      total = curr_pos + curr_neg
      if total > EPSILON:
        out[i] = 100.0 * curr_pos / total
      else:
        out[i] = 0.0  # Or 50?

  return out
