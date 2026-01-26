"""Numba-optimized rolling min/max calculations using Parallel Chunked Lazy Rescan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray

EPSILON = 1e-14


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_midprice_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPRICE using Parallel Chunked strategy."""
  # Fuse operation: (Max(High) + Min(Low)) / 2
  n = len(high)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  # Adaptive parallelism parameters
  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # LOCAL STATE RE-COMPUTE (Warmup for this chunk)
    # We need to find the Min/Max of the window ENDING at start_v
    # Window is [start_v - period + 1, start_v]
    h_val, l_val = -np.inf, np.inf
    h_idx, l_idx = -1, -1

    scan_start = start_v - period + 1
    for k in range(scan_start, start_v + 1):
      if high[k] >= h_val:
        h_val, h_idx = high[k], k
      if low[k] <= l_val:
        l_val, l_idx = low[k], k

    # Store first value of chunk
    out[start_v] = (h_val + l_val) * 0.5

    # PROCESS CHUNK
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1

      # Update High (Max)
      tmp_h = high[i]
      if h_idx < trailing:
        # Rescan
        h_val = high[trailing]
        h_idx = trailing
        for k in range(trailing + 1, i + 1):
          if high[k] >= h_val:
            h_val = high[k]
            h_idx = k
      elif tmp_h >= h_val:
        h_val = tmp_h
        h_idx = i

      # Update Low (Min)
      tmp_l = low[i]
      if l_idx < trailing:
        # Rescan
        l_val = low[trailing]
        l_idx = trailing
        for k in range(trailing + 1, i + 1):
          if low[k] <= l_val:
            l_val = low[k]
            l_idx = k
      elif tmp_l <= l_val:
        l_val = tmp_l
        l_idx = i

      out[i] = (h_val + l_val) * 0.5

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_midpoint_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPOINT using Parallel Chunked strategy."""
  min_vals, max_vals = compute_minmax_numba(data, period)
  return (min_vals + max_vals) * 0.5


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_min_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MIN using Parallel Chunked Lazy Rescan."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initialize Chunk State
    l_val = np.inf
    l_idx = -1
    for k in range(start_v - period + 1, start_v + 1):
      if data[k] <= l_val:
        l_val, l_idx = data[k], k

    out[start_v] = l_val

    # Process Chunk
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1
      tmp = data[i]

      if l_idx < trailing:
        l_val = data[trailing]
        l_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] <= l_val:
            l_val = data[k]
            l_idx = k
      elif tmp <= l_val:
        l_val = tmp
        l_idx = i

      out[i] = l_val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_max_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MAX using Parallel Chunked Lazy Rescan."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initialize Chunk State
    h_val = -np.inf
    h_idx = -1
    for k in range(start_v - period + 1, start_v + 1):
      if data[k] >= h_val:
        h_val, h_idx = data[k], k

    out[start_v] = h_val

    # Process Chunk
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1
      tmp = data[i]

      if h_idx < trailing:
        h_val = data[trailing]
        h_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] >= h_val:
            h_val = data[k]
            h_idx = k
      elif tmp >= h_val:
        h_val = tmp
        h_idx = i

      out[i] = h_val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_minindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MININDEX using Parallel Chunked Lazy Rescan."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initialize Chunk State
    l_val = np.inf
    l_idx = -1
    for k in range(start_v - period + 1, start_v + 1):
      if data[k] <= l_val:
        l_val, l_idx = data[k], k

    out[start_v] = float(l_idx)

    # Process Chunk
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1
      tmp = data[i]

      if l_idx < trailing:
        l_val = data[trailing]
        l_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] <= l_val:
            l_val = data[k]
            l_idx = k
      elif tmp <= l_val:
        l_val = tmp
        l_idx = i

      out[i] = float(l_idx)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_maxindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MAXINDEX using Parallel Chunked Lazy Rescan."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initialize Chunk State
    h_val = -np.inf
    h_idx = -1
    for k in range(start_v - period + 1, start_v + 1):
      if data[k] >= h_val:
        h_val, h_idx = data[k], k

    out[start_v] = float(h_idx)

    # Process Chunk
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1
      tmp = data[i]

      if h_idx < trailing:
        h_val = data[trailing]
        h_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] >= h_val:
            h_val = data[k]
            h_idx = k
      elif tmp >= h_val:
        h_val = tmp
        h_idx = i

      out[i] = float(h_idx)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_sum_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling SUM using O(1) rolling update."""
  # SUM is already efficient serial, but parallel doesn't help much on simple sum
  # due to memory bounds. We keep serial O(1) or could optimize.
  # Actually, Prefix Sum (scan) is parallelizable, but rolling sum is O(1) serially.
  # Let's keep it serial for now as it beats 1.0x usually.
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  current_sum = 0.0
  # Warmup
  for i in range(period):
    current_sum += data[i]

  out[period - 1] = current_sum

  # Rolling
  for i in range(period, n):
    current_sum += data[i] - data[i - period]
    out[i] = current_sum

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_minmax_numba(
  data: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Calculate rolling MIN and MAX simultaneously using Parallel Chunked Lazy Rescan."""
  n = len(data)
  out_min = np.empty(n, dtype=np.float64)
  out_max = np.empty(n, dtype=np.float64)
  out_min[:] = np.nan
  out_max[:] = np.nan

  if n < period:
    return out_min, out_max

  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  # Parallel Chunk Scan
  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initialize Chunk State
    l_val, h_val = np.inf, -np.inf
    l_idx, h_idx = -1, -1

    scan_start = start_v - period + 1
    for k in range(scan_start, start_v + 1):
      val = data[k]
      if val <= l_val:
        l_val, l_idx = val, k
      if val >= h_val:
        h_val, h_idx = val, k

    out_min[start_v] = l_val
    out_max[start_v] = h_val

    # Process Chunk
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1
      tmp = data[i]

      # Min Update
      if l_idx < trailing:
        l_val = data[trailing]
        l_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] <= l_val:
            l_val = data[k]
            l_idx = k
      elif tmp <= l_val:
        l_val = tmp
        l_idx = i

      # Max Update
      if h_idx < trailing:
        h_val = data[trailing]
        h_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] >= h_val:
            h_val = data[k]
            h_idx = k
      elif tmp >= h_val:
        h_val = tmp
        h_idx = i

      out_min[i] = l_val
      out_max[i] = h_val

  return out_min, out_max


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_minmaxindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Calculate rolling MININDEX and MAXINDEX (Using Parallel Chunked Lazy Rescan)."""
  n = len(data)
  out_min = np.empty(n, dtype=np.float64)
  out_max = np.empty(n, dtype=np.float64)
  out_min[:] = np.nan
  out_max[:] = np.nan

  if n < period:
    return out_min, out_max

  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  # Parallel Chunk Scan
  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initialize Chunk State
    l_val, h_val = np.inf, -np.inf
    l_idx, h_idx = -1, -1

    scan_start = start_v - period + 1
    for k in range(scan_start, start_v + 1):
      val = data[k]
      if val <= l_val:
        l_val, l_idx = val, k
      if val >= h_val:
        h_val, h_idx = val, k

    out_min[start_v] = float(l_idx)
    out_max[start_v] = float(h_idx)

    # Process Chunk
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1
      tmp = data[i]

      # Min Update
      if l_idx < trailing:
        l_val = data[trailing]
        l_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] <= l_val:
            l_val = data[k]
            l_idx = k
      elif tmp <= l_val:
        l_val = tmp
        l_idx = i

      # Max Update
      if h_idx < trailing:
        h_val = data[trailing]
        h_idx = trailing
        for k in range(trailing + 1, i + 1):
          if data[k] >= h_val:
            h_val = data[k]
            h_idx = k
      elif tmp >= h_val:
        h_val = tmp
        h_idx = i

      out_min[i] = float(l_idx)
      out_max[i] = float(h_idx)

  return out_min, out_max


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_min_low_max_high_numba(
  low: NDArray[np.float64],
  high: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Calculate rolling MIN(low) and MAX(high) simultaneously using fused loop."""
  n = len(low)
  out_min = np.empty(n, dtype=np.float64)
  out_max = np.empty(n, dtype=np.float64)
  out_min[:] = np.nan
  out_max[:] = np.nan

  if n < period:
    return out_min, out_max

  total_len = n - period + 1
  num_chunks = 16
  if total_len < 4096:
    num_chunks = 1

  chunk_size = total_len // num_chunks
  if chunk_size < 1:
    chunk_size = total_len
    num_chunks = 1

  warmup_idx = period - 1

  # Parallel Chunk Scan
  for c in prange(num_chunks + 1):
    start_v = warmup_idx + c * chunk_size
    end_v = warmup_idx + (c + 1) * chunk_size
    if c == num_chunks:
      end_v = n
    if start_v >= n:
      continue

    # Initialize Chunk State
    l_val, h_val = np.inf, -np.inf
    l_idx, h_idx = -1, -1

    scan_start = start_v - period + 1
    for k in range(scan_start, start_v + 1):
      v_l = low[k]
      v_h = high[k]
      if v_l <= l_val:
        l_val, l_idx = v_l, k
      if v_h >= h_val:
        h_val, h_idx = v_h, k

    out_min[start_v] = l_val
    out_max[start_v] = h_val

    # Process Chunk
    for i in range(start_v + 1, end_v):
      trailing = i - period + 1
      tmp_l = low[i]
      tmp_h = high[i]

      # Min Update (on LOW)
      if l_idx < trailing:
        l_val = low[trailing]
        l_idx = trailing
        for k in range(trailing + 1, i + 1):
          if low[k] <= l_val:
            l_val = low[k]
            l_idx = k
      elif tmp_l <= l_val:
        l_val = tmp_l
        l_idx = i

      # Max Update (on HIGH)
      if h_idx < trailing:
        h_val = high[trailing]
        h_idx = trailing
        for k in range(trailing + 1, i + 1):
          if high[k] >= h_val:
            h_val = high[k]
            h_idx = k
      elif tmp_h >= h_val:
        h_val = tmp_h
        h_idx = i

      out_min[i] = l_val
      out_max[i] = h_val

  return out_min, out_max
