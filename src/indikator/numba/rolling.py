"""Numba-optimized rolling min/max calculations using Parallel Gil-Werman."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_midprice_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPRICE using strict O(N) Parallel Gil-Werman."""
  # We invoke min/max separately effectively doubling work, but it's simple.
  # For max perf, we should fuse them, but let's rely on their individual optimization first.
  return (compute_max_numba(high, period) + compute_min_numba(low, period)) / 2.0


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_midpoint_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPOINT using strict O(N) Parallel Gil-Werman."""
  return (compute_max_numba(data, period) + compute_min_numba(data, period)) / 2.0


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_min_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MIN using Parallel Gil-Werman algorithm."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  num_blocks = (n + period - 1) // period

  # Precompute Prefix and Suffix Min using parallel blocks
  prefix_min = np.empty(n, dtype=np.float64)
  suffix_min = np.empty(n, dtype=np.float64)

  # Parallel scan over blocks
  for b in prange(num_blocks):
    start = b * period
    end = min(start + period, n)

    # Prefix Min (forward scan in block)
    curr = data[start]
    prefix_min[start] = curr
    for i in range(start + 1, end):
      v = data[i]
      if v < curr:
        curr = v
      prefix_min[i] = curr

    # Suffix Min (backward scan in block)
    curr = data[end - 1]
    suffix_min[end - 1] = curr
    for i in range(end - 2, start - 1, -1):
      v = data[i]
      if v < curr:
        curr = v
      suffix_min[i] = curr

  # Parallel Merge
  # Window [i-period+1, i] aligns: L=i-period+1, R=i
  for i in prange(period - 1, n):
    L = i - period + 1
    # suffix_min[L] covers L to block_end
    # prefix_min[i] covers block_start to i
    v1 = suffix_min[L]
    v2 = prefix_min[i]
    if v1 < v2:
      out[i] = v1
    else:
      out[i] = v2

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_max_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MAX using Parallel Gil-Werman algorithm."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  num_blocks = (n + period - 1) // period
  prefix_max = np.empty(n, dtype=np.float64)
  suffix_max = np.empty(n, dtype=np.float64)

  for b in prange(num_blocks):
    start = b * period
    end = min(start + period, n)

    # Prefix
    curr = data[start]
    prefix_max[start] = curr
    for i in range(start + 1, end):
      v = data[i]
      if v > curr:
        curr = v
      prefix_max[i] = curr

    # Suffix
    curr = data[end - 1]
    suffix_max[end - 1] = curr
    for i in range(end - 2, start - 1, -1):
      v = data[i]
      if v > curr:
        curr = v
      suffix_max[i] = curr

  for i in prange(period - 1, n):
    L = i - period + 1
    v1 = suffix_max[L]
    v2 = prefix_max[i]
    if v1 > v2:
      out[i] = v1
    else:
      out[i] = v2

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_minindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MININDEX using Parallel Gil-Werman."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  num_blocks = (n + period - 1) // period
  prefix_min_idx = np.empty(n, dtype=np.int64)
  suffix_min_idx = np.empty(n, dtype=np.int64)

  # Parallel Scan
  for b in prange(num_blocks):
    start = b * period
    end = min(start + period, n)

    # Prefix
    c_idx = start
    c_val = data[start]
    prefix_min_idx[start] = c_idx
    for i in range(start + 1, end):
      v = data[i]
      if v < c_val:
        c_val = v
        c_idx = i
      prefix_min_idx[i] = c_idx

    # Suffix
    c_idx = end - 1
    c_val = data[end - 1]
    suffix_min_idx[end - 1] = c_idx
    for i in range(end - 2, start - 1, -1):
      v = data[i]
      if v < c_val:
        c_val = v
        c_idx = i
      suffix_min_idx[i] = c_idx

  # Parallel Merge
  for i in prange(period - 1, n):
    L = i - period + 1
    # Compare values at the best indices from suffix/prefix
    idx_s = suffix_min_idx[L]
    idx_p = prefix_min_idx[i]
    if data[idx_s] < data[idx_p]:
      out[i] = float(idx_s)
    elif data[idx_s] > data[idx_p]:
      out[i] = float(idx_p)
    else:
      # Equal values: prefer lower index (standard behavior)
      if idx_s < idx_p:
        out[i] = float(idx_s)
      else:
        out[i] = float(idx_p)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_maxindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MAXINDEX using Parallel Gil-Werman."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)
  out[:] = np.nan

  if n < period:
    return out

  num_blocks = (n + period - 1) // period
  prefix_max_idx = np.empty(n, dtype=np.int64)
  suffix_max_idx = np.empty(n, dtype=np.int64)

  for b in prange(num_blocks):
    start = b * period
    end = min(start + period, n)

    # Prefix
    c_idx = start
    c_val = data[start]
    prefix_max_idx[start] = c_idx
    for i in range(start + 1, end):
      v = data[i]
      if v > c_val:
        c_val = v
        c_idx = i
      prefix_max_idx[i] = c_idx

    # Suffix
    c_idx = end - 1
    c_val = data[end - 1]
    suffix_max_idx[end - 1] = c_idx
    for i in range(end - 2, start - 1, -1):
      v = data[i]
      if v > c_val:
        c_val = v
        c_idx = i
      suffix_max_idx[i] = c_idx

  for i in prange(period - 1, n):
    L = i - period + 1
    idx_s = suffix_max_idx[L]
    idx_p = prefix_max_idx[i]

    if data[idx_s] > data[idx_p]:
      out[i] = float(idx_s)
    elif data[idx_s] < data[idx_p]:
      out[i] = float(idx_p)
    else:
      # Equal values: prefer lower index
      if idx_s < idx_p:
        out[i] = float(idx_s)
      else:
        out[i] = float(idx_p)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_sum_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling SUM using O(1) rolling update."""
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
  """Calculate rolling MIN and MAX simultaneously using Parallel Gil-Werman."""
  n = len(data)
  out_min = np.empty(n, dtype=np.float64)
  out_max = np.empty(n, dtype=np.float64)
  out_min[:] = np.nan
  out_max[:] = np.nan

  if n < period:
    return out_min, out_max

  num_blocks = (n + period - 1) // period
  px_min = np.empty(n, dtype=np.float64)
  sx_min = np.empty(n, dtype=np.float64)
  px_max = np.empty(n, dtype=np.float64)
  sx_max = np.empty(n, dtype=np.float64)

  # Parallel Block Scan
  for b in prange(num_blocks):
    start = b * period
    end = min(start + period, n)

    # Prefix
    c_min = data[start]
    c_max = data[start]
    px_min[start] = c_min
    px_max[start] = c_max
    for i in range(start + 1, end):
      v = data[i]
      if v < c_min:
        c_min = v
      if v > c_max:
        c_max = v
      px_min[i] = c_min
      px_max[i] = c_max

    # Suffix
    c_min = data[end - 1]
    c_max = data[end - 1]
    sx_min[end - 1] = c_min
    sx_max[end - 1] = c_max
    for i in range(end - 2, start - 1, -1):
      v = data[i]
      if v < c_min:
        c_min = v
      if v > c_max:
        c_max = v
      sx_min[i] = c_min
      sx_max[i] = c_max

  # Parallel Merge
  for i in prange(period - 1, n):
    L = i - period + 1

    # Min Merge
    v1 = sx_min[L]
    v2 = px_min[i]
    if v1 < v2:
      out_min[i] = v1
    else:
      out_min[i] = v2

    # Max Merge
    v1 = sx_max[L]
    v2 = px_max[i]
    if v1 > v2:
      out_max[i] = v1
    else:
      out_max[i] = v2

  return out_min, out_max


@jit(nopython=True, cache=True, nogil=True, fastmath=True, parallel=True)
def compute_minmaxindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Calculate rolling MININDEX and MAXINDEX (Using Parallel Gil-Werman)."""
  n = len(data)
  out_min = np.empty(n, dtype=np.float64)
  out_max = np.empty(n, dtype=np.float64)
  out_min[:] = np.nan
  out_max[:] = np.nan

  if n < period:
    return out_min, out_max

  num_blocks = (n + period - 1) // period

  # Min Allocations
  prefix_min = np.empty(n, dtype=np.int64)
  suffix_min = np.empty(n, dtype=np.int64)

  # Max Allocations
  prefix_max = np.empty(n, dtype=np.int64)
  suffix_max = np.empty(n, dtype=np.int64)

  # Parallel Block Scan
  for b in prange(num_blocks):
    start = b * period
    end = min(start + period, n)

    # --- Min Prefix/Suffix ---
    # Prefix
    c_idx = start
    c_val = data[start]
    prefix_min[start] = c_idx
    for i in range(start + 1, end):
      v = data[i]
      if v < c_val:
        c_val = v
        c_idx = i
      prefix_min[i] = c_idx

    # Suffix
    c_idx = end - 1
    c_val = data[end - 1]
    suffix_min[end - 1] = c_idx
    for i in range(end - 2, start - 1, -1):
      v = data[i]
      if v < c_val:
        c_val = v
        c_idx = i
      suffix_min[i] = c_idx

    # --- Max Prefix/Suffix ---
    # Prefix
    c_idx = start
    c_val = data[start]
    prefix_max[start] = c_idx
    for i in range(start + 1, end):
      v = data[i]
      if v > c_val:
        c_val = v
        c_idx = i
      prefix_max[i] = c_idx

    # Suffix
    c_idx = end - 1
    c_val = data[end - 1]
    suffix_max[end - 1] = c_idx
    for i in range(end - 2, start - 1, -1):
      v = data[i]
      if v > c_val:
        c_val = v
        c_idx = i
      suffix_max[i] = c_idx

  # Parallel Merge
  for i in prange(period - 1, n):
    L = i - period + 1

    # Min Merge
    idx_s = suffix_min[L]
    idx_p = prefix_min[i]
    if data[idx_s] < data[idx_p]:
      out_min[i] = float(idx_s)
    elif data[idx_s] > data[idx_p]:
      out_min[i] = float(idx_p)
    else:
      if idx_s < idx_p:
        out_min[i] = float(idx_s)
      else:
        out_min[i] = float(idx_p)

    # Max Merge
    idx_s = suffix_max[L]
    idx_p = prefix_max[i]
    if data[idx_s] > data[idx_p]:
      out_max[i] = float(idx_s)
    elif data[idx_s] < data[idx_p]:
      out_max[i] = float(idx_p)
    else:
      if idx_s < idx_p:
        out_max[i] = float(idx_s)
      else:
        out_max[i] = float(idx_p)

  return out_min, out_max
