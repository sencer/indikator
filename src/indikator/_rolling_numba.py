"""Numba-optimized rolling min/max calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_midprice_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPRICE using amortized O(n) lazy rescan.

  MIDPRICE = (highest high + lowest low) / 2 over period.

  Uses lazy rescan: tracks index of current max/min and only rescans
  when that index falls outside the sliding window.
  """
  n = len(high)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    for i in range(n):
      out[i] = np.nan
    return out

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # Initialize trackers with first window
  h_idx = 0
  h_val = high[0]
  l_idx = 0
  l_val = low[0]

  for k in range(1, period):
    if high[k] >= h_val:
      h_val = high[k]
      h_idx = k
    if low[k] <= l_val:
      l_val = low[k]
      l_idx = k

  out[period - 1] = (h_val + l_val) / 2.0

  # Main loop with lazy rescan
  for i in range(period, n):
    trailing = i - period + 1

    # Update high - lazy rescan
    if h_idx < trailing:
      # Max fell out of window, rescan
      h_idx = trailing
      h_val = high[trailing]
      for k in range(trailing + 1, i + 1):
        if high[k] >= h_val:
          h_val = high[k]
          h_idx = k
    elif high[i] >= h_val:
      h_val = high[i]
      h_idx = i

    # Update low - lazy rescan
    if l_idx < trailing:
      # Min fell out of window, rescan
      l_idx = trailing
      l_val = low[trailing]
      for k in range(trailing + 1, i + 1):
        if low[k] <= l_val:
          l_val = low[k]
          l_idx = k
    elif low[i] <= l_val:
      l_val = low[i]
      l_idx = i

    out[i] = (h_val + l_val) / 2.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_midpoint_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate MIDPOINT using amortized O(n) lazy rescan.

  MIDPOINT = (highest + lowest) / 2 over period.

  Uses lazy rescan: tracks index of current max/min and only rescans
  when that index falls outside the sliding window.
  """
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    for i in range(n):
      out[i] = np.nan
    return out

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # Initialize trackers with first window
  max_idx = 0
  max_val = data[0]
  min_idx = 0
  min_val = data[0]

  for k in range(1, period):
    if data[k] >= max_val:
      max_val = data[k]
      max_idx = k
    if data[k] <= min_val:
      min_val = data[k]
      min_idx = k

  out[period - 1] = (max_val + min_val) / 2.0

  # Main loop with lazy rescan
  for i in range(period, n):
    trailing = i - period + 1

    # Update max - lazy rescan
    if max_idx < trailing:
      max_idx = trailing
      max_val = data[trailing]
      for k in range(trailing + 1, i + 1):
        if data[k] >= max_val:
          max_val = data[k]
          max_idx = k
    elif data[i] >= max_val:
      max_val = data[i]
      max_idx = i

    # Update min - lazy rescan
    if min_idx < trailing:
      min_idx = trailing
      min_val = data[trailing]
      for k in range(trailing + 1, i + 1):
        if data[k] <= min_val:
          min_val = data[k]
          min_idx = k
    elif data[i] <= min_val:
      min_val = data[i]
      min_idx = i

    out[i] = (max_val + min_val) / 2.0

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_min_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MIN using amortized O(n) lazy rescan."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    for i in range(n):
      out[i] = np.nan
    return out

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # First window
  l_idx = 0
  l_val = data[0]
  for k in range(1, period):
    if data[k] <= l_val:
      l_val = data[k]
      l_idx = k
  out[period - 1] = l_val

  # Lazy rescan
  for i in range(period, n):
    trailing = i - period + 1

    # Update low
    if l_idx < trailing:
      # Rescan
      l_idx = trailing
      l_val = data[trailing]
      for k in range(trailing + 1, i + 1):
        if data[k] <= l_val:
          l_val = data[k]
          l_idx = k
    elif data[i] <= l_val:
      l_val = data[i]
      l_idx = i

    out[i] = l_val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_max_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MAX using amortized O(n) lazy rescan."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    for i in range(n):
      out[i] = np.nan
    return out

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # First window
  h_idx = 0
  h_val = data[0]
  for k in range(1, period):
    if data[k] >= h_val:
      h_val = data[k]
      h_idx = k
  out[period - 1] = h_val

  # Lazy rescan
  for i in range(period, n):
    trailing = i - period + 1

    # Update high
    if h_idx < trailing:
      # Rescan
      h_idx = trailing
      h_val = data[trailing]
      for k in range(trailing + 1, i + 1):
        if data[k] >= h_val:
          h_val = data[k]
          h_idx = k
    elif data[i] >= h_val:
      h_val = data[i]
      h_idx = i

    out[i] = h_val

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_minindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MININDEX (index relative to start of array)."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    for i in range(n):
      out[i] = np.nan
    return out

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # First window
  l_idx = 0
  l_val = data[0]
  for k in range(1, period):
    if data[k] <= l_val:
      l_val = data[k]
      l_idx = k
  out[period - 1] = float(l_idx)

  # Lazy rescan
  for i in range(period, n):
    trailing = i - period + 1

    if l_idx < trailing:
      # Rescan
      l_idx = trailing
      l_val = data[trailing]
      for k in range(trailing + 1, i + 1):
        if data[k] <= l_val:
          l_val = data[k]
          l_idx = k
    elif data[i] <= l_val:
      l_val = data[i]
      l_idx = i

    out[i] = float(l_idx)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_maxindex_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling MAXINDEX (index relative to start of array)."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    for i in range(n):
      out[i] = np.nan
    return out

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # First window
  h_idx = 0
  h_val = data[0]
  for k in range(1, period):
    if data[k] >= h_val:
      h_val = data[k]
      h_idx = k
  out[period - 1] = float(h_idx)

  # Lazy rescan
  for i in range(period, n):
    trailing = i - period + 1

    if h_idx < trailing:
      # Rescan
      h_idx = trailing
      h_val = data[trailing]
      for k in range(trailing + 1, i + 1):
        if data[k] >= h_val:
          h_val = data[k]
          h_idx = k
    elif data[i] >= h_val:
      h_val = data[i]
      h_idx = i

    out[i] = float(h_idx)

  return out


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_sum_numba(
  data: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Calculate rolling SUM using O(1) rolling update."""
  n = len(data)
  out = np.empty(n, dtype=np.float64)

  if n < period:
    for i in range(n):
      out[i] = np.nan
    return out

  # NaN for warmup
  for i in range(period - 1):
    out[i] = np.nan

  # First window sum
  current_sum = 0.0
  for i in range(period):
    current_sum += data[i]

  out[period - 1] = current_sum

  # Rolling update
  for i in range(period, n):
    old_val = data[i - period]
    new_val = data[i]
    current_sum = current_sum - old_val + new_val
    out[i] = current_sum

  return out
