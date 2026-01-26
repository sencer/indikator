"""Numba-optimized VWAP calculation.

This module contains JIT-compiled functions for VWAP calculation.
Separated for better code organization and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit, prange
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_vwap_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  volumes: NDArray[np.float64],
  reset_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
  """Numba JIT-compiled VWAP calculation with reset support.

  VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)

  The reset_mask indicates where to reset the cumulative sum (e.g., at
  session boundaries). When reset_mask[i] is True, the VWAP calculation
  starts fresh from that bar.

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of close prices
    volumes: Array of volumes
    reset_mask: Boolean array indicating where to reset VWAP calculation

  Returns:
    Array of VWAP values
  """
  n = len(high)
  vwap = np.zeros(n, dtype=np.float64)

  cum_pv = 0.0  # Cumulative price * volume
  cum_v = 0.0  # Cumulative volume

  for i in range(n):
    # Reset if needed
    if reset_mask[i]:
      cum_pv = 0.0
      cum_v = 0.0

    # Calculate typical price inline
    tp = (high[i] + low[i] + close[i]) / 3.0

    # Update cumulative sums
    cum_pv += tp * volumes[i]
    cum_v += volumes[i]

    # Calculate VWAP (avoid division by zero)
    if cum_v > 0:
      vwap[i] = cum_pv / cum_v
    else:
      vwap[i] = tp  # Fallback to typical price

  return vwap


@jit(nopython=True, cache=True, nogil=True, parallel=True, fastmath=True)
def compute_vwap_parallel_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  volumes: NDArray[np.float64],
  reset_indices: NDArray[np.int64],
) -> NDArray[np.float64]:
  """Parallel VWAP calculation by session blocks.

  Args:
      reset_indices: Array of indices where a new session starts.
  """
  n = len(high)
  vwap = np.zeros(n, dtype=np.float64)
  num_sessions = len(reset_indices)

  for s in prange(num_sessions):
    start_idx = reset_indices[s]
    end_idx = reset_indices[s + 1] if s < num_sessions - 1 else n

    cum_pv = 0.0
    cum_v = 0.0
    for i in range(start_idx, end_idx):
      tp = (high[i] + low[i] + close[i]) / 3.0
      cum_pv += tp * volumes[i]
      cum_v += volumes[i]
      if cum_v > 1e-12:
        vwap[i] = cum_pv / cum_v
      else:
        vwap[i] = tp

  return vwap


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_anchored_vwap_numba(
  typical_prices: NDArray[np.float64],
  volumes: NDArray[np.float64],
  anchor_index: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled anchored VWAP calculation from a specific bar.

  Calculates VWAP starting from anchor_index forward. All bars before
  anchor_index will have NaN values.

  Args:
    typical_prices: Array of typical prices (usually (H+L+C)/3)
    volumes: Array of volumes
    anchor_index: Starting index for VWAP calculation

  Returns:
    Array of VWAP values (NaN before anchor_index)
  """
  n = len(typical_prices)
  vwap = np.full(n, np.nan)

  if n == 0 or anchor_index >= n or anchor_index < 0:
    return vwap

  # Initialize cumulative sums
  cum_pv = 0.0  # Cumulative price * volume
  cum_v = 0.0  # Cumulative volume

  for i in range(anchor_index, n):
    # Update cumulative sums
    cum_pv += typical_prices[i] * volumes[i]
    cum_v += volumes[i]

    # Calculate VWAP (avoid division by zero)
    if cum_v > 0:
      vwap[i] = cum_pv / cum_v
    else:
      vwap[i] = typical_prices[i]  # Fallback to typical price

  return vwap
