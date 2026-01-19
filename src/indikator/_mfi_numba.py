"""Numba-optimized MFI (Money Flow Index) calculation.

This module contains JIT-compiled functions for MFI calculation.
Separated for better code organization and testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_mfi_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  close: NDArray[np.float64],
  volume: NDArray[np.float64],
  window: int,
  epsilon: float = 1e-9,
) -> NDArray[np.float64]:
  """Numba JIT-compiled MFI with inline Typical Price calculation.

  Optimized for memory bandwidth by calculating Typical Price inline (loop fusion).
  Uses a circular buffer for tracking positive/negative money flow changes,
  avoiding redundant logic re-evaluation for elements leaving the window.

  Args:
    high: Array of high prices
    low: Array of low prices
    close: Array of closing prices
    volume: Array of volumes
    window: Lookback period (typically 14)
    epsilon: Small value to prevent division by zero

  Returns:
    Array of MFI values (0-100 range, NaN for initial bars)
  """
  n = len(close)

  if n < window + 1:
    return np.full(n, np.nan)

  mfi = np.empty(n)
  mfi[:window] = np.nan

  # Circular buffers for Money Flow (stored as contributions)
  buf_pos = np.zeros(window, dtype=np.float64)
  buf_neg = np.zeros(window, dtype=np.float64)
  buf_idx = 0

  pos_sum = 0.0
  neg_sum = 0.0

  # First typical price at index 0 (used for diff at index 1)
  prev_tp = (high[0] + low[0] + close[0]) / 3.0

  # Initialization: fill buffer with first 'window' flows (indices 1 to window)
  for i in range(1, window + 1):
    curr_tp = (high[i] + low[i] + close[i]) / 3.0
    mf = curr_tp * volume[i]

    if curr_tp > prev_tp:
      buf_pos[buf_idx] = mf
      pos_sum += mf
      buf_neg[buf_idx] = 0.0
    elif curr_tp < prev_tp:
      buf_neg[buf_idx] = mf
      neg_sum += mf
      buf_pos[buf_idx] = 0.0
    else:
      buf_pos[buf_idx] = 0.0
      buf_neg[buf_idx] = 0.0

    prev_tp = curr_tp
    buf_idx += 1
    if buf_idx >= window:
      buf_idx = 0

  # Calculate MFI at index `window`
  total_flow = pos_sum + neg_sum
  if total_flow < epsilon:
    mfi[window] = 0.0
  else:
    mfi[window] = 100.0 * pos_sum / total_flow

  # Main Loop (window + 1 to n)
  for i in range(window + 1, n):
    # Remove oldest value (at current buf_idx)
    pos_sum -= buf_pos[buf_idx]
    neg_sum -= buf_neg[buf_idx]

    # Calculate new
    curr_tp = (high[i] + low[i] + close[i]) / 3.0
    mf = curr_tp * volume[i]

    if curr_tp > prev_tp:
      buf_pos[buf_idx] = mf
      pos_sum += mf
      buf_neg[buf_idx] = 0.0
    elif curr_tp < prev_tp:
      buf_neg[buf_idx] = mf
      neg_sum += mf
      buf_pos[buf_idx] = 0.0
    else:
      buf_pos[buf_idx] = 0.0
      buf_neg[buf_idx] = 0.0

    prev_tp = curr_tp
    buf_idx += 1
    if buf_idx >= window:
      buf_idx = 0

    # MFI
    total_flow = pos_sum + neg_sum
    if total_flow < epsilon:
      mfi[i] = 0.0
    else:
      mfi[i] = 100.0 * pos_sum / total_flow

  return mfi
