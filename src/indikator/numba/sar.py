"""Numba-optimized Parabolic SAR calculation.

Uses state machine with acceleration factor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_sar_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  af_start: float,
  af_increment: float,
  af_max: float,
) -> NDArray[np.float64]:  # pragma: no cover
  """Numba JIT-compiled Parabolic SAR.

  SAR tracks price with an acceleration factor that increases
  when new extremes are made.

  Uses register variables for state machine optimization.

  Args:
    high: High prices
    low: Low prices
    af_start: Starting acceleration factor (typically 0.02)
    af_increment: AF increment on new extreme (typically 0.02)
    af_max: Maximum AF (typically 0.2)

  Returns:
    Array of SAR values
  """
  n = len(high)

  if n < 2:
    return np.full(n, np.nan)

  sar = np.empty(n, dtype=np.float64)
  sar[0] = np.nan

  # TA-Lib initial trend detection: looks at first two bars
  # If close would be above SAR (using low[0] as potential long SAR), start long
  # Actually TA-Lib starts with long if high[1] > high[0]
  # SAR[1] is set to the extreme point OF THE OPPOSITE trend direction

  # TA-Lib logic:
  # If starting LONG: SAR[1] = lowest low of first 2 bars, EP = highest high
  # If starting SHORT: SAR[1] = highest high of first 2 bars, EP = lowest low

  # Determine trend from first two highs (TA-Lib approach)
  if high[1] > high[0]:
    # Uptrend
    is_long = True
    ep = high[1]  # Extreme point is highest high
    sar_value = min(low[0], low[1])  # SAR is the lowest low
  else:
    # Downtrend
    is_long = False
    ep = low[1]  # Extreme point is lowest low
    sar_value = max(high[0], high[1])  # SAR is the highest high

  af = af_start
  sar[1] = sar_value

  for i in range(2, n):
    # Calculate SAR for today based on yesterday's state
    sar_value += af * (ep - sar_value)

    # SAR cannot penetrate prior two bars (but skip initial bar where SAR was set)
    if is_long:
      # In uptrend, SAR cannot be above prior two lows
      if i > 2:
        sar_value = min(sar_value, low[i - 1], low[i - 2])
      else:
        sar_value = min(sar_value, low[i - 1])
    # In downtrend, SAR cannot be below prior two highs
    elif i > 2:
      sar_value = max(sar_value, high[i - 1], high[i - 2])
    else:
      sar_value = max(sar_value, high[i - 1])

    # Check for reversal
    if is_long:
      # Check if price pierced SAR (low went below SAR)
      if low[i] < sar_value:
        # Reversal to downtrend
        is_long = False
        sar_value = ep  # SAR becomes prior EP
        ep = low[i]  # New EP is current low
        af = af_start
      # Continue uptrend
      elif high[i] > ep:
        ep = high[i]
        af = min(af + af_increment, af_max)
    # Check if price pierced SAR (high went above SAR)
    elif high[i] > sar_value:
      # Reversal to uptrend
      is_long = True
      sar_value = ep  # SAR becomes prior EP
      ep = high[i]  # New EP is current high
      af = af_start
    # Continue downtrend
    elif low[i] < ep:
      ep = low[i]
      af = min(af + af_increment, af_max)

    sar[i] = sar_value

  return sar


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_sarext_numba(
  high: NDArray[np.float64],
  low: NDArray[np.float64],
  start_value: float,
  offset_on_reversal: float,
  acceleration_init_long: float,
  acceleration_long: float,
  acceleration_max_long: float,
  acceleration_init_short: float,
  acceleration_short: float,
  acceleration_max_short: float,
) -> NDArray[np.float64]:
  """Compute SAREXT - Parabolic SAR Extended.

  Matches TA-Lib SAREXT logic.
  """
  n = len(high)
  sar = np.empty(n, dtype=np.float64)

  if n < 2:
    sar[:] = np.nan
    return sar

  # Initialization logic from TA-Lib:
  # If start_value > 0, use it. If < 0, use high-start.
  # For simplicity, we assume start_value logic is handled in wrapper or match standard.

  h0 = high[0]
  h1 = high[1]
  l0 = low[0]
  l1 = low[1]

  # Determine initial trend (Standard logic for now)
  if h1 > h0:
    is_long = True
    ep = h1
    sar_val = min(l0, l1)
  else:
    is_long = False
    ep = l1
    sar_val = max(h0, h1)

  if start_value != 0:
    sar_val = abs(start_value)
    is_long = start_value > 0

  af = acceleration_init_long if is_long else acceleration_init_short
  sar[0] = np.nan
  sar[1] = sar_val

  for i in range(2, n):
    hi = high[i]
    li = low[i]
    hi_p1 = high[i - 1]
    li_p1 = low[i - 1]

    sar_val += af * (ep - sar_val)

    if is_long:
      if i > 2:
        hi_p2 = high[i - 2]
        li_p2 = low[i - 2]
        sar_val = min(sar_val, li_p1, li_p2)
      else:
        sar_val = min(sar_val, li_p1)
    elif i > 2:
      hi_p2 = high[i - 2]
      li_p2 = low[i - 2]
      sar_val = max(sar_val, hi_p1, hi_p2)
    else:
      sar_val = max(sar_val, hi_p1)

    if is_long:
      if li < sar_val:
        # Reversal to short
        is_long = False
        # Adjustment on reversal
        sar_val = ep + offset_on_reversal if offset_on_reversal != 0 else ep
        ep = li
        af = acceleration_init_short
      elif hi > ep:
        ep = hi
        af = min(af + acceleration_long, acceleration_max_long)
    elif hi > sar_val:
      # Reversal to long
      is_long = True
      sar_val = ep - offset_on_reversal if offset_on_reversal != 0 else ep
      ep = hi
      af = acceleration_init_long
    elif li < ep:
      ep = li
      af = min(af + acceleration_short, acceleration_max_short)

    sar[i] = sar_val

  return sar
