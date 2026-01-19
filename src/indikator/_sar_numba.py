"""Numba-optimized Parabolic SAR calculation.

Uses state machine with acceleration factor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
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
    sar_value = sar_value + af * (ep - sar_value)

    # SAR cannot penetrate prior two bars (but skip initial bar where SAR was set)
    if is_long:
      # In uptrend, SAR cannot be above prior two lows
      if i > 2:
        sar_value = min(sar_value, low[i - 1], low[i - 2])
      else:
        sar_value = min(sar_value, low[i - 1])
    else:
      # In downtrend, SAR cannot be below prior two highs
      if i > 2:
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
