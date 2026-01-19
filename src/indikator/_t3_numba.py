"""Numba-optimized T3 Moving Average calculation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_t3_numba(
  prices: NDArray[np.float64],
  period: int,
  vfactor: float,
) -> NDArray[np.float64]:
  """Calculate T3 Moving Average using a fused single-pass kernel.

  T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
  where e1..e6 are cascaded EMAs.

  Uses SMA for initialization of each EMA stage (matching TA-Lib behavior).
  Assumes clean data (no NaNs) - validated by caller.
  """
  n = len(prices)

  # Each EMA needs 'period' values to produce first output (SMA init)
  # Total warmup: 6 * (period - 1) indices before first valid T3
  warmup = 6 * (period - 1)

  if n <= warmup:
    return np.full(n, np.nan, dtype=np.float64)

  out = np.empty(n, dtype=np.float64)

  alpha = 2.0 / (period + 1.0)
  one_minus_alpha = 1.0 - alpha

  # T3 coefficients
  v = vfactor
  v2 = v * v
  v3 = v2 * v
  c1 = -v3
  c2 = 3.0 * v2 + 3.0 * v3
  c3 = -6.0 * v2 - 3.0 * v - 3.0 * v3
  c4 = 1.0 + 3.0 * v + v3 + 3.0 * v2

  # Fill NaN for warmup
  for i in range(warmup):
    out[i] = np.nan

  # State variables for 6 cascaded EMAs
  # Each needs count for SMA phase, sum for SMA, and current EMA value
  cnt1 = 0
  sum1 = 0.0
  e1 = 0.0
  cnt2 = 0
  sum2 = 0.0
  e2 = 0.0
  cnt3 = 0
  sum3 = 0.0
  e3 = 0.0
  cnt4 = 0
  sum4 = 0.0
  e4 = 0.0
  cnt5 = 0
  sum5 = 0.0
  e5 = 0.0
  cnt6 = 0
  sum6 = 0.0
  e6 = 0.0

  for i in range(n):
    p = prices[i]

    # --- EMA 1 ---
    cnt1 += 1
    if cnt1 < period:
      sum1 += p
      v1_valid = False
    elif cnt1 == period:
      sum1 += p
      e1 = sum1 / period
      v1_valid = True
    else:
      e1 = alpha * p + one_minus_alpha * e1
      v1_valid = True

    # --- EMA 2 ---
    if v1_valid:
      cnt2 += 1
      if cnt2 < period:
        sum2 += e1
        v2_valid = False
      elif cnt2 == period:
        sum2 += e1
        e2 = sum2 / period
        v2_valid = True
      else:
        e2 = alpha * e1 + one_minus_alpha * e2
        v2_valid = True
    else:
      v2_valid = False

    # --- EMA 3 ---
    if v2_valid:
      cnt3 += 1
      if cnt3 < period:
        sum3 += e2
        v3_valid = False
      elif cnt3 == period:
        sum3 += e2
        e3 = sum3 / period
        v3_valid = True
      else:
        e3 = alpha * e2 + one_minus_alpha * e3
        v3_valid = True
    else:
      v3_valid = False

    # --- EMA 4 ---
    if v3_valid:
      cnt4 += 1
      if cnt4 < period:
        sum4 += e3
        v4_valid = False
      elif cnt4 == period:
        sum4 += e3
        e4 = sum4 / period
        v4_valid = True
      else:
        e4 = alpha * e3 + one_minus_alpha * e4
        v4_valid = True
    else:
      v4_valid = False

    # --- EMA 5 ---
    if v4_valid:
      cnt5 += 1
      if cnt5 < period:
        sum5 += e4
        v5_valid = False
      elif cnt5 == period:
        sum5 += e4
        e5 = sum5 / period
        v5_valid = True
      else:
        e5 = alpha * e4 + one_minus_alpha * e5
        v5_valid = True
    else:
      v5_valid = False

    # --- EMA 6 ---
    if v5_valid:
      cnt6 += 1
      if cnt6 < period:
        sum6 += e5
      elif cnt6 == period:
        sum6 += e5
        e6 = sum6 / period
        out[i] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
      else:
        e6 = alpha * e5 + one_minus_alpha * e6
        out[i] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

  return out
