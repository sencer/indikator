"""Numba-optimized Stochastic RSI calculation.

Applies Stochastic formula to RSI values for more sensitive oscillator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def compute_stochrsi_numba(
  prices: NDArray[np.float64],
  rsi_period: int,
  stoch_period: int,
  k_period: int,
  d_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # pragma: no cover
  """Numba JIT-compiled StochRSI matching TA-Lib exactly.

  TA-Lib StochRSI:
  1. RSI starts at index rsi_period
  2. fastk = (RSI - min(RSI, stoch_period)) / (max(RSI, stoch_period) - min(RSI, stoch_period)) * 100
     First valid fastk at index rsi_period + stoch_period
  3. fastd = SMA(fastk, d_period)
     First valid fastd at index rsi_period + stoch_period + d_period - 1

  Note: TA-Lib k_period is stoch_period, fastk_period is used for the Stochastic calculation

  Args:
    prices: Price array
    rsi_period: RSI lookback (typically 14)
    stoch_period: Stochastic lookback on RSI (typically 14)
    k_period: Not used by TA-Lib in standard StochRSI (pass same as stoch_period)
    d_period: %D SMA smoothing period (typically 3)

  Returns:
    Tuple of (fastk, fastd)
  """
  n = len(prices)

  # First valid indices
  rsi_start = rsi_period  # First RSI at rsi_period
  fastk_start = rsi_period + stoch_period  # First fastk at rsi_period + stoch_period
  fastd_start = fastk_start + d_period - 1  # First fastd needs d_period fastk values

  if n <= fastd_start:
    return np.full(n, np.nan), np.full(n, np.nan)

  fastk = np.full(n, np.nan)
  fastd = np.full(n, np.nan)

  # Step 1: Compute RSI
  rsi = np.full(n, np.nan)

  # RSI initialization using Wilder's smoothing
  avg_gain = 0.0
  avg_loss = 0.0

  for i in range(1, rsi_period + 1):
    change = prices[i] - prices[i - 1]
    if change > 0:
      avg_gain += change
    else:
      avg_loss -= change

  avg_gain /= rsi_period
  avg_loss /= rsi_period

  if avg_loss < 1e-10:
    rsi[rsi_period] = 100.0
  else:
    rs = avg_gain / avg_loss
    rsi[rsi_period] = 100.0 - 100.0 / (1.0 + rs)

  inv_period = 1.0 / rsi_period
  period_m1 = rsi_period - 1

  for i in range(rsi_period + 1, n):
    change = prices[i] - prices[i - 1]
    gain = change if change > 0.0 else 0.0
    loss = -change if change < 0.0 else 0.0

    avg_gain = (avg_gain * period_m1 + gain) * inv_period
    avg_loss = (avg_loss * period_m1 + loss) * inv_period

    if avg_loss < 1e-10:
      rsi[i] = 100.0
    else:
      rs = avg_gain / avg_loss
      rsi[i] = 100.0 - 100.0 / (1.0 + rs)

  # Step 2: Compute raw stochastic on RSI (fastk)
  # Using lazy rescan for min/max
  max_idx, max_val = -1, -np.inf
  min_idx, min_val = -1, np.inf

  for i in range(fastk_start, n):
    trailing = i - stoch_period + 1  # Window is [trailing, i]

    # Update max
    if max_idx < trailing:
      max_idx, max_val = trailing, rsi[trailing]
      for k in range(trailing + 1, i + 1):
        if rsi[k] >= max_val:
          max_val = rsi[k]
          max_idx = k
    elif rsi[i] >= max_val:
      max_val = rsi[i]
      max_idx = i

    # Update min
    if min_idx < trailing:
      min_idx, min_val = trailing, rsi[trailing]
      for k in range(trailing + 1, i + 1):
        if rsi[k] <= min_val:
          min_val = rsi[k]
          min_idx = k
    elif rsi[i] <= min_val:
      min_val = rsi[i]
      min_idx = i

    # StochRSI fastk
    rng = max_val - min_val
    if rng > 1e-10:
      fastk[i] = 100.0 * (rsi[i] - min_val) / rng
    else:
      fastk[i] = 100.0  # TA-Lib returns 100 when range is 0

  # Step 3: Compute fastd = SMA(fastk, d_period) using O(1) rolling sum
  inv_d = 1.0 / d_period

  # Initialize sum for first fastd
  d_sum = 0.0
  for j in range(fastd_start - d_period + 1, fastd_start + 1):
    d_sum += fastk[j]
  fastd[fastd_start] = d_sum * inv_d

  # Main loop with O(1) update
  for i in range(fastd_start + 1, n):
    d_sum = d_sum - fastk[i - d_period] + fastk[i]
    fastd[i] = d_sum * inv_d

  return fastk, fastd
