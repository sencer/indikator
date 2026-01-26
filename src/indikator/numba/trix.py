"""Numba-optimized TRIX (Triple Exponential Average) calculation.

This module contains JIT-compiled functions for TRIX calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True, fastmath=True)  # pragma: no cover
def compute_trix_numba(
  prices: NDArray[np.float64],
  period: int,
) -> NDArray[np.float64]:
  """Numba JIT-compiled TRIX calculation with Loop Fusion."""
  n = len(prices)
  trix = np.full(n, np.nan)

  # Need at least 3 * period - 2 bars for first valid TRIX
  min_bars = 3 * period - 2
  if n < min_bars + 1:
    return trix

  k = 2.0 / (period + 1)

  # State variables
  ema1 = 0.0
  ema2 = 0.0
  ema3 = 0.0

  # Accumulators for SMA seeding
  sma1_sum = 0.0
  sma2_sum = 0.0
  sma3_sum = 0.0

  # Initialization handling
  # We need to compute EMA1 first, then we can feed EMA2 accumulator, etc.
  # But we can do it in one pass if we are careful with phases.

  # Phase 1: SMA1 Setup (0 to period-1)
  for i in range(period):
    sma1_sum += prices[i]

  ema1 = sma1_sum / period

  # Setup dependent accumulators?
  # The first EMA1 value is at index 'period-1'.
  # This feeds into SMA2 accumulation.
  sma2_sum += ema1

  # Phase 2: Iterate and propogate
  # We iterate from period to n

  # Track count of valid values for seeding
  # ema1 count (valid values produced including initial)
  valid_ema1_count = 1
  valid_ema2_count = 0

  epsilon = 1e-10

  for i in range(period, n):
    # Update EMA1
    ema1 = prices[i] * k + ema1 * (1.0 - k)
    valid_ema1_count += 1

    # Handle EMA2
    # We need 'period' values of ema1 to seed ema2
    if valid_ema1_count <= period:
      sma2_sum += ema1
      if valid_ema1_count == period:
        # Seed EMA2
        ema2 = sma2_sum / period
        valid_ema2_count = 1
        # Feed to SMA3
        sma3_sum += ema2
    else:
      # EMA2 running
      ema2 = ema1 * k + ema2 * (1.0 - k)
      valid_ema2_count += 1

      # Handle EMA3
      if valid_ema2_count <= period:
        if (
          valid_ema2_count > 1
        ):  # Already added the first one at seeding time? No, wait logic flow.
          # When valid_ema2_count became 1 (prev step), we added to sma3_sum.
          # Now we add current.
          sma3_sum += ema2

        if valid_ema2_count == period:
          # Seed EMA3
          ema3 = sma3_sum / period
          # Ready for TRIX?
          # First TRIX needs 2 values of EMA3 (curr and prev).
          # This is the first value.
          # Next step we can calc TRIX.
      else:
        # EMA3 running
        prev_ema3 = ema3
        ema3 = ema2 * k + ema3 * (1.0 - k)

        # Calculate TRIX
        if abs(prev_ema3) > epsilon:
          trix[i] = ((ema3 - prev_ema3) / prev_ema3) * 100.0
        else:
          trix[i] = 0.0

  return trix
