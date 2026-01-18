"""Numba-optimized MACD calculation.

This module contains JIT-compiled functions for MACD calculation.
Matches TA-lib MACD exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_macd_numba(  # noqa: PLR0914
  prices: NDArray[np.float64],
  fast_period: int,
  slow_period: int,
  signal_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled MACD calculation matching TA-lib exactly.

  MACD = EMA(fast) - EMA(slow)
  Signal = EMA(MACD, signal_period)
  Histogram = MACD - Signal

  TA-lib MACD seeds EMAs as follows:
  - Fast EMA: SMA of prices[slow-fast : slow]
  - Slow EMA: SMA of prices[0 : slow]

  The signal line is seeded with SMA of the first signal_period MACD values,
  where the first MACD value is fast_seed - slow_seed (at conceptual index
  slow_period - 1).

  Args:
    prices: Array of prices (must not contain NaN)
    fast_period: Fast EMA period (typically 12)
    slow_period: Slow EMA period (typically 26)
    signal_period: Signal EMA period (typically 9)

  Returns:
    Tuple of (macd, signal, histogram) arrays
  """
  n = len(prices)
  macd = np.full(n, np.nan)
  signal = np.full(n, np.nan)
  histogram = np.full(n, np.nan)

  lookback = slow_period + signal_period - 2
  if n <= lookback:
    return macd, signal, histogram

  # EMA multipliers
  k_fast = 2.0 / (fast_period + 1)
  k_slow = 2.0 / (slow_period + 1)
  k_signal = 2.0 / (signal_period + 1)

  # TA-lib seeds:
  # Fast EMA: SMA of prices[slow_period - fast_period : slow_period]
  # Slow EMA: SMA of prices[0 : slow_period]
  fast_start = slow_period - fast_period
  fast_seed = 0.0
  for i in range(fast_start, slow_period):
    fast_seed += prices[i]
  fast_seed /= fast_period

  slow_seed = 0.0
  for i in range(slow_period):
    slow_seed += prices[i]
  slow_seed /= slow_period

  # Initialize EMAs
  fast_ema = fast_seed
  slow_ema = slow_seed

  # Signal seed: SMA of first signal_period MACD values
  # First MACD value is fast_seed - slow_seed (at conceptual index 25)
  # Then MACD values at indices 26, 27, ..., 33 (signal_period - 1 more values)
  macd_values = np.zeros(signal_period)
  macd_values[0] = fast_seed - slow_seed  # MACD at index 25

  # Calculate MACD values at indices 26 through 33 (signal_period - 1 values)
  for j in range(1, signal_period):
    i = slow_period - 1 + j  # indices 26, 27, ..., 33
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)
    slow_ema = prices[i] * k_slow + slow_ema * (1.0 - k_slow)
    macd_values[j] = fast_ema - slow_ema

  # Signal seed is SMA of these signal_period MACD values
  signal_ema = 0.0
  for j in range(signal_period):
    signal_ema += macd_values[j]
  signal_ema /= signal_period

  # At this point, fast_ema and slow_ema are at index 33 (lookback)
  # First output
  macd_val = macd_values[signal_period - 1]  # MACD at index 33
  macd[lookback] = macd_val
  signal[lookback] = signal_ema
  histogram[lookback] = macd_val - signal_ema

  # Continue for remaining values
  for i in range(lookback + 1, n):
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)
    slow_ema = prices[i] * k_slow + slow_ema * (1.0 - k_slow)
    macd_val = fast_ema - slow_ema
    signal_ema = macd_val * k_signal + signal_ema * (1.0 - k_signal)

    macd[i] = macd_val
    signal[i] = signal_ema
    histogram[i] = macd_val - signal_ema

  return macd, signal, histogram
