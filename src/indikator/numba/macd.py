"""Numba-optimized MACD calculation.

This module contains JIT-compiled functions for MACD calculation.
Matches TA-lib MACD exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit
import numpy as np

if TYPE_CHECKING:
  from numpy.typing import NDArray


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_macd_numba(
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

  lookback_macd = slow_period - 1
  lookback_signal = lookback_macd + signal_period - 1
  total_lookback = lookback_signal

  if n <= total_lookback:
    return macd, signal, histogram

  # EMA multipliers
  k_fast = 2.0 / (fast_period + 1)
  k_slow = 2.0 / (slow_period + 1)
  k_signal = 2.0 / (signal_period + 1)

  # 1. Initialize Fast EMA at index fast_period - 1
  fast_ema = 0.0
  for i in range(fast_period):
    fast_ema += prices[i]
  fast_ema /= fast_period

  # 2. Advance Fast EMA until index slow_period - 1
  for i in range(fast_period, slow_period):
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)

  # 3. Initialize Slow EMA at index slow_period - 1
  slow_ema = 0.0
  for i in range(slow_period):
    slow_seed = 0.0  # Unused in this loop context
    slow_ema += prices[i]
  slow_ema /= slow_period

  # 4. Calculate first signal_period MACD values to seed signal line
  # Both fast_ema and slow_ema are currently at index slow_period - 1
  macd_values = np.zeros(signal_period)
  macd_values[0] = fast_ema - slow_ema

  for j in range(1, signal_period):
    i = slow_period - 1 + j
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)
    slow_ema = prices[i] * k_slow + slow_ema * (1.0 - k_slow)
    macd_values[j] = fast_ema - slow_ema

  # 5. Initialize Signal line at index total_lookback
  signal_ema = 0.0
  for j in range(signal_period):
    signal_ema += macd_values[j]
  signal_ema /= signal_period

  # Store first synchronized output values
  macd_val = macd_values[signal_period - 1]
  macd[total_lookback] = macd_val
  signal[total_lookback] = signal_ema
  histogram[total_lookback] = macd_val - signal_ema

  # 6. Continue for remaining values
  for i in range(total_lookback + 1, n):
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)
    slow_ema = prices[i] * k_slow + slow_ema * (1.0 - k_slow)
    macd_val = fast_ema - slow_ema
    signal_ema = macd_val * k_signal + signal_ema * (1.0 - k_signal)

    macd[i] = macd_val
    signal[i] = signal_ema
    histogram[i] = macd_val - signal_ema

  return macd, signal, histogram


@jit(nopython=True, cache=True, nogil=True)  # pragma: no cover
def compute_macdfix_numba(
  prices: NDArray[np.float64],
  signal_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Numba JIT-compiled MACDFIX calculation matching TA-lib exactly.

  Uses legacy multipliers: Fast=0.15, Slow=0.075.
  """
  n = len(prices)
  macd = np.full(n, np.nan)
  signal = np.full(n, np.nan)
  histogram = np.full(n, np.nan)

  # TA-lib MACDFIX uses periods 12 and 26 for lookback calculation
  # but uses the constants for the multipliers.
  lookback_macd = 26 - 1
  lookback_signal = lookback_macd + signal_period - 1
  total_lookback = lookback_signal

  if n <= total_lookback:
    return macd, signal, histogram

  k_fast = 0.15
  k_slow = 0.075
  k_signal = 2.0 / (signal_period + 1)

  # 1. Initialize Fast EMA at index 11
  fast_ema = 0.0
  for i in range(12):
    fast_ema += prices[i]
  fast_ema /= 12.0

  # 2. Advance Fast EMA until index 25
  for i in range(12, 26):
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)

  # 3. Initialize Slow EMA at index 25
  slow_ema = 0.0
  for i in range(26):
    slow_ema += prices[i]
  slow_ema /= 26.0

  # 4. Calculate first signal_period MACD values
  macd_values = np.zeros(signal_period)
  macd_values[0] = fast_ema - slow_ema

  for j in range(1, signal_period):
    i = 25 + j
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)
    slow_ema = prices[i] * k_slow + slow_ema * (1.0 - k_slow)
    macd_values[j] = fast_ema - slow_ema

  # 5. Initialize Signal line
  signal_ema = 0.0
  for j in range(signal_period):
    signal_ema += macd_values[j]
  signal_ema /= signal_period

  macd_val = macd_values[signal_period - 1]
  macd[total_lookback] = macd_val
  signal[total_lookback] = signal_ema
  histogram[total_lookback] = macd_val - signal_ema

  # 6. Continue
  for i in range(total_lookback + 1, n):
    fast_ema = prices[i] * k_fast + fast_ema * (1.0 - k_fast)
    slow_ema = prices[i] * k_slow + slow_ema * (1.0 - k_slow)
    macd_val = fast_ema - slow_ema
    signal_ema = macd_val * k_signal + signal_ema * (1.0 - k_signal)

    macd[i] = macd_val
    signal[i] = signal_ema
    histogram[i] = macd_val - signal_ema

  return macd, signal, histogram
