"""Numba-optimized MACD calculation.

This module contains JIT-compiled functions for MACD calculation.
Separated for better code organization and testability.
"""
# pyright: reportAny=false

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import jit  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@jit(nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator]  # pragma: no cover
def compute_ema_numba(
    prices: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Numba JIT-compiled EMA calculation.

    EMA = Price[today] * k + EMA[yesterday] * (1 - k)
    where k = 2 / (window + 1)

    Handles NaN values by skipping them in the SMA calculation and
    only calculating EMA where prices are not NaN.

    Args:
      prices: Array of prices (may contain NaN)
      window: EMA period

    Returns:
      Array of EMA values (NaN for initial bars or where input is NaN)
    """
    n = len(prices)
    ema = np.full(n, np.nan)

    if n < window:
        return ema

    # Calculate smoothing factor
    k = 2.0 / (window + 1.0)

    # Find first window of non-NaN values
    found_start = False
    start_idx = 0

    for i in range(n - window + 1):
        # Check if we have window consecutive non-NaN values
        all_valid = True
        for j in range(i, i + window):
            if np.isnan(prices[j]):
                all_valid = False
                break

        if all_valid:
            start_idx = i
            found_start = True
            break

    if not found_start:
        return ema

    # Initialize with SMA of first window
    sma = 0.0
    for i in range(start_idx, start_idx + window):
        sma += prices[i]
    sma /= window
    ema[start_idx + window - 1] = sma

    # Calculate EMA
    for i in range(start_idx + window, n):
        if not np.isnan(prices[i]) and not np.isnan(ema[i - 1]):
            ema[i] = prices[i] * k + ema[i - 1] * (1.0 - k)

    return ema


@jit(nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator]  # pragma: no cover
def compute_macd_numba(
    prices: NDArray[np.float64],
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Numba JIT-compiled MACD calculation.

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Args:
      prices: Array of prices
      fast_period: Fast EMA period (typically 12)
      slow_period: Slow EMA period (typically 26)
      signal_period: Signal EMA period (typically 9)

    Returns:
      Tuple of (macd, signal, histogram) arrays
    """
    n = len(prices)

    # Calculate fast and slow EMAs
    fast_ema = compute_ema_numba(prices, fast_period)
    slow_ema = compute_ema_numba(prices, slow_period)

    # Calculate MACD line
    macd = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]):
            macd[i] = fast_ema[i] - slow_ema[i]

    # Calculate signal line (EMA of MACD)
    signal = compute_ema_numba(macd, signal_period)

    # Calculate histogram
    histogram = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(macd[i]) and not np.isnan(signal[i]):
            histogram[i] = macd[i] - signal[i]

    return macd, signal, histogram
