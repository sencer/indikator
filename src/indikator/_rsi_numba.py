"""Numba-optimized RSI (Relative Strength Index) calculation.

This module contains JIT-compiled functions for RSI calculation.
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
def compute_rsi_numba(
    prices: NDArray[np.float64],
    window: int,
    epsilon: float = 1e-9,
) -> NDArray[np.float64]:
    """Numba JIT-compiled RSI calculation using Wilder's smoothing.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Uses Wilder's smoothing method:
    - First average: simple average of gains/losses over 'window' periods
    - Subsequent averages: (previous avg * (window-1) + current value) / window

    Args:
      prices: Array of prices (typically closing prices)
      window: Lookback period (typically 14)
      epsilon: Small value to prevent division by zero

    Returns:
      Array of RSI values (NaN for initial bars where window not satisfied)
    """
    n = len(prices)
    rsi = np.full(n, np.nan)

    if n < window + 1:  # Need at least window+1 bars (for first change)
        return rsi

    # Calculate price changes
    changes = np.zeros(n)
    for i in range(1, n):
        changes[i] = prices[i] - prices[i - 1]

    # Separate gains and losses
    gains = np.zeros(n)
    losses = np.zeros(n)

    for i in range(1, n):
        if changes[i] > 0:
            gains[i] = changes[i]
        else:
            losses[i] = -changes[i]  # Store as positive value

    # Calculate initial average gain and loss (simple average)
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(1, window + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]

    avg_gain /= window
    avg_loss /= window

    # Calculate first RSI
    if avg_loss < epsilon:
        rsi[window] = 100.0  # No losses, RSI = 100
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(window + 1, n):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

        if avg_loss < epsilon:
            rsi[i] = 100.0  # No losses, RSI = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi
