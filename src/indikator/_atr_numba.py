"""Numba-optimized ATR (Average True Range) calculation.

This module contains JIT-compiled functions for ATR calculation.
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
def compute_true_range_numba(
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Numba JIT-compiled true range calculation.

    True Range is the greatest of:
    1. Current high - current low
    2. |Current high - previous close|
    3. |Current low - previous close|

    For the first bar, true range equals high - low since there's no previous close.

    Args:
      highs: Array of high prices
      lows: Array of low prices
      closes: Array of close prices

    Returns:
      Array of true range values
    """
    n = len(highs)
    tr = np.zeros(n, dtype=np.float64)

    if n == 0:
        return tr

    # First bar: TR = high - low
    tr[0] = highs[0] - lows[0]

    # Subsequent bars: TR = max(high-low, |high-prev_close|, |low-prev_close|)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    return tr


@jit(nopython=True)  # pyright: ignore[reportUntypedFunctionDecorator]  # pragma: no cover
def compute_atr_numba(
    true_ranges: NDArray[np.float64],
    window: int,
) -> NDArray[np.float64]:
    """Numba JIT-compiled ATR calculation using Wilder's smoothing.

    Uses Wilder's smoothing method (similar to EMA but with different smoothing factor):
    ATR = (ATR_previous * (window - 1) + TR_current) / window

    For the initial ATR, uses simple moving average of first 'window' true ranges.

    Args:
      true_ranges: Array of true range values
      window: Smoothing period (typically 14)

    Returns:
      Array of ATR values (NaN for initial bars where window not satisfied)
    """
    n = len(true_ranges)
    atr = np.full(n, np.nan)

    if n < window:
        return atr

    # Calculate initial ATR as simple average of first 'window' true ranges
    sum_tr = 0.0
    for i in range(window):
        sum_tr += true_ranges[i]
    atr[window - 1] = sum_tr / window

    # Use Wilder's smoothing for subsequent values
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + true_ranges[i]) / window

    return atr
