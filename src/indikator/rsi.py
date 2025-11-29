"""Relative Strength Index (RSI) indicator module.

This module provides RSI calculation, a momentum oscillator that measures
the speed and magnitude of price changes. One of the most popular technical
indicators.
"""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from hipr import Ge, Gt, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
    Finite,
    NonEmpty,
    Validated,
    validated,
)

from indikator._rsi_numba import compute_rsi_numba


@configurable
@validated
def rsi(
    data: Validated[pd.Series, Finite, NonEmpty],
    window: Hyper[int, Ge[2]] = 14,
    epsilon: Hyper[float, Gt[0.0]] = 1e-9,
) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI is a momentum oscillator that measures the speed and magnitude of
    price changes. It oscillates between 0 and 100, with readings above 70
    typically considered overbought and below 30 considered oversold.

    Formula:
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over N periods

    Uses Wilder's smoothing method for averaging:
    - First average: simple average of gains/losses over 'window' periods
    - Subsequent averages: (previous avg * (window-1) + current value) / window

    Interpretation:
    - RSI > 70: Overbought (potential reversal down)
    - RSI < 30: Oversold (potential reversal up)
    - RSI = 50: Neutral (no clear momentum)
    - RSI crossing 50: Momentum shift (bullish if crossing up, bearish if down)
    - Divergence: RSI making higher lows while price makes lower lows = bullish

    Common strategies:
    - Mean reversion: Sell when RSI > 70, buy when RSI < 30
    - Trend following: Buy when RSI crosses above 50 in uptrend
    - Divergence trading: Look for price/RSI divergences

    Features:
    - Numba-optimized for performance
    - Wilder's smoothing (original method)
    - Handles edge cases (no losses, no gains)
    - Works with any numeric column

    Args:
      data: Input Series.
      window: Lookback period (default: 14, Wilder's original)
      epsilon: Small value to prevent division by zero

    Returns:
      Series with RSI values (0-100 range)

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
      >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
      >>> result = rsi(prices, window=5)
      >>> # Returns RSI values (typically 30-70 range)
    """
    # Convert to numpy for Numba
    values = data.values.astype(np.float64)

    # Calculate RSI using Numba-optimized function
    rsi_values = compute_rsi_numba(values, window, epsilon)

    return pd.Series(rsi_values, index=data.index, name="rsi")
