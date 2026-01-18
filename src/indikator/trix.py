"""TRIX (Triple Exponential Average) indicator module.

This module provides TRIX calculation, a momentum oscillator that shows
the percent rate of change of a triple exponentially smoothed moving
average of prices.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._trix_numba import compute_trix_numba


@configurable
@validate
def trix(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> pd.Series:
  """Calculate TRIX (Triple Exponential Average).

  TRIX is a momentum oscillator that displays the percent rate of change
  of a triple exponentially smoothed moving average. It filters out minor
  price movements, making it useful for identifying overbought/oversold
  conditions and divergences.

  Formula:
  1. EMA1 = EMA(Price, period)
  2. EMA2 = EMA(EMA1, period)
  3. EMA3 = EMA(EMA2, period)
  4. TRIX = ((EMA3[today] - EMA3[yesterday]) / EMA3[yesterday]) * 100

  Interpretation:
  - TRIX > 0: Bullish momentum (triple EMA rising)
  - TRIX < 0: Bearish momentum (triple EMA falling)
  - Zero line crossovers: Trend change signals
  - Divergence: TRIX direction differs from price (reversal signal)

  Common strategies:
  - Signal line: Use 9-period EMA of TRIX as signal line
  - Zero line crossovers: Buy when TRIX crosses above 0
  - Divergence trading: Look for price/TRIX divergences

  Features:
  - Numba-optimized for performance
  - Triple smoothing eliminates short-term noise
  - Shows rate of change, not absolute values
  - Good for identifying trend changes

  Args:
    data: Input Series (typically closing prices)
    period: EMA period (default: 14)

  Returns:
    Series with TRIX values (percentage)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 5)
    >>> result = trix(prices, period=5)
    >>> # Returns TRIX values (typically small percentages near 0)
  """
  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate TRIX using Numba-optimized function
  trix_values = compute_trix_numba(values, period)

  return pd.Series(trix_values, index=data.index, name="trix")
