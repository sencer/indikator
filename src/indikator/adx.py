"""ADX (Average Directional Index) indicator module.

This module provides ADX calculation, a trend strength indicator that
measures how strong a trend is, regardless of direction.
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

from indikator._adx_numba import compute_adx_numba


@configurable
@validate
def adx(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> pd.DataFrame:
  """Calculate Average Directional Index (ADX).

  ADX measures trend strength regardless of direction. It's derived from
  the Directional Movement System developed by Welles Wilder.

  Components:
  - ADX: Average Directional Index (trend strength)
  - +DI: Plus Directional Indicator (bullish pressure)
  - -DI: Minus Directional Indicator (bearish pressure)

  Interpretation:
  - ADX < 20: Weak trend / ranging market
  - ADX 20-25: Trend emerging
  - ADX 25-50: Strong trend
  - ADX 50-75: Very strong trend
  - ADX > 75: Extremely strong trend

  Directional Indicators:
  - +DI > -DI: Bullish
  - -DI > +DI: Bearish
  - +DI crossing above -DI: Buy signal
  - -DI crossing above +DI: Sell signal

  Common strategies:
  - Trade only when ADX > 25 (confirms trend)
  - Use DI crossovers for entry signals
  - Exit when ADX starts declining

  Features:
  - Numba-optimized for performance
  - Wilder's smoothing method
  - Returns ADX and both directional indicators

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    DataFrame with 'adx', 'plus_di', 'minus_di' columns

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> high = pd.Series([105, 107, 106, 108, 110])
    >>> low = pd.Series([100, 102, 101, 103, 105])
    >>> close = pd.Series([102, 105, 104, 106, 108])
    >>> result = adx(high, low, close)
  """
  # Convert to numpy for Numba
  high_arr = cast(
    "NDArray[np.float64]",
    high.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )
  low_arr = cast(
    "NDArray[np.float64]",
    low.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )
  close_arr = cast(
    "NDArray[np.float64]",
    close.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate ADX using Numba-optimized function
  adx_values, plus_di, minus_di = compute_adx_numba(
    high_arr, low_arr, close_arr, period
  )

  return pd.DataFrame(
    {"adx": adx_values, "plus_di": plus_di, "minus_di": minus_di},
    index=close.index,
  )
