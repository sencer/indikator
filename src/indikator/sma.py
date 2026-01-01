"""SMA (Simple Moving Average) indicator module.

This module provides SMA calculation, a trend-following indicator that
averages prices over a specified period.
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

from indikator._sma_numba import compute_sma_numba


@configurable
@validate
def sma(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> pd.Series:
  """Calculate Simple Moving Average (SMA).

  SMA is a trend-following indicator that averages prices over a specified
  period. All prices are weighted equally.

  Formula:
  SMA = (P1 + P2 + ... + Pn) / n

  Interpretation:
  - Price above SMA: Bullish
  - Price below SMA: Bearish
  - SMA crossovers: Trend change signals
  - Multiple SMAs: Short crossing long = golden/death cross

  Common periods:
  - 10/20: Short-term trend
  - 50: Medium-term trend
  - 200: Long-term trend

  Features:
  - Numba-optimized for performance
  - Rolling sum algorithm for O(n) efficiency
  - Works with any numeric column

  Args:
    data: Input Series.
    period: Lookback period (default: 20)

  Returns:
    Series with SMA values

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> result = sma(prices, period=5)
  """
  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate SMA using Numba-optimized function
  sma_values = compute_sma_numba(values, period)

  return pd.Series(sma_values, index=data.index, name="sma")
