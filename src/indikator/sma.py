"""SMA (Simple Moving Average) indicator module.

This module provides SMA calculation, a trend-following indicator that
averages prices over a specified period.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import SMAResult
from indikator._sma_numba import compute_sma_numba
from indikator.utils import to_numpy


@configurable
@validate
def sma(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> SMAResult:
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
      SMAResult(index, sma)

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
  from indikator.utils import to_numpy
      >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
      >>> result = sma(prices, period=5).to_pandas()
  """
  # Convert to numpy for Numba
  values = to_numpy(data)

  # Calculate SMA using Numba-optimized function
  sma_values = compute_sma_numba(values, period)

  return SMAResult(data_index=data.index, sma=sma_values)
