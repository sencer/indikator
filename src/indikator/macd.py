"""MACD (Moving Average Convergence Divergence) indicator module.

This module provides MACD calculation, a trend-following momentum indicator
that shows the relationship between two moving averages.
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

from indikator._macd_numba import compute_macd_numba
from indikator._results import MACDResult


@configurable
@validate
def macd(
  data: Validated[pd.Series, Finite, NotEmpty],
  fast_period: Hyper[int, Ge[2]] = 12,
  slow_period: Hyper[int, Ge[2]] = 26,
  signal_period: Hyper[int, Ge[2]] = 9,
) -> MACDResult:
  """Calculate Moving Average Convergence Divergence (MACD).

  MACD is a trend-following momentum indicator that shows the relationship
  between two moving averages of a security's price.

  Formula:
  MACD Line = EMA(fast_period) - EMA(slow_period)
  Signal Line = EMA(MACD Line, signal_period)
  Histogram = MACD Line - Signal Line

  Interpretation:
  - MACD crossing above Signal: Bullish
  - MACD crossing below Signal: Bearish
  - MACD > 0: Fast MA > Slow MA (Uptrend)
  - MACD < 0: Fast MA < Slow MA (Downtrend)
  - Histogram widening: Trend strengthening
  - Histogram narrowing: Trend weakening

  Args:
    data: Input Series.
    fast_period: Fast EMA period (default: 12)
    slow_period: Slow EMA period (default: 26)
    signal_period: Signal line EMA period (default: 9)

  Returns:
    DataFrame with 'macd', 'macd_signal', 'macd_histogram' columns

  Raises:
    ValueError: If fast_period >= slow_period

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> result = macd(prices)
    >>> # Returns DataFrame with MACD components
  """
  # Validate parameters
  if fast_period >= slow_period:
    raise ValueError(
      f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
    )

  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate MACD using Numba-optimized function
  macd_line, signal_line, histogram = compute_macd_numba(
    values, fast_period, slow_period, signal_period
  )

  return MACDResult(
    index=data.index, macd=macd_line, signal=signal_line, histogram=histogram
  )
