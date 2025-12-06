"""MACD (Moving Average Convergence Divergence) indicator module.

This module provides MACD calculation, a trend-following momentum indicator
that shows the relationship between two moving averages.
"""

from typing import TYPE_CHECKING, cast

from hipr import Ge, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
  Finite,
  Validated,
  validated,
)

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._macd_numba import compute_macd_numba


@configurable
@validated
def macd(
  data: Validated[pd.Series, Finite],
  fast_period: Hyper[int, Ge[2]] = 12,
  slow_period: Hyper[int, Ge[2]] = 26,
  signal_period: Hyper[int, Ge[1]] = 9,
) -> pd.DataFrame:
  """Calculate MACD (Moving Average Convergence Divergence).

  MACD is a trend-following momentum indicator that shows the relationship
  between two exponential moving averages (EMAs) of a price series.

  Components:
  1. MACD Line = EMA(fast_period) - EMA(slow_period)
  2. Signal Line = EMA(MACD Line, signal_period)
  3. Histogram = MACD Line - Signal Line

  Interpretation:
  - MACD > 0: Price is above the slow EMA (bullish)
  - MACD < 0: Price is below the slow EMA (bearish)
  - MACD crossing above signal: Bullish signal (buy)
  - MACD crossing below signal: Bearish signal (sell)
  - Histogram growing: Trend strengthening
  - Histogram shrinking: Trend weakening
  - Divergence: Price makes new high but MACD doesn't = bearish

  Common strategies:
  - Signal crossovers: Buy on MACD cross above signal, sell on cross below
  - Zero crossovers: Buy on MACD cross above 0, sell on cross below 0
  - Divergence: Look for price/MACD divergences for reversal signals
  - Histogram: Use histogram for early trend change detection

  Features:
  - Numba-optimized for performance
  - Standard parameters (12, 26, 9) by default
  - Returns all three components
  - Works with any numeric column

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
    data.values.astype(np.float64),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate MACD using Numba-optimized function
  macd_line, signal_line, histogram = compute_macd_numba(
    values, fast_period, slow_period, signal_period
  )

  # Create result dataframe
  return pd.DataFrame(
    {"macd": macd_line, "macd_signal": signal_line, "macd_histogram": histogram},
    index=data.index,
  )
