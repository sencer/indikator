"""TRIX (Triple Exponential Average) indicator module.

This module provides TRIX calculation, a momentum oscillator that shows
the percent rate of change of a triple exponentially smoothed moving
average of prices.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import IndicatorResult
from indikator.numba.trix import compute_trix_numba
from indikator.utils import to_numpy


@configurable
@validate
def trix(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
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
  - TRIX > 0: Momentum is positive (uptrend)
  - TRIX < 0: Momentum is negative (downtrend)
  - Signal line crossover can be used for entries/exits
  - Divergences indicate potential reversals

  Features:
  - Numba-optimized for performance
  - Filters out insignificant price movements (due to triple smoothing)
  - Standard 30 period default (often 15 or 30)

  Args:
    data: Input Series.
    period: Lookback period (default: 30)

  Returns:
    IndicatorResult(index, trix)
  """
  # Convert to numpy for Numba
  values = to_numpy(data)

  # Calculate TRIX using Numba-optimized function
  trix_values = compute_trix_numba(values, period)

  return IndicatorResult(data_index=data.index, value=trix_values, name="trix")
