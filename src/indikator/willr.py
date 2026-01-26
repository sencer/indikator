"""Williams %R indicator module.

This module provides Williams %R calculation, a momentum indicator that
measures overbought/oversold levels.
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
from indikator.numba.momentum import compute_willr_numba
from indikator.utils import to_numpy


@configurable
@validate
def willr(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate Williams %R.

  Williams %R is a momentum indicator that measures overbought and oversold levels.
  It moves between 0 and -100.

  Formula:
  %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

  Interpretation:
  - %R > -20: Overbought
  - %R < -80: Oversold
  - Similar to Stochastic Oscillator Fast %K, but scaled -100 to 0

  Features:
  - Numba-optimized for performance
  - Standard 14 period default

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    IndicatorResult(index, willr)
  """
  # Convert to numpy for Numba
  high_arr = to_numpy(high)
  low_arr = to_numpy(low)
  close_arr = to_numpy(close)

  # Calculate WillR using Numba-optimized function
  willr_values = compute_willr_numba(high_arr, low_arr, close_arr, period)

  return IndicatorResult(data_index=high.index, value=willr_values, name="willr")
