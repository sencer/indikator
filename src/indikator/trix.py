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

from indikator._results import TRIXResult
from indikator._trix_numba import compute_trix_numba


@configurable
@validate
def trix(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> TRIXResult:
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
    TRIXResult(index, trix)
  """
  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate TRIX using Numba-optimized function
  trix_values = compute_trix_numba(values, period)

  return TRIXResult(index=data.index, trix=trix_values)
