"""EMA (Exponential Moving Average) indicator module.

This module provides EMA calculation, a trend-following indicator that
gives more weight to recent prices.
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

from indikator._ema_numba import compute_ema_numba
from indikator._results import EMAResult


@configurable
@validate
def ema(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> EMAResult:
  """Calculate Exponential Moving Average (EMA).

  EMA is a trend-following indicator that gives more weight to recent prices.
  It reacts faster to price changes than a Simple Moving Average.

  Formula:
  EMA = Price(t) * k + EMA(t-1) * (1-k)
  where k = 2 / (period + 1)

  Interpretation:
  - Price above EMA: Bullish
  - Price below EMA: Bearish
  - EMA crossovers: Trend change signals

  Common uses:
  - Trend identification
  - Support/resistance levels
  - MACD calculation (uses 12 and 26 period EMAs)
  - Signal line smoothing

  Features:
  - Numba-optimized for performance
  - First EMA value is SMA of first 'period' values (standard initialization)
  - Works with any numeric column
  - Returns named tuple with .to_pandas() conversion

  Args:
    data: Input Series.
    period: Lookback period (default: 20)

  Returns:
    EMAResult(index, ema_array)
  """
  # Convert to numpy for Numba
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate EMA using Numba-optimized function
  ema_values = compute_ema_numba(values, period)

  return EMAResult(data_index=data.index, ema=ema_values)
