"""Williams %R indicator module.

This module provides Williams %R calculation, a momentum indicator that
measures overbought/oversold levels.
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

from indikator._momentum_numba import compute_willr_numba
from indikator._results import WillRResult


@configurable
@validate
def willr(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> WillRResult:
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
    WillRResult(index, willr)
  """
  # Convert to numpy for Numba
  high_arr = cast(
    "NDArray[np.float64]",
    high.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  low_arr = cast(
    "NDArray[np.float64]",
    low.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )
  close_arr = cast(
    "NDArray[np.float64]",
    close.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Calculate WillR using Numba-optimized function
  willr_values = compute_willr_numba(high_arr, low_arr, close_arr, period)

  return WillRResult(index=high.index, willr=willr_values)
