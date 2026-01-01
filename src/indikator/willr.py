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

from indikator._willr_numba import compute_willr_numba


@configurable
@validate
def willr(
  high: Validated[pd.Series, Finite, NotEmpty],
  low: Validated[pd.Series, Finite, NotEmpty],
  close: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> pd.Series:
  """Calculate Williams %R.

  Williams %R is a momentum indicator that measures overbought/oversold levels.
  It's similar to the Stochastic Oscillator but inverted and on a negative scale.

  Formula:
  %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)

  Range: -100 to 0

  Interpretation:
  - %R between -20 and 0: Overbought
  - %R between -100 and -80: Oversold
  - %R crossing -50: Momentum shift

  Common strategies:
  - Buy when %R moves from below -80 to above -80
  - Sell when %R moves from above -20 to below -20
  - Divergence: Price making new high but %R doesn't = bearish

  Features:
  - Numba-optimized for performance
  - Standard 14-period default
  - Works with OHLC data

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with Williams %R values (-100 to 0 range)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> high = pd.Series([105, 107, 106, 108, 110])
    >>> low = pd.Series([100, 102, 101, 103, 105])
    >>> close = pd.Series([102, 105, 104, 106, 108])
    >>> result = willr(high, low, close)
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

  # Calculate Williams %R using Numba-optimized function
  willr_values = compute_willr_numba(high_arr, low_arr, close_arr, period)

  return pd.Series(willr_values, index=close.index, name="willr")
