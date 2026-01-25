"""Stochastic Oscillator indicator module.

This module provides Stochastic calculation, a momentum indicator comparing
closing price to its price range over a given time period.
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

from indikator._results import StochResult
from indikator._stoch_numba import compute_stoch_numba


@configurable
@validate
def stoch(  # noqa: PLR0913, PLR0917
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  k_period: Hyper[int, Ge[2]] = 14,
  k_slowing: Hyper[int, Ge[1]] = 3,
  d_period: Hyper[int, Ge[1]] = 3,
) -> StochResult:
  """Calculate Stochastic Oscillator.

  The Stochastic Oscillator is a momentum indicator comparing closing price
  to its price range over a given time period.

  Formula:
  %K = 100 * SMA((Close - Lowest Low) / (Highest High - Lowest Low), k_slowing)
  %D = SMA(%K, d_period)

  Interpretation:
  - %K > 80: Overbought
  - %K < 20: Oversold
  - %K crossing %D: Signal (bullish if crossing up, bearish if crossing down)
  - Divergence: Price making new high but stochastic doesn't = bearish

  Common strategies:
  - Buy when %K crosses above %D below 20
  - Sell when %K crosses below %D above 80

  Features:
  - Numba-optimized for performance
  - Standard Fast Stochastic with slowing
  - Returns both %K and %D lines

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    k_period: Period for highest high / lowest low (default: 14)
    k_slowing: Slowing period for %K (default: 3)
    d_period: Period for %D smoothing (default: 3)

  Returns:
    DataFrame with 'stoch_k' and 'stoch_d' columns

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> high = pd.Series([105, 107, 106, 108, 110])
    >>> low = pd.Series([100, 102, 101, 103, 105])
    >>> close = pd.Series([102, 105, 104, 106, 108])
    >>> result = stoch(high, low, close)
  """
  # Convert to numpy for Numba
  # Use to_numpy with copy=False to avoid copying if already float64 and compatible
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

  # Calculate Stochastic using Numba-optimized function
  stoch_k, stoch_d = compute_stoch_numba(
    high_arr, low_arr, close_arr, k_period, k_slowing, d_period
  )

  return StochResult(
    data_index=high.index,
    stoch_k=stoch_k,
    stoch_d=stoch_d,
  )


@configurable
@validate
def stochf(
  high: Validated[pd.Series[float], Finite, NotEmpty],
  low: Validated[pd.Series[float], Finite, NotEmpty],
  close: Validated[pd.Series[float], Finite, NotEmpty],
  fastk_period: Hyper[int, Ge[2]] = 5,
  fastd_period: Hyper[int, Ge[1]] = 3,
) -> StochResult:
  """Calculate Fast Stochastic Oscillator (STOCHF).

  The Fast Stochastic Oscillator corresponds to the %K and %D lines where:
  %K is the raw stochastic (unsmoothed).
  %D is the SMA of %K.

  This is equivalent to calling `stoch` with `k_slowing=1`.

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    fastk_period: Period for raw %K calculation (default: 5)
    fastd_period: Period for %D smoothing (default: 3)

  Returns:
    StochResult(index, stoch_k, stoch_d)
    where stoch_k is Fast %K and stoch_d is Fast %D.
  """
  return stoch(
    high=high,
    low=low,
    close=close,
    k_period=fastk_period,
    k_slowing=1,  # Fast Stochastic has no slowing on %K
    d_period=fastd_period,
  )
