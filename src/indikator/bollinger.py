"""Bollinger Bands indicator module.

This module provides Bollinger Bands calculation, a volatility indicator
consisting of a moving average with upper and lower bands.
"""

from typing import TYPE_CHECKING, cast

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Gt, Hyper, configurable
import numpy as np
import pandas as pd

if TYPE_CHECKING:
  from numpy.typing import NDArray

from indikator._bollinger_numba import (
  compute_bollinger_basic_numba,
  compute_bollinger_numba_fast,
)
from indikator._results import BollingerBandsResult, BollingerResult


@configurable
@validate
def bollinger_bands(
  data: Validated[pd.Series, Finite, NotEmpty],
  window: Hyper[int, Ge[2]] = 20,
  num_std: Hyper[float, Gt[0.0]] = 2.0,
) -> BollingerBandsResult:
  """Calculate Bollinger Bands.

  Bollinger Bands consist of a middle band (SMA) and two outer bands that
  are standard deviations away from the middle band. They expand and contract
  based on market volatility.

  This implementation uses population standard deviation (ddof=0) to match
  TA-lib behavior. For bandwidth and %B metrics, use `bollinger_with_bandwidth()`.

  Components:
  - Middle Band = SMA(close, window)
  - Upper Band = Middle Band + (std_dev * num_std)
  - Lower Band = Middle Band - (std_dev * num_std)

  Interpretation:
  - Price near upper band: Overbought
  - Price near lower band: Oversold
  - Bands squeezing: Low volatility, potential breakout coming
  - Bands expanding: High volatility, trend in motion

  Args:
    data: Input Series.
    window: Rolling window size (default: 20)
    num_std: Number of standard deviations for bands (default: 2.0)

  Returns:
    BollingerBandsResult(index, bb_upper, bb_middle, bb_lower)

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105])
    >>> result = bollinger_bands(prices, window=5, num_std=2.0)
    >>> result.bb_upper  # Access as array
    >>> df = result.to_pandas()  # Convert to DataFrame
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Use optimized Numba implementation (returns middle, upper, lower)
  middle, upper, lower = compute_bollinger_basic_numba(values, window, num_std)

  return BollingerBandsResult(
    index=data.index, bb_upper=upper, bb_middle=middle, bb_lower=lower
  )


@configurable
@validate
def bollinger_with_bandwidth(
  data: Validated[pd.Series, Finite, NotEmpty],
  window: Hyper[int, Ge[2]] = 20,
  num_std: Hyper[float, Gt[0.0]] = 2.0,
) -> BollingerResult:
  """Calculate Bollinger Bands with bandwidth and %B metrics.

  Extended Bollinger Bands calculation that includes bandwidth and %B
  in addition to the standard three bands. Uses sample standard deviation
  (ddof=1) which is statistically more correct for estimating population
  parameters from a sample.

  Components:
  - Middle Band = SMA(close, window)
  - Upper Band = Middle Band + (std_dev * num_std)
  - Lower Band = Middle Band - (std_dev * num_std)
  - Bandwidth = (Upper Band - Lower Band) / abs(Middle Band)
  - %B = (Price - Lower Band) / (Upper Band - Lower Band)

  Interpretation:
  - %B > 1: Price above upper band (very overbought)
  - %B < 0: Price below lower band (very oversold)
  - %B = 0.5: Price at middle band

  Note: Uses sample std (ddof=1), so bands will be slightly wider than
  the standard `bollinger_bands()` which uses population std (ddof=0).

  Args:
    data: Input Series.
    window: Rolling window size (default: 20)
    num_std: Number of standard deviations for bands (default: 2.0)

  Returns:
    BollingerResult with bb_middle, bb_upper, bb_lower, bb_bandwidth, bb_percent

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> result = bollinger_with_bandwidth(prices, window=5, num_std=2.0)
    >>> print(result.bb_bandwidth)  # Bandwidth values
    >>> print(result.bb_percent)    # %B values
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  # Use optimized Parallel Chunked implementation (fast)
  # DataWarden guarantees input validity, so we can skip NaN checks in the tight loop
  middle, upper, lower, bandwidth, percent_b = compute_bollinger_numba_fast(
    values, window, num_std
  )

  return BollingerResult(
    index=data.index,
    bb_middle=middle,
    bb_upper=upper,
    bb_lower=lower,
    bb_bandwidth=bandwidth,
    bb_percent=percent_b,
  )
