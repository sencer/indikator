"""Z-Score indicator module.

This module provides a generic Z-Score calculator that measures how many
standard deviations a value is away from its rolling mean.
"""

from datawarden import (
  Datetime,
  Finite,
  Index,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Gt, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._constants import DEFAULT_EPSILON, DEFAULT_MIN_SAMPLES
from indikator._results import IndicatorResult
from indikator.intraday import intraday_stats
from indikator.numba.zscore import compute_zscore_numba
from indikator.utils import to_numpy


@configurable
@validate
def zscore(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 20,
) -> IndicatorResult:
  """Calculate Z-Score (Standard Score).

  Z-Score measures how many standard deviations a price is from the mean.
  It is a mean-reversion indicator.

  Formula:
  Z = (Price - One_Year_Mean) / Standard_Deviation

  Interpretation:
  - Z > 2: Price is 2 standard deviations above mean (statistically rare/expensive)
  - Z < -2: Price is 2 standard deviations below mean (cheap)
  - Extreme values indicate potential reversal (mean reversion)
  - Z = 0: Price is exactly at the mean

  Features:
  - Numba-optimized for performance
  - Rolling window calculation
  - Standard 20-period default (similar to Bollinger Bands)

  Args:
    data: Input Series.
    period: Lookback period (default: 20)

  Returns:
    IndicatorResult(index, zscore)
  """
  # Convert to numpy for Numba
  values = to_numpy(data)

  # Calculate Z-Score using Numba-optimized function
  zscore_values = compute_zscore_numba(values, period)

  return IndicatorResult(data_index=data.index, value=zscore_values, name="zscore")


@configurable
@validate
def zscore_intraday(
  data: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
  lookback_days: Hyper[int] | None = None,
  min_samples: Hyper[int, Ge[2]] = DEFAULT_MIN_SAMPLES,
  epsilon: Hyper[float, Gt[0.0]] = DEFAULT_EPSILON,
) -> IndicatorResult:
  """Calculate time-of-day adjusted Z-Score.

    Compares current value to the historical mean and std dev for that specific
    time of day (e.g., 10:30 AM today vs. all previous 10:30 AM bars).

    This accounts for intraday patterns:
    - Price tends to be volatile at market open
    - Volume tends to be high at open/close, low at lunch
    - Spread/volatility patterns vary by time of day

    Regular Z-score might show "high volatility" during market open even when
    it's normal for that time. Intraday Z-score correctly identifies "high for
    this time of day".

    Features:
    - Accounts for natural intraday patterns
    - Works with any column (price, volume, spread, etc.)
    - Configurable lookback period
    - Requires minimum samples per time slot for reliability

    Args:
      data: Series with DatetimeIndex
      lookback_days: Number of days to look back (None = use all history)
      min_samples: Minimum observations required before calculating aggregate (NaN until met)
      epsilon: Small value to prevent division by zero

    Returns:
      Series with intraday Z-Score values

    Raises:
      ValueError: If index is not DatetimeIndex

    Example:
      >>> import pandas as pd
  from indikator.utils import to_numpy
      >>> dates = pd.date_range('2024-01-01 09:30', periods=10, freq='1D')
      >>> data = pd.Series([100]*9 + [150], index=dates)
      >>> # Same time each day, last day has spike
      >>> result = zscore_intraday(data)
      >>> # Will show high z-score on last day
  """

  # Get historical mean and std for each time slot in a single pass
  # Get historical mean and std for each time slot in a single pass
  stats = intraday_stats(
    data,
    lookback_days=lookback_days,
    min_samples=min_samples,
  )
  mean_by_time = stats.mean
  std_by_time = stats.std

  # Calculate Z-Score with division by zero protection
  z_score_values = np.zeros(len(data))
  valid_std = std_by_time > epsilon

  z_score_values[valid_std] = (data[valid_std] - mean_by_time[valid_std]) / std_by_time[
    valid_std
  ]

  # Create result series
  # Create result series
  return IndicatorResult(
    data_index=data.index, value=z_score_values, name="zscore_intraday"
  )
