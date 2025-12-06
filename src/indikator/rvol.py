"""Relative Volume (RVOL) indicator module.

This module calculates relative volume, which measures current trading volume
relative to the average volume over a lookback period.
"""

from hipr import Ge, Gt, Hyper, configurable
import pandas as pd
from pdval import (
  Datetime,
  Index,
  NonNegative,
  Validated,
  validated,
)

from indikator._intraday import intraday_aggregate

# Default minimum samples per time slot
_DEFAULT_MIN_SAMPLES = 3

__all__ = ["rvol", "rvol_intraday"]


@configurable
@validated
def rvol(
  data: Validated[pd.Series, NonNegative],
  window: Hyper[int, Ge[2]] = 20,
  epsilon: Hyper[float, Gt[0.0]] = 1e-9,
) -> pd.Series:
  """Calculate Relative Volume (RVOL).

  Measures current volume relative to average volume over a lookback window.
  RVOL is a key indicator for identifying unusual trading activity:
  - RVOL > 1.0: Above-average volume (more activity than usual)
  - RVOL = 1.0: Average volume (normal activity)
  - RVOL < 1.0: Below-average volume (less activity than usual)

  Values significantly above 1.0 (e.g., > 2.0 or > 3.0) indicate
  exceptional volume that may signal:
  - Breakouts or breakdowns
  - News events or earnings
  - Institutional accumulation/distribution
  - Potential trend changes

  Features:
  - Division by zero protection with epsilon
  - Handles insufficient data gracefully (returns 1.0 as neutral)
  - Simple moving average for stable baseline

  Args:
    data: Volume Series
    window: Rolling window size for average volume calculation
    epsilon: Small value to prevent division by zero

  Returns:
    Series with RVOL values

  Raises:
    ValueError: If validation fails
    pandera.errors.SchemaError: If validation fails

  Example:
    >>> import pandas as pd
    >>> data = pd.Series([1000, 1000, 1000, 2000, 3000])
    >>> result = rvol(data, window=3)
    >>> # RVOL will be ~1.0 for first bars, then spike to 2.0, 3.0
  """

  # Handle insufficient data for window
  if len(data) < window:
    return pd.Series(1.0, index=data.index, name="rvol")

  # Calculate simple moving average of volume
  sma_volume = data.rolling(window=window).mean()

  # Calculate relative volume with division by zero protection
  # Use where clause to handle zero/NaN denominator
  rvol_values = pd.Series(1.0, index=data.index, name="rvol")
  valid_sma = sma_volume > epsilon

  rvol_values[valid_sma] = data[valid_sma] / sma_volume[valid_sma]

  return rvol_values


@configurable
@validated
def rvol_intraday(
  data: Validated[pd.Series, NonNegative, Index[Datetime]],
  lookback_days: int | None = None,
  min_samples: Hyper[int, Ge[1]] = _DEFAULT_MIN_SAMPLES,
  epsilon: Hyper[float, Gt[0.0]] = 1e-9,
) -> pd.Series:
  """Calculate intraday RVOL based on time-of-day historical averages.

  Compares current volume to the historical average volume for that specific
  time of day (e.g., 10:30 AM today vs. all previous 10:30 AM bars).

  This captures intraday seasonality patterns:
  - Market open (9:30-10:00) typically has high volume
  - Lunch (12:00-13:00) typically has low volume
  - Market close (15:30-16:00) typically has high volume

  Regular RVOL might show "high" during market open even when it's normal
  for that time. Intraday RVOL correctly identifies "high for this time of day".

  Features:
  - Accounts for natural intraday volume patterns
  - Configurable lookback period (None = use all history)
  - Requires minimum samples per time slot for reliability
  - Division by zero protection

  Args:
    data: Series (e.g., volume) with DatetimeIndex
    lookback_days: Number of days to look back (None = use all history)
    min_samples: Minimum historical samples required per time slot
    epsilon: Small value to prevent division by zero

  Returns:
    Series with intraday RVOL values

  Raises:
    ValueError: If index is not DatetimeIndex
  """

  # Get historical averages for each time slot using generic aggregator
  avg_volume_by_time = intraday_aggregate(
    data,
    agg_func="mean",
    lookback_days=lookback_days,
    min_samples=min_samples,
  )

  # Calculate RVOL with division by zero protection
  rvol_values = pd.Series(1.0, index=data.index, name="rvol_intraday")
  valid_avg = avg_volume_by_time > epsilon

  rvol_values[valid_avg] = data[valid_avg].div(avg_volume_by_time[valid_avg])

  return rvol_values
