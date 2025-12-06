"""Z-Score indicator module.

This module provides a generic Z-Score calculator that measures how many
standard deviations a value is away from its rolling mean.
"""

from hipr import Ge, Gt, Hyper, configurable
import numpy as np
import pandas as pd
from pdval import (
    Datetime,
    Finite,
    Index,
    NonEmpty,
    NonNaN,
    Validated,
    validated,
)

from indikator.rvol import MIN_SAMPLES_PER_SLOT, intraday_aggregate


@configurable
@validated
def zscore(
    data: Validated[pd.Series, Finite, NonNaN, NonEmpty],
    window: Hyper[int, Ge[2]] = 20,
    epsilon: Hyper[float, Gt[0.0]] = 1e-9,
) -> pd.Series:
    """Calculate Z-Score (Standard Score) over a rolling window.

    Z-Score measures how many standard deviations a data point is from the mean.
    - Z > 2.0: Significantly above average (potential overbought/outlier)
    - Z < -2.0: Significantly below average (potential oversold/outlier)
    - Z ~ 0: Near average

    Args:
      data: Series of values (e.g., close prices, volume, etc.)
      window: Rolling window size for mean and std dev calculation
      epsilon: Small value to prevent division by zero.

    Returns:
      Series with Z-Score values

    Raises:
      ValueError: If validation fails
    """

    # Calculate rolling mean and std
    rolling = data.rolling(window=window)
    mean = rolling.mean()
    std = rolling.std()

    # Calculate Z-Score with division by zero protection
    # Use where clause to handle zero/NaN denominator
    zscore_values = pd.Series(0.0, index=data.index, name=data.name)
    valid_std = std > epsilon

    zscore_values[valid_std] = (data[valid_std] - mean[valid_std]) / std[valid_std]

    return zscore_values


@configurable
@validated
def zscore_intraday(
    data: Validated[pd.Series, Index[Datetime], NonEmpty],
    lookback_days: int | None = None,
    min_samples: Hyper[int, Ge[2]] = MIN_SAMPLES_PER_SLOT,
    epsilon: Hyper[float, Gt[0.0]] = 1e-9,
) -> pd.Series:
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
      min_samples: Minimum historical samples required per time slot
      epsilon: Small value to prevent division by zero

    Returns:
      Series with intraday Z-Score values

    Raises:
      ValueError: If index is not DatetimeIndex

    Example:
      >>> import pandas as pd
      >>> dates = pd.date_range('2024-01-01 09:30', periods=10, freq='1D')
      >>> data = pd.Series([100]*9 + [150], index=dates)
      >>> # Same time each day, last day has spike
      >>> result = zscore_intraday(data)
      >>> # Will show high z-score on last day
    """

    # Get historical mean and std for each time slot using generic aggregator
    mean_by_time: pd.Series = intraday_aggregate(
        data,
        agg_func=pd.Series.mean,
        lookback_days=lookback_days,
        min_samples=min_samples,
    )

    std_by_time: pd.Series = intraday_aggregate(
        data,
        agg_func=pd.Series.std,
        lookback_days=lookback_days,
        min_samples=min_samples,
    )

    # Calculate Z-Score with division by zero protection
    z_score_values = np.zeros(len(data))
    valid_std = std_by_time > epsilon

    z_score_values[valid_std] = (
        data[valid_std] - mean_by_time[valid_std]
    ) / std_by_time[valid_std]

    # Create result series
    return pd.Series(
        z_score_values,
        index=data.index,
        name=f"{data.name}_zscore_intraday" if data.name else "zscore_intraday",
    )
