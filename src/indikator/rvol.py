"""Relative Volume (RVOL) indicator module.

This module calculates relative volume, which measures current trading volume
relative to the average volume over a lookback period.
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
from indikator._intraday import intraday_aggregate
from indikator._results import RVOLResult
from indikator.utils import to_numpy

__all__ = ["rvol", "rvol_intraday"]


@configurable
@validate
def rvol(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  window: Hyper[int, Ge[2]] = 14,
  epsilon: Hyper[float, Gt[0.0]] = DEFAULT_EPSILON,
) -> RVOLResult:
  """Calculate Relative Volume (RVOL) using simple moving average.

  Measures current volume relative to average volume over a rolling window.

  Formula:
  RVOL = Volume / SMA(Volume, window)

  Interpretation:
  - RVOL > 1.0: Above-average volume
  - RVOL > 2.0: High volume spike
  - RVOL < 1.0: Below-average volume

  Args:
    data: Volume Series
    window: Rolling window size (default: 14)
    epsilon: Small value to prevent division by zero

  Returns:
    RVOLResult(index, rvol)
  """
  # Calculate simple moving average of volume
  sma_volume = data.rolling(window=window).mean()

  # Calculate relative volume
  # Get numpy arrays for calculation
  vol_arr = to_numpy(data)
  sma_arr = to_numpy(sma_volume)

  rvol_values = np.ones_like(vol_arr)  # Default to 1.0

  # Valid mask where SMA > epsilon
  # (and handle NaNs in SMA due to window)
  valid_sma = (sma_arr > epsilon) & ~np.isnan(sma_arr) & ~np.isnan(vol_arr)

  rvol_values[valid_sma] = vol_arr[valid_sma] / sma_arr[valid_sma]

  return RVOLResult(data_index=data.index, rvol=rvol_values)


@configurable
@validate
def rvol_intraday(
  data: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
  lookback_days: Hyper[int] | None = None,
  min_samples: Hyper[int, Ge[1]] = DEFAULT_MIN_SAMPLES,
  epsilon: Hyper[float, Gt[0.0]] = DEFAULT_EPSILON,
) -> RVOLResult:
  """Calculate intraday RVOL based on time-of-day historical averages.

  Compares current volume to the historical average volume for that specific
  time of day (e.g., 10:30 AM today vs. all previous 10:30 AM bars).

  Formula:
  RVOL = Volume / AvgVolume(TimeOfDay)

  Features:
  - Accounts for natural intraday volume patterns
  - Configurable lookback period

  Args:
    data: Series (e.g., volume) with DatetimeIndex
    lookback_days: Number of days to look back
    min_samples: Minimum observations required
    epsilon: Small value to prevent division by zero asd

  Returns:
    RVOLResult(index, rvol)
  """

  # Get historical averages for each time slot using generic aggregator
  avg_volume_by_time = intraday_aggregate(
    data,
    agg_func="mean",
    lookback_days=lookback_days,
    min_samples=min_samples,
  )

  # Calculate RVOL with division by zero protection
  # Calculate RVOL with division by zero protection
  vol_arr = to_numpy(data)
  # avg_volume_by_time is IntradaySeriesResult
  avg_arr = avg_volume_by_time.values

  rvol_values = np.ones_like(vol_arr)

  valid_avg = (avg_arr > epsilon) & ~np.isnan(avg_arr) & ~np.isnan(vol_arr)
  rvol_values[valid_avg] = vol_arr[valid_avg] / avg_arr[valid_avg]

  return RVOLResult(data_index=data.index, rvol=rvol_values)
