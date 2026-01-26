"""Linear Regression indicator family module.

This module provides Numba-optimized implementations of:
- LINEARREG: Linear regression value at end of window
- LINEARREG_INTERCEPT: Y-intercept of regression line
- LINEARREG_ANGLE: Angle of regression line in degrees
- LINEARREG_SLOPE: Slope of regression line (same as slope)
- TSF: Time Series Forecast (1-bar-ahead projection)

All implementations use O(1) rolling updates for optimal performance.
"""

from datawarden import (
  Finite,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import pandas as pd

from indikator._results import (
  IndicatorResult,
)
from indikator.numba.slope import (
  compute_linearreg_angle_numba,
  compute_linearreg_intercept_numba,
  compute_linearreg_numba,
  compute_slope_numba,
  compute_tsf_numba,
)
from indikator.utils import to_numpy


@configurable
@validate
def linearreg(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate LINEARREG: linear regression value at end of window.

  Fits a linear regression line y = mx + b over the past `period` bars,
  then returns the value of the line at the last (most recent) bar.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg)
  """
  values = to_numpy(data)

  result = compute_linearreg_numba(values, period)

  return IndicatorResult(data_index=data.index, value=result, name="linearreg")


@configurable
@validate
def linearreg_intercept(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate LINEARREG_INTERCEPT: y-intercept of regression line.

  Fits a linear regression line y = mx + b over the past `period` bars,
  then returns the y-intercept (b).

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg_intercept)
  """
  values = to_numpy(data)

  result = compute_linearreg_intercept_numba(values, period)

  return IndicatorResult(
    data_index=data.index, value=result, name="linearreg_intercept"
  )


@configurable
@validate
def linearreg_angle(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate LINEARREG_ANGLE: angle of regression line in degrees.

  Fits a linear regression line over the past `period` bars,
  then returns the angle of the line in degrees (atan(slope) * 180/pi).

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg_angle)
  """
  values = to_numpy(data)

  result = compute_linearreg_angle_numba(values, period)

  return IndicatorResult(data_index=data.index, value=result, name="linearreg_angle")


@configurable
@validate
def linearreg_slope(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate LINEARREG_SLOPE: slope of regression line.

  Fits a linear regression line over the past `period` bars,
  then returns the slope (m in y = mx + b).

  This is equivalent to the `slope` indicator.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg_slope)
  """
  values = to_numpy(data)

  result = compute_slope_numba(values, period)

  return IndicatorResult(data_index=data.index, value=result, name="linearreg_slope")


@configurable
@validate
def tsf(
  data: Validated[pd.Series[float], Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> IndicatorResult:
  """Calculate TSF: Time Series Forecast.

  Fits a linear regression line over the past `period` bars,
  then projects the line 1 bar forward (TSF = intercept + slope * period).

  Useful for predicting the next value based on recent trend.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, tsf)
  """
  values = to_numpy(data)

  result = compute_tsf_numba(values, period)

  return IndicatorResult(data_index=data.index, value=result, name="tsf")
