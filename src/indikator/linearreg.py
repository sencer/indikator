"""Linear Regression indicator family module.

This module provides Numba-optimized implementations of:
- LINEARREG: Linear regression value at end of window
- LINEARREG_INTERCEPT: Y-intercept of regression line
- LINEARREG_ANGLE: Angle of regression line in degrees
- LINEARREG_SLOPE: Slope of regression line (same as slope)
- TSF: Time Series Forecast (1-bar-ahead projection)

All implementations use O(1) rolling updates for optimal performance.
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

from indikator._results import (
  LINEARREGAngleResult,
  LINEARREGInterceptResult,
  LINEARREGResult,
  LINEARREGSlopeResult,
  TSFResult,
)
from indikator._slope_numba import (
  compute_linearreg_angle_numba,
  compute_linearreg_intercept_numba,
  compute_linearreg_numba,
  compute_slope_numba,
  compute_tsf_numba,
)


@configurable
@validate
def linearreg(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> LINEARREGResult:
  """Calculate LINEARREG: linear regression value at end of window.

  Fits a linear regression line y = mx + b over the past `period` bars,
  then returns the value of the line at the last (most recent) bar.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    LINEARREGResult(index, linearreg)
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_linearreg_numba(values, period)

  return LINEARREGResult(index=data.index, linearreg=result)


@configurable
@validate
def linearreg_intercept(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> LINEARREGInterceptResult:
  """Calculate LINEARREG_INTERCEPT: y-intercept of regression line.

  Fits a linear regression line y = mx + b over the past `period` bars,
  then returns the y-intercept (b).

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    LINEARREGInterceptResult(index, linearreg_intercept)
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_linearreg_intercept_numba(values, period)

  return LINEARREGInterceptResult(index=data.index, linearreg_intercept=result)


@configurable
@validate
def linearreg_angle(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> LINEARREGAngleResult:
  """Calculate LINEARREG_ANGLE: angle of regression line in degrees.

  Fits a linear regression line over the past `period` bars,
  then returns the angle of the line in degrees (atan(slope) * 180/pi).

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    LINEARREGAngleResult(index, linearreg_angle)
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_linearreg_angle_numba(values, period)

  return LINEARREGAngleResult(index=data.index, linearreg_angle=result)


@configurable
@validate
def linearreg_slope(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> LINEARREGSlopeResult:
  """Calculate LINEARREG_SLOPE: slope of regression line.

  Fits a linear regression line over the past `period` bars,
  then returns the slope (m in y = mx + b).

  This is equivalent to the `slope` indicator.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    LINEARREGSlopeResult(index, linearreg_slope)
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_slope_numba(values, period)

  return LINEARREGSlopeResult(index=data.index, linearreg_slope=result)


@configurable
@validate
def tsf(
  data: Validated[pd.Series, Finite, NotEmpty],
  period: Hyper[int, Ge[2]] = 14,
) -> TSFResult:
  """Calculate TSF: Time Series Forecast.

  Fits a linear regression line over the past `period` bars,
  then projects the line 1 bar forward (TSF = intercept + slope * period).

  Useful for predicting the next value based on recent trend.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    TSFResult(index, tsf)
  """
  values = cast(
    "NDArray[np.float64]",
    data.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
  )

  result = compute_tsf_numba(values, period)

  return TSFResult(index=data.index, tsf=result)
