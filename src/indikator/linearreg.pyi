"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _linearreg_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _linearreg_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for linearreg.

  Configuration:
      period (int)
  """

  period: int

class _linearreg_Config(_NCMakeableModel[_linearreg_Bound]):
  """Configuration class for linearreg.

  Calculate LINEARREG: linear regression value at end of window.

  Fits a linear regression line y = mx + b over the past `period` bars,
  then returns the value of the line at the last (most recent) bar.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for linearreg.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _linearreg_Bound: ...

class linearreg:
  Type = _linearreg_Bound
  Config = _linearreg_Config
  ConfigDict = _linearreg_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...

class _linearreg_intercept_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _linearreg_intercept_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for linearreg_intercept.

  Configuration:
      period (int)
  """

  period: int

class _linearreg_intercept_Config(_NCMakeableModel[_linearreg_intercept_Bound]):
  """Configuration class for linearreg_intercept.

  Calculate LINEARREG_INTERCEPT: y-intercept of regression line.

  Fits a linear regression line y = mx + b over the past `period` bars,
  then returns the y-intercept (b).

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg_intercept)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for linearreg_intercept.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _linearreg_intercept_Bound: ...

class linearreg_intercept:
  Type = _linearreg_intercept_Bound
  Config = _linearreg_intercept_Config
  ConfigDict = _linearreg_intercept_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...

class _linearreg_angle_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _linearreg_angle_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for linearreg_angle.

  Configuration:
      period (int)
  """

  period: int

class _linearreg_angle_Config(_NCMakeableModel[_linearreg_angle_Bound]):
  """Configuration class for linearreg_angle.

  Calculate LINEARREG_ANGLE: angle of regression line in degrees.

  Fits a linear regression line over the past `period` bars,
  then returns the angle of the line in degrees (atan(slope) * 180/pi).

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg_angle)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for linearreg_angle.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _linearreg_angle_Bound: ...

class linearreg_angle:
  Type = _linearreg_angle_Bound
  Config = _linearreg_angle_Config
  ConfigDict = _linearreg_angle_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...

class _linearreg_slope_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _linearreg_slope_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for linearreg_slope.

  Configuration:
      period (int)
  """

  period: int

class _linearreg_slope_Config(_NCMakeableModel[_linearreg_slope_Bound]):
  """Configuration class for linearreg_slope.

  Calculate LINEARREG_SLOPE: slope of regression line.

  Fits a linear regression line over the past `period` bars,
  then returns the slope (m in y = mx + b).

  This is equivalent to the `slope` indicator.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, linearreg_slope)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for linearreg_slope.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _linearreg_slope_Bound: ...

class linearreg_slope:
  Type = _linearreg_slope_Bound
  Config = _linearreg_slope_Config
  ConfigDict = _linearreg_slope_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...

class _tsf_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _tsf_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for tsf.

  Configuration:
      period (int)
  """

  period: int

class _tsf_Config(_NCMakeableModel[_tsf_Bound]):
  """Configuration class for tsf.

  Calculate TSF: Time Series Forecast.

  Fits a linear regression line over the past `period` bars,
  then projects the line 1 bar forward (TSF = intercept + slope * period).

  Useful for predicting the next value based on recent trend.

  Args:
    data: Series of prices
    period: Rolling window size (default: 14)

  Returns:
    IndicatorResult(index, tsf)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for tsf.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _tsf_Bound: ...

class tsf:
  Type = _tsf_Bound
  Config = _tsf_Config
  ConfigDict = _tsf_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...
