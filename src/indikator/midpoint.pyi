"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _midpoint_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _midpoint_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for midpoint.

  Configuration:
      period (int)
  """

  period: int

class _midpoint_Config(_NCMakeableModel[_midpoint_Bound]):
  """Configuration class for midpoint.

  Calculate Midpoint over period.

  MIDPOINT = (highest value + lowest value) / 2

  Args:
    data: Input prices
    period: Lookback period (default 14)

  Returns:
    IndicatorResult

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for midpoint.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _midpoint_Bound: ...

class midpoint:
  Type = _midpoint_Bound
  Config = _midpoint_Config
  ConfigDict = _midpoint_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...
