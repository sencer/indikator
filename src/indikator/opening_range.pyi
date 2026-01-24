"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Datetime, Finite, Index, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import OpeningRangeResult

class _opening_range_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period_minutes(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
    low: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
    close: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
  ) -> OpeningRangeResult: ...

class _opening_range_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for opening_range.

  Configuration:
      period_minutes (int)
  """

  period_minutes: int

class _opening_range_Config(_NCMakeableModel[_opening_range_Bound]):
  """Configuration class for opening_range.

  Calculate Opening Range Breakout (ORB) levels.

  Identifies the high and low of the first N minutes of the trading session.

  Outputs:
  - or_high: High of the opening range (held constant for the day)
  - or_low: Low of the opening range
  - or_mid: Midpoint of the range
  - or_range: Size of the range (High - Low)
  - or_breakout: Signal (1=Breakout Up, -1=Breakout Down, 0=None)

  Args:
    high: High prices with DatetimeIndex
    low: Low prices with DatetimeIndex
    close: Close prices with DatetimeIndex
    period_minutes: Opening range duration in minutes (default: 30)

  Returns:
    OpeningRangeResult(index, or_high, or_low, or_mid, or_range, or_breakout)

  Configuration:
      period_minutes (int)
  """

  period_minutes: int
  def __init__(self, *, period_minutes: int = ...) -> None: ...
  """Initialize configuration for opening_range.

    Configuration:
        period_minutes (int)
    """

  @override
  def make(self) -> _opening_range_Bound: ...

class opening_range:
  Type = _opening_range_Bound
  Config = _opening_range_Config
  ConfigDict = _opening_range_ConfigDict
  period_minutes: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
    low: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
    close: Validated[pd.Series, Finite, Index(Datetime), NotEmpty],
    period_minutes: int = ...,
  ) -> OpeningRangeResult: ...
