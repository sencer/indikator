"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _apo_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def fast_period(self) -> int: ...
  @property
  def slow_period(self) -> int: ...
  @property
  def matype(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _apo_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for apo.

  Configuration:
      fast_period (int)
      slow_period (int)
      matype (int)
  """

  fast_period: int
  slow_period: int
  matype: int

class _apo_Config(_NCMakeableModel[_apo_Bound]):
  """Configuration class for apo.

  Calculate Absolute Price Oscillator (APO).

  APO = FastMA - SlowMA

  Args:
    data: Input Series.
    fast_period: Fast MA period (default 12).
    slow_period: Slow MA period (default 26).
    matype: Moving Average type (0=SMA, 1=EMA). Default 0.

  Returns:
    IndicatorResult

  Configuration:
      fast_period (int)
      slow_period (int)
      matype (int)
  """

  fast_period: int
  slow_period: int
  matype: int
  def __init__(
    self, *, fast_period: int = ..., slow_period: int = ..., matype: int = ...
  ) -> None: ...
  """Initialize configuration for apo.

    Configuration:
        fast_period (int)
        slow_period (int)
        matype (int)
    """

  @override
  def make(self) -> _apo_Bound: ...

class apo:
  Type = _apo_Bound
  Config = _apo_Config
  ConfigDict = _apo_ConfigDict
  fast_period: ClassVar[int]
  slow_period: ClassVar[int]
  matype: ClassVar[int]
  def __new__(
    cls,
    data: Validated[pd.Series[float], Finite, NotEmpty],
    fast_period: int = ...,
    slow_period: int = ...,
    matype: int = ...,
  ) -> IndicatorResult: ...
