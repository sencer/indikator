"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel

MAType: ...

class _ma_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  @property
  def matype(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> Any: ...

class _ma_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for ma.

  Configuration:
      period (int)
      matype (int)
  """

  period: int
  matype: int

class _ma_Config(_NCMakeableModel[_ma_Bound]):
  """Configuration class for ma.

  Universal Moving Average wrapper.

  Args:
    data: Input Series.
    period: Lookback period (default: 30)
    matype: Moving Average type (0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3)

  Returns:
    IndicatorResult: Result object (SMAResult, EMAResult, etc.)

  Configuration:
      period (int)
      matype (int)
  """

  period: int
  matype: int
  def __init__(self, *, period: int = ..., matype: int = ...) -> None: ...
  """Initialize configuration for ma.

    Configuration:
        period (int)
        matype (int)
    """

  @override
  def make(self) -> _ma_Bound: ...

class ma:
  Type = _ma_Bound
  Config = _ma_Config
  ConfigDict = _ma_ConfigDict
  period: ClassVar[int]
  matype: ClassVar[int]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
    matype: int = ...,
  ) -> Any: ...
