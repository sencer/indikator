"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated

from indikator._results import T3Result

class _t3_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  @property
  def vfactor(self) -> float: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> T3Result: ...

class _t3_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for t3.

  Configuration:
      period (int)
      vfactor (float)
  """

  period: int
  vfactor: float

class _t3_Config(_NCMakeableModel[_t3_Bound]):
  """Configuration class for t3.

  Calculate T3 Moving Average.

  T3 is a smooth, low-lag moving average developed by Tim Tilson.
  It uses a "volume factor" (vfactor) to control responsiveness.
  Default vfactor is 0.7.

  Args:
    data: Input price Series.
    period: EMA period (default 5).
    vfactor: Volume factor (default 0.7).

  Returns:
    T3Result

  Configuration:
      period (int)
      vfactor (float)
  """

  period: int
  vfactor: float
  def __init__(self, *, period: int = ..., vfactor: float = ...) -> None: ...
  """Initialize configuration for t3.

    Configuration:
        period (int)
        vfactor (float)
    """

  @override
  def make(self) -> _t3_Bound: ...

class t3:
  Type = _t3_Bound
  Config = _t3_Config
  ConfigDict = _t3_ConfigDict
  period: ClassVar[int]
  vfactor: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
    vfactor: float = ...,
  ) -> T3Result: ...
