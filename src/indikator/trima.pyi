"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated

from indikator._results import TRIMAResult

class _trima_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> TRIMAResult: ...

class _trima_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for trima.

  Configuration:
      period (int)
  """

  period: int

class _trima_Config(_NCMakeableModel[_trima_Bound]):
  """Configuration class for trima.

  Calculate Triangular Moving Average (TRIMA).

  TRIMA is a smoothed version of SMA, calculated as SMA of SMA.
  The weights form a triangular shape.

  Args:
    data: Input price Series.
    period: Lookback period (default: 30).

  Returns:
    TRIMAResult

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for trima.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _trima_Bound: ...

class trima:
  Type = _trima_Bound
  Config = _trima_Config
  ConfigDict = _trima_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...
  ) -> TRIMAResult: ...
