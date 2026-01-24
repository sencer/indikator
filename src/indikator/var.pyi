"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

if TYPE_CHECKING:
  from datawarden import Finite, NotEmpty, Validated

from indikator._results import VARResult

class _var_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  @property
  def nbdev(self) -> float: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> VARResult: ...

class _var_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for var.

  Configuration:
      period (int)
      nbdev (float)
  """

  period: int
  nbdev: float

class _var_Config(_NCMakeableModel[_var_Bound]):
  """Configuration class for var.

  Calculate Variance over period.

  Args:
    data: Input prices
    period: Lookback period (default 5)
    nbdev: Number of deviations (multiplier, default 1.0)

  Returns:
    VARResult

  Configuration:
      period (int)
      nbdev (float)
  """

  period: int
  nbdev: float
  def __init__(self, *, period: int = ..., nbdev: float = ...) -> None: ...
  """Initialize configuration for var.

    Configuration:
        period (int)
        nbdev (float)
    """

  @override
  def make(self) -> _var_Bound: ...

class var:
  Type = _var_Bound
  Config = _var_Config
  ConfigDict = _var_ConfigDict
  period: ClassVar[int]
  nbdev: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
    nbdev: float = ...,
  ) -> VARResult: ...
