"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import STDDEVResult

class _stddev_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  @property
  def nbdev(self) -> float: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> STDDEVResult: ...

class _stddev_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for stddev.

  Configuration:
      period (int)
      nbdev (float)
  """

  period: int
  nbdev: float

class _stddev_Config(_NCMakeableModel[_stddev_Bound]):
  """Configuration class for stddev.

  Calculate Standard Deviation over period.

  Args:
    data: Input prices
    period: Lookback period (default 5)
    nbdev: Number of deviations (multiplier, default 1.0)

  Returns:
    STDDEVResult

  Configuration:
      period (int)
      nbdev (float)
  """

  period: int
  nbdev: float
  def __init__(self, *, period: int = ..., nbdev: float = ...) -> None: ...
  """Initialize configuration for stddev.

    Configuration:
        period (int)
        nbdev (float)
    """

  @override
  def make(self) -> _stddev_Bound: ...

class stddev:
  Type = _stddev_Bound
  Config = _stddev_Config
  ConfigDict = _stddev_ConfigDict
  period: ClassVar[int]
  nbdev: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
    nbdev: float = ...,
  ) -> STDDEVResult: ...
