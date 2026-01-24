"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import MAMAResult

class _mama_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def fastlimit(self) -> float: ...
  @property
  def slowlimit(self) -> float: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> MAMAResult: ...

class _mama_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for mama.

  Configuration:
      fastlimit (float)
      slowlimit (float)
  """

  fastlimit: float
  slowlimit: float

class _mama_Config(_NCMakeableModel[_mama_Bound]):
  """Configuration class for mama.

  Calculate MESA Adaptive Moving Average (MAMA) and FAMA.

  MAMA and FAMA provide a smoothed, responsive trendline that avoids
  whipsaws in consolidation but adapts quickly to new trends.

  Args:
    data: Input price Series.
    fastlimit: MESA fast limit (default: 0.5).
    slowlimit: MESA slow limit (default: 0.05).

  Returns:
    MAMAResult(index, mama, fama)

  Configuration:
      fastlimit (float)
      slowlimit (float)
  """

  fastlimit: float
  slowlimit: float
  def __init__(self, *, fastlimit: float = ..., slowlimit: float = ...) -> None: ...
  """Initialize configuration for mama.

    Configuration:
        fastlimit (float)
        slowlimit (float)
    """

  @override
  def make(self) -> _mama_Bound: ...

class mama:
  Type = _mama_Bound
  Config = _mama_Config
  ConfigDict = _mama_ConfigDict
  fastlimit: ClassVar[float]
  slowlimit: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    fastlimit: float = ...,
    slowlimit: float = ...,
  ) -> MAMAResult: ...
