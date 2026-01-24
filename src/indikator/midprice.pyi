"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import MIDPRICEResult

class _midprice_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
  ) -> MIDPRICEResult: ...

class _midprice_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for midprice.

  Configuration:
      period (int)
  """

  period: int

class _midprice_Config(_NCMakeableModel[_midprice_Bound]):
  """Configuration class for midprice.

  Calculate Midpoint Price over period.

  MIDPRICE = (highest high + lowest low) / 2

  Args:
    high: High prices
    low: Low prices
    period: Lookback period (default 14)

  Returns:
    MIDPRICEResult

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for midprice.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _midprice_Bound: ...

class midprice:
  Type = _midprice_Bound
  Config = _midprice_Config
  ConfigDict = _midprice_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> MIDPRICEResult: ...
