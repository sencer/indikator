"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import TYPPRICEResult

class _typprice_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> TYPPRICEResult: ...

class _typprice_ConfigDict(TypedDict, total=False):
  pass

class _typprice_Config(_NCMakeableModel[_typprice_Bound]):
  """Configuration class for typprice.

  Calculate Typical Price.

  TYPPRICE = (High + Low + Close) / 3

  Args:
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    TYPPRICEResult
  """

  pass

class typprice:
  Type = _typprice_Bound
  Config = _typprice_Config
  ConfigDict = _typprice_ConfigDict
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> TYPPRICEResult: ...
