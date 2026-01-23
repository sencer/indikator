"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from __future__ import annotations

from typing import Protocol
from typing import TypedDict
from typing import override

from nonfig import MakeableModel as _NCMakeableModel

from datawarden import Finite, NotEmpty, Validated
import pandas as pd
from indikator._results import AVGPRICEResult

class _avgprice_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    open: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> AVGPRICEResult: ...

class _avgprice_ConfigDict(TypedDict, total=False):
  pass

class _avgprice_Config(_NCMakeableModel[_avgprice_Bound]):
  """Configuration class for avgprice.

  Calculate Average Price.

  AVGPRICE = (Open + High + Low + Close) / 4

  Args:
    open: Open prices
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    AVGPRICEResult
  """

  pass

class avgprice:
  Type = _avgprice_Bound
  Config = _avgprice_Config
  ConfigDict = _avgprice_ConfigDict
  def __new__(
    cls,
    open: Validated[pd.Series, Finite, NotEmpty],
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> AVGPRICEResult: ...
