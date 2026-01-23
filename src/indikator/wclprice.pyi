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
from indikator._results import WCLPRICEResult

class _wclprice_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> WCLPRICEResult: ...

class _wclprice_ConfigDict(TypedDict, total=False):
  pass

class _wclprice_Config(_NCMakeableModel[_wclprice_Bound]):
  """Configuration class for wclprice.

  Calculate Weighted Close Price.

  WCLPRICE = (High + Low + 2*Close) / 4

  Args:
    high: High prices
    low: Low prices
    close: Close prices

  Returns:
    WCLPRICEResult
  """

  pass

class wclprice:
  Type = _wclprice_Bound
  Config = _wclprice_Config
  ConfigDict = _wclprice_ConfigDict
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> WCLPRICEResult: ...
