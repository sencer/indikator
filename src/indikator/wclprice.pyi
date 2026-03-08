"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _wclprice_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty], close: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

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
      IndicatorResult
    """

    pass

class wclprice:
    Type = _wclprice_Bound
    Config = _wclprice_Config
    ConfigDict = _wclprice_ConfigDict
    def __new__(cls, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty], close: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...
