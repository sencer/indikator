"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Protocol, TypedDict

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _medprice_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    def __call__(self, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _medprice_ConfigDict(TypedDict, total=False):
    pass

class _medprice_Config(_NCMakeableModel[_medprice_Bound]):
    """Configuration class for medprice.

    Calculate Median Price.

    MEDPRICE = (High + Low) / 2

    Args:
      high: High prices
      low: Low prices

    Returns:
      IndicatorResult
    """

    pass

class medprice:
    Type = _medprice_Bound
    Config = _medprice_Config
    ConfigDict = _medprice_ConfigDict
    def __new__(cls, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...
