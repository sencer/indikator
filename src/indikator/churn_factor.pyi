"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _churn_factor_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def epsilon(self) -> float: ...
    def __call__(self, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty], volume: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _churn_factor_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for churn_factor.

    Configuration:
        epsilon (float)
    """

    epsilon: float

class _churn_factor_Config(_NCMakeableModel[_churn_factor_Bound]):
    """Configuration class for churn_factor.

    Calculate Churn Factor.

    Churn factor measures the efficiency of volume in moving the price.
    High churn means high volume but low price movement (indecision/turning point).

    Formula:
    Churn = Volume / (High - Low)

    Interpretation:
    - High Churn: High volume with tight range (distribution/accumulation)
    - Low Churn: Price moving freely on low volume (or low vol/low range)

    Args:
      high: High prices Series.
      low: Low prices Series.
      volume: Volume Series.
      epsilon: Division by zero protection.

    Returns:
      IndicatorResult(index, churn)

    Configuration:
        epsilon (float)
    """

    epsilon: float
    def __init__(self, *, epsilon: float = ...) -> None: ...
    """Initialize configuration for churn_factor.

    Configuration:
        epsilon (float)
    """

    @override
    def make(self) -> _churn_factor_Bound: ...

class churn_factor:
    Type = _churn_factor_Bound
    Config = _churn_factor_Config
    ConfigDict = _churn_factor_ConfigDict
    epsilon: ClassVar[float]
    def __new__(cls, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty], volume: Validated[pd.Series[float], Finite, NotEmpty], epsilon: float = ...) -> IndicatorResult: ...
