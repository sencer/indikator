"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _correl_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, x: Validated[pd.Series[float], Finite, NotEmpty], y: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _correl_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for correl.

    Configuration:
        period (int)
    """

    period: int

class _correl_Config(_NCMakeableModel[_correl_Bound]):
    """Configuration class for correl.

    Calculate rolling Pearson correlation coefficient.

    CORREL measures the linear relationship between two variables.
    Range: -1 to +1

    Interpretation:
    - CORREL = +1: Perfect positive correlation
    - CORREL = 0: No linear correlation
    - CORREL = -1: Perfect negative correlation

    Uses O(1) rolling update for efficiency.

    Args:
      x: First variable
      y: Second variable
      period: Rolling window size (default: 30)

    Returns:
      IndicatorResult(index, correl)

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for correl.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _correl_Bound: ...

class correl:
    Type = _correl_Bound
    Config = _correl_Config
    ConfigDict = _correl_ConfigDict
    period: ClassVar[int]
    def __new__(cls, x: Validated[pd.Series[float], Finite, NotEmpty], y: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...
