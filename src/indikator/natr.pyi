"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _natr_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty], close: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _natr_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for natr.

    Configuration:
        period (int)
    """

    period: int

class _natr_Config(_NCMakeableModel[_natr_Bound]):
    """Configuration class for natr.

    Calculate Normalized Average True Range (NATR).

    NATR normalizes ATR as a percentage of the closing price, allowing
    volatility comparison across different price levels and instruments.

    Formula:
    NATR = (ATR / Close) * 100

    Interpretation:
    - Higher NATR: More volatile relative to price
    - Lower NATR: Less volatile relative to price
    - Useful for position sizing across different instruments
    - Allows volatility comparison between $10 and $1000 stocks

    Features:
    - Fused Numba kernel: computes TR, ATR, and normalization in single loop
    - No intermediate arrays

    Args:
      high: High prices
      low: Low prices
      close: Close prices
      period: ATR lookback period (default: 14)

    Returns:
      IndicatorResult with normalized ATR values (percentage)

    Example:
      >>> result = natr(high, low, close, period=14)

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for natr.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _natr_Bound: ...

class natr:
    Type = _natr_Bound
    Config = _natr_Config
    ConfigDict = _natr_ConfigDict
    period: ClassVar[int]
    def __new__(cls, high: Validated[pd.Series[float], Finite, NotEmpty], low: Validated[pd.Series[float], Finite, NotEmpty], close: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...
