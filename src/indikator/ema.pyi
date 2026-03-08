"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _ema_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ema_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for ema.

    Configuration:
        period (int)
    """

    period: int

class _ema_Config(_NCMakeableModel[_ema_Bound]):
    """Configuration class for ema.

    Calculate Exponential Moving Average (EMA).

    EMA is a trend-following indicator that gives more weight to recent prices.
    It reacts faster to price changes than a Simple Moving Average.

    Formula:
    EMA = Price(t) * k + EMA(t-1) * (1-k)
    where k = 2 / (period + 1)

    Interpretation:
    - Price above EMA: Bullish
    - Price below EMA: Bearish
    - EMA crossovers: Trend change signals

    Common uses:
    - Trend identification
    - Support/resistance levels
    - MACD calculation (uses 12 and 26 period EMAs)
    - Signal line smoothing

    Features:
    - Numba-optimized for performance
    - First EMA value is SMA of first 'period' values (standard initialization)
    - Works with any numeric column
    - Returns named tuple with .to_pandas() conversion

    Args:
      data: Input Series.
      period: Lookback period (default: 20)

    Returns:
      IndicatorResult(index, ema_array)

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for ema.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _ema_Bound: ...

class ema:
    Type = _ema_Bound
    Config = _ema_Config
    ConfigDict = _ema_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...
