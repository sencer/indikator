"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ADOSCResult

class _adosc_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def fast_period(self) -> int: ...
    @property
    def slow_period(self) -> int: ...
    def __call__(self, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty], volume: Validated[pd.Series, Finite, NotEmpty]) -> ADOSCResult: ...

class _adosc_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for adosc.

    Configuration:
        fast_period (int)
        slow_period (int)
    """

    fast_period: int
    slow_period: int

class _adosc_Config(_NCMakeableModel[_adosc_Bound]):
    """Configuration class for adosc.

    Calculate Accumulation/Distribution Oscillator (Chaikin Oscillator).

    ADOSC measures the momentum of the A/D line by taking the difference
    between fast and slow EMAs of the cumulative A/D values.

    Formula:
    ADOSC = EMA(AD, fast) - EMA(AD, slow)

    Interpretation:
    - Positive ADOSC: Bullish momentum (A/D accelerating up)
    - Negative ADOSC: Bearish momentum (A/D accelerating down)
    - Zero crossing: Momentum shift

    Features:
    - Fused computation: AD + dual EMA in single pass
    - O(N) complexity

    Args:
      high: High prices
      low: Low prices
      close: Close prices
      volume: Volume
      fast_period: Fast EMA period (default: 3)
      slow_period: Slow EMA period (default: 10)

    Returns:
      ADOSCResult with oscillator values

    Example:
      >>> result = adosc(high, low, close, volume, fast_period=3, slow_period=10)

    Configuration:
        fast_period (int)
        slow_period (int)
    """

    fast_period: int
    slow_period: int
    def __init__(self, *, fast_period: int = ..., slow_period: int = ...) -> None: ...
    """Initialize configuration for adosc.

    Configuration:
        fast_period (int)
        slow_period (int)
    """

    @override
    def make(self) -> _adosc_Bound: ...

class adosc:
    Type = _adosc_Bound
    Config = _adosc_Config
    ConfigDict = _adosc_ConfigDict
    fast_period: ClassVar[int]
    slow_period: ClassVar[int]
    def __new__(cls, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty], volume: Validated[pd.Series, Finite, NotEmpty], fast_period: int = ..., slow_period: int = ...) -> ADOSCResult: ...
