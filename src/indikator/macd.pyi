"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import MACDResult

class _macd_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def fast_period(self) -> int: ...
    @property
    def slow_period(self) -> int: ...
    @property
    def signal_period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> MACDResult: ...

class _macd_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for macd.

    Configuration:
        fast_period (int)
        slow_period (int)
        signal_period (int)
    """

    fast_period: int
    slow_period: int
    signal_period: int

class _macd_Config(_NCMakeableModel[_macd_Bound]):
    """Configuration class for macd.

    Calculate Moving Average Convergence Divergence (MACD).

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.

    Formula:
    MACD Line = EMA(fast_period) - EMA(slow_period)
    Signal Line = EMA(MACD Line, signal_period)
    Histogram = MACD Line - Signal Line

    Interpretation:
    - MACD crossing above Signal: Bullish
    - MACD crossing below Signal: Bearish
    - MACD > 0: Fast MA > Slow MA (Uptrend)
    - MACD < 0: Fast MA < Slow MA (Downtrend)
    - Histogram widening: Trend strengthening
    - Histogram narrowing: Trend weakening

    Args:
      data: Input Series.
      fast_period: Fast EMA period (default: 12)
      slow_period: Slow EMA period (default: 26)
      signal_period: Signal line EMA period (default: 9)

    Returns:
      DataFrame with 'macd', 'macd_signal', 'macd_histogram' columns

    Raises:
      ValueError: If fast_period >= slow_period

    Example:
      >>> import pandas as pd
      >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
      >>> result = macd(prices)
      >>> # Returns DataFrame with MACD components

    Configuration:
        fast_period (int)
        slow_period (int)
        signal_period (int)
    """

    fast_period: int
    slow_period: int
    signal_period: int
    def __init__(self, *, fast_period: int = ..., slow_period: int = ..., signal_period: int = ...) -> None: ...
    """Initialize configuration for macd.

    Configuration:
        fast_period (int)
        slow_period (int)
        signal_period (int)
    """

    @override
    def make(self) -> _macd_Bound: ...

class macd:
    Type = _macd_Bound
    Config = _macd_Config
    ConfigDict = _macd_ConfigDict
    fast_period: ClassVar[int]
    slow_period: ClassVar[int]
    signal_period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], fast_period: int = ..., slow_period: int = ..., signal_period: int = ...) -> MACDResult: ...

class _macdext_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def fast_period(self) -> int: ...
    @property
    def fast_matype(self) -> int: ...
    @property
    def slow_period(self) -> int: ...
    @property
    def slow_matype(self) -> int: ...
    @property
    def signal_period(self) -> int: ...
    @property
    def signal_matype(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> MACDResult: ...

class _macdext_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for macdext.

    Configuration:
        fast_period (int)
        fast_matype (int)
        slow_period (int)
        slow_matype (int)
        signal_period (int)
        signal_matype (int)
    """

    fast_period: int
    fast_matype: int
    slow_period: int
    slow_matype: int
    signal_period: int
    signal_matype: int

class _macdext_Config(_NCMakeableModel[_macdext_Bound]):
    """Configuration class for macdext.

    Calculate MACD with full control over MA types.

    Args:
      data: Input Series.
      fast_period: Fast period (default: 12)
      fast_matype: Fast MA type (default: 0 - SMA)
      slow_period: Slow period (default: 26)
      slow_matype: Slow MA type (default: 0 - SMA)
      signal_period: Signal period (default: 9)
      signal_matype: Signal MA type (default: 0 - SMA)

    Returns:
      MACDResult(index, macd, signal, histogram)

    Configuration:
        fast_period (int)
        fast_matype (int)
        slow_period (int)
        slow_matype (int)
        signal_period (int)
        signal_matype (int)
    """

    fast_period: int
    fast_matype: int
    slow_period: int
    slow_matype: int
    signal_period: int
    signal_matype: int
    def __init__(self, *, fast_period: int = ..., fast_matype: int = ..., slow_period: int = ..., slow_matype: int = ..., signal_period: int = ..., signal_matype: int = ...) -> None: ...
    """Initialize configuration for macdext.

    Configuration:
        fast_period (int)
        fast_matype (int)
        slow_period (int)
        slow_matype (int)
        signal_period (int)
        signal_matype (int)
    """

    @override
    def make(self) -> _macdext_Bound: ...

class macdext:
    Type = _macdext_Bound
    Config = _macdext_Config
    ConfigDict = _macdext_ConfigDict
    fast_period: ClassVar[int]
    fast_matype: ClassVar[int]
    slow_period: ClassVar[int]
    slow_matype: ClassVar[int]
    signal_period: ClassVar[int]
    signal_matype: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], fast_period: int = ..., fast_matype: int = ..., slow_period: int = ..., slow_matype: int = ..., signal_period: int = ..., signal_matype: int = ...) -> MACDResult: ...

class _macdfix_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def signal_period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> MACDResult: ...

class _macdfix_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for macdfix.

    Configuration:
        signal_period (int)
    """

    signal_period: int

class _macdfix_Config(_NCMakeableModel[_macdfix_Bound]):
    """Configuration class for macdfix.

    Calculate MACD with fixed periods (12, 26).

    Matches TA-Lib MACDFIX.

    Configuration:
        signal_period (int)
    """

    signal_period: int
    def __init__(self, *, signal_period: int = ...) -> None: ...
    """Initialize configuration for macdfix.

    Configuration:
        signal_period (int)
    """

    @override
    def make(self) -> _macdfix_Bound: ...

class macdfix:
    Type = _macdfix_Bound
    Config = _macdfix_Config
    ConfigDict = _macdfix_ConfigDict
    signal_period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], signal_period: int = ...) -> MACDResult: ...
