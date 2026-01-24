"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import SARResult

class _sar_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def acceleration(self) -> float: ...
    @property
    def maximum(self) -> float: ...
    def __call__(self, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty]) -> SARResult: ...

class _sar_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for sar.

    Configuration:
        acceleration (float)
        maximum (float)
    """

    acceleration: float
    maximum: float

class _sar_Config(_NCMakeableModel[_sar_Bound]):
    """Configuration class for sar.

    Calculate Parabolic SAR (Stop and Reverse).

    Parabolic SAR provides potential entry and exit points by trailing
    price with an accelerating stop level.

    Formula:
    SAR(t+1) = SAR(t) + AF * (EP - SAR(t))

    Where:
    - AF: Acceleration Factor (starts at 0.02, increments by 0.02 on new EP)
    - EP: Extreme Point (highest high in uptrend, lowest low in downtrend)

    Interpretation:
    - Price above SAR: Uptrend (SAR is support)
    - Price below SAR: Downtrend (SAR is resistance)
    - SAR flip: Potential trend reversal

    Features:
    - State machine with register optimization
    - Handles trend reversals automatically

    Args:
      high: High prices
      low: Low prices
      acceleration: AF start and increment (default: 0.02)
      maximum: Maximum AF (default: 0.2)

    Returns:
      SARResult with SAR values

    Example:
      >>> result = sar(high, low, acceleration=0.02, maximum=0.2)

    Configuration:
        acceleration (float)
        maximum (float)
    """

    acceleration: float
    maximum: float
    def __init__(self, *, acceleration: float = ..., maximum: float = ...) -> None: ...
    """Initialize configuration for sar.

    Configuration:
        acceleration (float)
        maximum (float)
    """

    @override
    def make(self) -> _sar_Bound: ...

class sar:
    Type = _sar_Bound
    Config = _sar_Config
    ConfigDict = _sar_ConfigDict
    acceleration: ClassVar[float]
    maximum: ClassVar[float]
    def __new__(cls, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], acceleration: float = ..., maximum: float = ...) -> SARResult: ...

class _sarext_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def start_value(self) -> float: ...
    @property
    def offset_on_reversal(self) -> float: ...
    @property
    def acceleration_init_long(self) -> float: ...
    @property
    def acceleration_long(self) -> float: ...
    @property
    def acceleration_max_long(self) -> float: ...
    @property
    def acceleration_init_short(self) -> float: ...
    @property
    def acceleration_short(self) -> float: ...
    @property
    def acceleration_max_short(self) -> float: ...
    def __call__(self, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty]) -> SARResult: ...

class _sarext_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for sarext.

    Configuration:
        start_value (float)
        offset_on_reversal (float)
        acceleration_init_long (float)
        acceleration_long (float)
        acceleration_max_long (float)
        acceleration_init_short (float)
        acceleration_short (float)
        acceleration_max_short (float)
    """

    start_value: float
    offset_on_reversal: float
    acceleration_init_long: float
    acceleration_long: float
    acceleration_max_long: float
    acceleration_init_short: float
    acceleration_short: float
    acceleration_max_short: float

class _sarext_Config(_NCMakeableModel[_sarext_Bound]):
    """Configuration class for sarext.

    Calculate Parabolic SAR Extended (SAREXT).

    More configurable version of the standard Parabolic SAR.

    Args:
      high: High prices.
      low: Low prices.
      start_value: Start value (default 0.0).
      offset_on_reversal: Offset on reversal (default 0.0).
      acceleration_init_long: Initial acceleration for long (default 0.02).
      acceleration_long: Acceleration increment for long (default 0.02).
      acceleration_max_long: Maximum acceleration for long (default 0.2).
      acceleration_init_short: Initial acceleration for short (default 0.02).
      acceleration_short: Acceleration increment for short (default 0.02).
      acceleration_max_short: Maximum acceleration for short (default 0.2).

    Returns:
      SARResult(index, sar)

    Configuration:
        start_value (float)
        offset_on_reversal (float)
        acceleration_init_long (float)
        acceleration_long (float)
        acceleration_max_long (float)
        acceleration_init_short (float)
        acceleration_short (float)
        acceleration_max_short (float)
    """

    start_value: float
    offset_on_reversal: float
    acceleration_init_long: float
    acceleration_long: float
    acceleration_max_long: float
    acceleration_init_short: float
    acceleration_short: float
    acceleration_max_short: float
    def __init__(self, *, start_value: float = ..., offset_on_reversal: float = ..., acceleration_init_long: float = ..., acceleration_long: float = ..., acceleration_max_long: float = ..., acceleration_init_short: float = ..., acceleration_short: float = ..., acceleration_max_short: float = ...) -> None: ...
    """Initialize configuration for sarext.

    Configuration:
        start_value (float)
        offset_on_reversal (float)
        acceleration_init_long (float)
        acceleration_long (float)
        acceleration_max_long (float)
        acceleration_init_short (float)
        acceleration_short (float)
        acceleration_max_short (float)
    """

    @override
    def make(self) -> _sarext_Bound: ...

class sarext:
    Type = _sarext_Bound
    Config = _sarext_Config
    ConfigDict = _sarext_ConfigDict
    start_value: ClassVar[float]
    offset_on_reversal: ClassVar[float]
    acceleration_init_long: ClassVar[float]
    acceleration_long: ClassVar[float]
    acceleration_max_long: ClassVar[float]
    acceleration_init_short: ClassVar[float]
    acceleration_short: ClassVar[float]
    acceleration_max_short: ClassVar[float]
    def __new__(cls, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], start_value: float = ..., offset_on_reversal: float = ..., acceleration_init_long: float = ..., acceleration_long: float = ..., acceleration_max_long: float = ..., acceleration_init_short: float = ..., acceleration_short: float = ..., acceleration_max_short: float = ...) -> SARResult: ...
