"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Datetime, Finite, Index, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import RVOLResult

class _rvol_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def window(self) -> int: ...
    @property
    def epsilon(self) -> float: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> RVOLResult: ...

class _rvol_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for rvol.

    Configuration:
        window (int)
        epsilon (float)
    """

    window: int
    epsilon: float

class _rvol_Config(_NCMakeableModel[_rvol_Bound]):
    """Configuration class for rvol.

    Calculate Relative Volume (RVOL) using simple moving average.

    Measures current volume relative to average volume over a rolling window.

    Formula:
    RVOL = Volume / SMA(Volume, window)

    Interpretation:
    - RVOL > 1.0: Above-average volume
    - RVOL > 2.0: High volume spike
    - RVOL < 1.0: Below-average volume

    Args:
      data: Volume Series
      window: Rolling window size (default: 14)
      epsilon: Small value to prevent division by zero

    Returns:
      RVOLResult(index, rvol)

    Configuration:
        window (int)
        epsilon (float)
    """

    window: int
    epsilon: float
    def __init__(self, *, window: int = ..., epsilon: float = ...) -> None: ...
    """Initialize configuration for rvol.

    Configuration:
        window (int)
        epsilon (float)
    """

    @override
    def make(self) -> _rvol_Bound: ...

class rvol:
    Type = _rvol_Bound
    Config = _rvol_Config
    ConfigDict = _rvol_ConfigDict
    window: ClassVar[int]
    epsilon: ClassVar[float]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], window: int = ..., epsilon: float = ...) -> RVOLResult: ...

class _rvol_intraday_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def min_samples(self) -> int: ...
    @property
    def epsilon(self) -> float: ...
    def __call__(self, data: Validated[pd.Series, Finite, Index(Datetime), NotEmpty], lookback_days: int | None = ...) -> RVOLResult: ...

class _rvol_intraday_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for rvol_intraday.

    Configuration:
        min_samples (int)
        epsilon (float)
    """

    min_samples: int
    epsilon: float

class _rvol_intraday_Config(_NCMakeableModel[_rvol_intraday_Bound]):
    """Configuration class for rvol_intraday.

    Calculate intraday RVOL based on time-of-day historical averages.

    Compares current volume to the historical average volume for that specific
    time of day (e.g., 10:30 AM today vs. all previous 10:30 AM bars).

    Formula:
    RVOL = Volume / AvgVolume(TimeOfDay)

    Features:
    - Accounts for natural intraday volume patterns
    - Configurable lookback period

    Args:
      data: Series (e.g., volume) with DatetimeIndex
      lookback_days: Number of days to look back
      min_samples: Minimum observations required
      epsilon: Small value to prevent division by zero asd

    Returns:
      RVOLResult(index, rvol)

    Configuration:
        min_samples (int)
        epsilon (float)
    """

    min_samples: int
    epsilon: float
    def __init__(self, *, min_samples: int = ..., epsilon: float = ...) -> None: ...
    """Initialize configuration for rvol_intraday.

    Configuration:
        min_samples (int)
        epsilon (float)
    """

    @override
    def make(self) -> _rvol_intraday_Bound: ...

class rvol_intraday:
    Type = _rvol_intraday_Bound
    Config = _rvol_intraday_Config
    ConfigDict = _rvol_intraday_ConfigDict
    min_samples: ClassVar[int]
    epsilon: ClassVar[float]
    def __new__(cls, data: Validated[pd.Series, Finite, Index(Datetime), NotEmpty], lookback_days: int | None = ..., min_samples: int = ..., epsilon: float = ...) -> RVOLResult: ...
