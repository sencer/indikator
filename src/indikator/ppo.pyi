"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _ppo_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def fast_period(self) -> int: ...
    @property
    def slow_period(self) -> int: ...
    @property
    def matype(self) -> int: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _ppo_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for ppo.

    Configuration:
        fast_period (int)
        slow_period (int)
        matype (int)
    """

    fast_period: int
    slow_period: int
    matype: int

class _ppo_Config(_NCMakeableModel[_ppo_Bound]):
    """Configuration class for ppo.

    Calculate Percentage Price Oscillator (PPO).

    PPO = (FastMA - SlowMA) / SlowMA * 100

    Args:
      data: Input Series.
      fast_period: Fast MA period (default 12).
      slow_period: Slow MA period (default 26).
      matype: Moving Average type (0=SMA, 1=EMA). Default 0.

    Returns:
      IndicatorResult

    Configuration:
        fast_period (int)
        slow_period (int)
        matype (int)
    """

    fast_period: int
    slow_period: int
    matype: int
    def __init__(self, *, fast_period: int = ..., slow_period: int = ..., matype: int = ...) -> None: ...
    """Initialize configuration for ppo.

    Configuration:
        fast_period (int)
        slow_period (int)
        matype (int)
    """

    @override
    def make(self) -> _ppo_Bound: ...

class ppo:
    Type = _ppo_Bound
    Config = _ppo_Config
    ConfigDict = _ppo_ConfigDict
    fast_period: ClassVar[int]
    slow_period: ClassVar[int]
    matype: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], fast_period: int = ..., slow_period: int = ..., matype: int = ...) -> IndicatorResult: ...
