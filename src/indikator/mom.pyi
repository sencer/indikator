"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _mom_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> IndicatorResult: ...

class _mom_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for mom.

    Configuration:
        period (int)
    """

    period: int

class _mom_Config(_NCMakeableModel[_mom_Bound]):
    """Configuration class for mom.

    Calculate Momentum (MOM).

    Momentum measures the absolute price change over a specified period.
    Unlike ROC which shows percentage change, MOM shows raw price difference.

    Formula:
    MOM = Price(t) - Price(t - period)

    Interpretation:
    - Positive MOM: Price ascending
    - Negative MOM: Price descending
    - Zero crossing: Potential trend change
    - Divergence from price: Potential reversal

    Common uses:
    - Trend confirmation
    - Overbought/oversold detection
    - Divergence analysis
    - Leading indicator for price reversals

    Features:
    - Numba-optimized with parallel execution
    - Simple and fast O(N) calculation

    Args:
      data: Input price Series (typically close prices)
      period: Lookback period (default: 10)

    Returns:
      IndicatorResult with momentum values

    Example:
      >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
      >>> result = mom(prices, period=3)

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for mom.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _mom_Bound: ...

class mom:
    Type = _mom_Bound
    Config = _mom_Config
    ConfigDict = _mom_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...) -> IndicatorResult: ...
