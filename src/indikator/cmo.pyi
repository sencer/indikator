"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import CMOResult

class _cmo_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> CMOResult: ...

class _cmo_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for cmo.

    Configuration:
        period (int)
    """

    period: int

class _cmo_Config(_NCMakeableModel[_cmo_Bound]):
    """Configuration class for cmo.

    Calculate Chande Momentum Oscillator (CMO).

    CMO measures the momentum of price changes. It oscillates between -100
    and +100, making it useful for identifying overbought/oversold conditions.

    Formula:
    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)

    Interpretation:
    - CMO > 50: Overbought (potential reversal down)
    - CMO < -50: Oversold (potential reversal up)
    - CMO = 0: Equal gains and losses
    - Crossing zero: Momentum shift

    Unlike RSI which divides sum_gains by sum_losses, CMO uses their
    difference divided by their sum, giving a true center at zero.

    Features:
    - Numba-optimized for performance
    - O(n) sliding window algorithm
    - Range: -100 to +100 (centered at 0)

    Args:
      data: Input Series (typically closing prices)
      period: Lookback period (default: 14)

    Returns:
      CMOResult(index, cmo)

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
      >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 3)
      >>> result = cmo(prices, period=5).to_pandas()
      >>> # Returns CMO values

    Configuration:
        period (int)
    """

    period: int
    def __init__(self, *, period: int = ...) -> None: ...
    """Initialize configuration for cmo.

    Configuration:
        period (int)
    """

    @override
    def make(self) -> _cmo_Bound: ...

class cmo:
    Type = _cmo_Bound
    Config = _cmo_Config
    ConfigDict = _cmo_ConfigDict
    period: ClassVar[int]
    def __new__(cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...) -> CMOResult: ...
