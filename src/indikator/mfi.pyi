"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import MFIResult

class _mfi_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def period(self) -> int: ...
    @property
    def epsilon(self) -> float: ...
    def __call__(self, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty], volume: Validated[pd.Series, Finite, NotEmpty]) -> MFIResult: ...

class _mfi_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for mfi.

    Configuration:
        period (int)
        epsilon (float)
    """

    period: int
    epsilon: float

class _mfi_Config(_NCMakeableModel[_mfi_Bound]):
    """Configuration class for mfi.

    Calculate Money Flow Index (MFI).

    MFI is a momentum indicator that uses both price and volume to measure
    buying and selling pressure. It is also known as volume-weighted RSI.

    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Money Flow = Typical Price * Volume
    3. Positive Money Flow = sum of MF when typical price increases
    4. Negative Money Flow = sum of MF when typical price decreases
    5. Money Flow Ratio = Positive Money Flow / Negative Money Flow
    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = Typical Price * Volume
    Money Ratio = Positive Money Flow / Negative Money Flow
    MFI = 100 - (100 / (1 + Money Ratio))

    Interpretation:
    - MFI > 80: Overbought
    - MFI < 20: Oversold
    - Divergence: Price makes new high but MFI doesn't = reversal signal

    Features:
    - Numba-optimized for performance
    - Handles edge cases (division by zero)
    - Uses standard 14 period default

    Args:
      high: High prices Series.
      low: Low prices Series.
      close: Close prices Series.
      volume: Volume Series.
      period: Lookback period (default: 14)

    Returns:
      MFIResult(index, mfi)

    Configuration:
        period (int)
        epsilon (float)
    """

    period: int
    epsilon: float
    def __init__(self, *, period: int = ..., epsilon: float = ...) -> None: ...
    """Initialize configuration for mfi.

    Configuration:
        period (int)
        epsilon (float)
    """

    @override
    def make(self) -> _mfi_Bound: ...

class mfi:
    Type = _mfi_Bound
    Config = _mfi_Config
    ConfigDict = _mfi_ConfigDict
    period: ClassVar[int]
    epsilon: ClassVar[float]
    def __new__(cls, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty], volume: Validated[pd.Series, Finite, NotEmpty], period: int = ..., epsilon: float = ...) -> MFIResult: ...
