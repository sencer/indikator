"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import StochResult

class _stoch_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def k_period(self) -> int: ...
    @property
    def k_slowing(self) -> int: ...
    @property
    def d_period(self) -> int: ...
    def __call__(self, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty]) -> StochResult: ...

class _stoch_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for stoch.

    Configuration:
        k_period (int)
        k_slowing (int)
        d_period (int)
    """

    k_period: int
    k_slowing: int
    d_period: int

class _stoch_Config(_NCMakeableModel[_stoch_Bound]):
    """Configuration class for stoch.

    Calculate Stochastic Oscillator.

    The Stochastic Oscillator is a momentum indicator comparing closing price
    to its price range over a given time period.

    Formula:
    %K = 100 * SMA((Close - Lowest Low) / (Highest High - Lowest Low), k_slowing)
    %D = SMA(%K, d_period)

    Interpretation:
    - %K > 80: Overbought
    - %K < 20: Oversold
    - %K crossing %D: Signal (bullish if crossing up, bearish if crossing down)
    - Divergence: Price making new high but stochastic doesn't = bearish

    Common strategies:
    - Buy when %K crosses above %D below 20
    - Sell when %K crosses below %D above 80

    Features:
    - Numba-optimized for performance
    - Standard Fast Stochastic with slowing
    - Returns both %K and %D lines

    Args:
      high: High prices Series.
      low: Low prices Series.
      close: Close prices Series.
      k_period: Period for highest high / lowest low (default: 14)
      k_slowing: Slowing period for %K (default: 3)
      d_period: Period for %D smoothing (default: 3)

    Returns:
      DataFrame with 'stoch_k' and 'stoch_d' columns

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
      >>> high = pd.Series([105, 107, 106, 108, 110])
      >>> low = pd.Series([100, 102, 101, 103, 105])
      >>> close = pd.Series([102, 105, 104, 106, 108])
      >>> result = stoch(high, low, close)

    Configuration:
        k_period (int)
        k_slowing (int)
        d_period (int)
    """

    k_period: int
    k_slowing: int
    d_period: int
    def __init__(self, *, k_period: int = ..., k_slowing: int = ..., d_period: int = ...) -> None: ...
    """Initialize configuration for stoch.

    Configuration:
        k_period (int)
        k_slowing (int)
        d_period (int)
    """

    @override
    def make(self) -> _stoch_Bound: ...

class stoch:
    Type = _stoch_Bound
    Config = _stoch_Config
    ConfigDict = _stoch_ConfigDict
    k_period: ClassVar[int]
    k_slowing: ClassVar[int]
    d_period: ClassVar[int]
    def __new__(cls, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty], k_period: int = ..., k_slowing: int = ..., d_period: int = ...) -> StochResult: ...

class _stochf_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def fastk_period(self) -> int: ...
    @property
    def fastd_period(self) -> int: ...
    def __call__(self, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty]) -> StochResult: ...

class _stochf_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for stochf.

    Configuration:
        fastk_period (int)
        fastd_period (int)
    """

    fastk_period: int
    fastd_period: int

class _stochf_Config(_NCMakeableModel[_stochf_Bound]):
    """Configuration class for stochf.

    Calculate Fast Stochastic Oscillator (STOCHF).

    The Fast Stochastic Oscillator corresponds to the %K and %D lines where:
    %K is the raw stochastic (unsmoothed).
    %D is the SMA of %K.

    This is equivalent to calling `stoch` with `k_slowing=1`.

    Args:
      high: High prices Series.
      low: Low prices Series.
      close: Close prices Series.
      fastk_period: Period for raw %K calculation (default: 5)
      fastd_period: Period for %D smoothing (default: 3)

    Returns:
      StochResult(index, stoch_k, stoch_d)
      where stoch_k is Fast %K and stoch_d is Fast %D.

    Configuration:
        fastk_period (int)
        fastd_period (int)
    """

    fastk_period: int
    fastd_period: int
    def __init__(self, *, fastk_period: int = ..., fastd_period: int = ...) -> None: ...
    """Initialize configuration for stochf.

    Configuration:
        fastk_period (int)
        fastd_period (int)
    """

    @override
    def make(self) -> _stochf_Bound: ...

class stochf:
    Type = _stochf_Bound
    Config = _stochf_Config
    ConfigDict = _stochf_ConfigDict
    fastk_period: ClassVar[int]
    fastd_period: ClassVar[int]
    def __new__(cls, high: Validated[pd.Series, Finite, NotEmpty], low: Validated[pd.Series, Finite, NotEmpty], close: Validated[pd.Series, Finite, NotEmpty], fastk_period: int = ..., fastd_period: int = ...) -> StochResult: ...
