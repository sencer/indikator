"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import BollingerBandsResult, BollingerResult

class _bollinger_bands_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def window(self) -> int: ...
    @property
    def num_std(self) -> float: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> BollingerBandsResult: ...

class _bollinger_bands_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for bollinger_bands.

    Configuration:
        window (int)
        num_std (float)
    """

    window: int
    num_std: float

class _bollinger_bands_Config(_NCMakeableModel[_bollinger_bands_Bound]):
    """Configuration class for bollinger_bands.

    Calculate Bollinger Bands.

      Bollinger Bands consist of a middle band (SMA) and two outer bands that
      are standard deviations away from the middle band. They expand and contract
      based on market volatility.

      This implementation uses population standard deviation (ddof=0) to match
      TA-lib behavior. For bandwidth and %B metrics, use `bollinger_with_bandwidth()`.

      Components:
      - Middle Band = SMA(close, window)
      - Upper Band = Middle Band + (std_dev * num_std)
      - Lower Band = Middle Band - (std_dev * num_std)

      Interpretation:
      - Price near upper band: Overbought
      - Price near lower band: Oversold
      - Bands squeezing: Low volatility, potential breakout coming
      - Bands expanding: High volatility, trend in motion

      Args:
        data: Input Series.
        window: Rolling window size (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)

      Returns:
        BollingerBandsResult(index, bb_upper, bb_middle, bb_lower)

      Example:
        >>> import pandas as pd
    from indikator.utils import to_numpy
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> result = bollinger_bands(prices, window=5, num_std=2.0)
        >>> result.bb_upper  # Access as array
        >>> df = result.to_pandas()  # Convert to DataFrame

    Configuration:
        window (int)
        num_std (float)
    """

    window: int
    num_std: float
    def __init__(self, *, window: int = ..., num_std: float = ...) -> None: ...
    """Initialize configuration for bollinger_bands.

    Configuration:
        window (int)
        num_std (float)
    """

    @override
    def make(self) -> _bollinger_bands_Bound: ...

class bollinger_bands:
    Type = _bollinger_bands_Bound
    Config = _bollinger_bands_Config
    ConfigDict = _bollinger_bands_ConfigDict
    window: ClassVar[int]
    num_std: ClassVar[float]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], window: int = ..., num_std: float = ...) -> BollingerBandsResult: ...

class _bollinger_with_bandwidth_Bound(Protocol):
    """Bound function with hyperparameters as attributes."""
    @property
    def window(self) -> int: ...
    @property
    def num_std(self) -> float: ...
    def __call__(self, data: Validated[pd.Series[float], Finite, NotEmpty]) -> BollingerResult: ...

class _bollinger_with_bandwidth_ConfigDict(TypedDict, total=False):
    """Configuration dictionary for bollinger_with_bandwidth.

    Configuration:
        window (int)
        num_std (float)
    """

    window: int
    num_std: float

class _bollinger_with_bandwidth_Config(_NCMakeableModel[_bollinger_with_bandwidth_Bound]):
    """Configuration class for bollinger_with_bandwidth.

    Calculate Bollinger Bands with bandwidth and %B metrics.

      Extended Bollinger Bands calculation that includes bandwidth and %B
      in addition to the standard three bands. Uses sample standard deviation
      (ddof=1) which is statistically more correct for estimating population
      parameters from a sample.

      Components:
      - Middle Band = SMA(close, window)
      - Upper Band = Middle Band + (std_dev * num_std)
      - Lower Band = Middle Band - (std_dev * num_std)
      - Bandwidth = (Upper Band - Lower Band) / abs(Middle Band)
      - %B = (Price - Lower Band) / (Upper Band - Lower Band)

      Interpretation:
      - %B > 1: Price above upper band (very overbought)
      - %B < 0: Price below lower band (very oversold)
      - %B = 0.5: Price at middle band

      Note: Uses sample std (ddof=1), so bands will be slightly wider than
      the standard `bollinger_bands()` which uses population std (ddof=0).

      Args:
        data: Input Series.
        window: Rolling window size (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)

      Returns:
        BollingerResult with bb_middle, bb_upper, bb_lower, bb_bandwidth, bb_percent

      Example:
        >>> import pandas as pd
    from indikator.utils import to_numpy
        >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        >>> result = bollinger_with_bandwidth(prices, window=5, num_std=2.0)
        >>> print(result.bb_bandwidth)  # Bandwidth values
        >>> print(result.bb_percent)    # %B values

    Configuration:
        window (int)
        num_std (float)
    """

    window: int
    num_std: float
    def __init__(self, *, window: int = ..., num_std: float = ...) -> None: ...
    """Initialize configuration for bollinger_with_bandwidth.

    Configuration:
        window (int)
        num_std (float)
    """

    @override
    def make(self) -> _bollinger_with_bandwidth_Bound: ...

class bollinger_with_bandwidth:
    Type = _bollinger_with_bandwidth_Bound
    Config = _bollinger_with_bandwidth_Config
    ConfigDict = _bollinger_with_bandwidth_ConfigDict
    window: ClassVar[int]
    num_std: ClassVar[float]
    def __new__(cls, data: Validated[pd.Series[float], Finite, NotEmpty], window: int = ..., num_std: float = ...) -> BollingerResult: ...
