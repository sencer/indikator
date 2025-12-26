"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd
from validated import Finite, NonEmpty, Validated

class _bollinger_bands_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def window(self) -> int: ...
  @property
  def num_std(self) -> float: ...
  def __call__(self, data: Validated[pd.Series, Finite, NonEmpty]) -> pd.DataFrame: ...

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

  Components:
  - Middle Band = SMA(close, window)
  - Upper Band = Middle Band + (std_dev * num_std)
  - Lower Band = Middle Band - (std_dev * num_std)
  - Bandwidth = (Upper Band - Lower Band) / Middle Band
  - %B = (Price - Lower Band) / (Upper Band - Lower Band)

  Interpretation:
  - Price near upper band: Overbought
  - Price near lower band: Oversold
  - Bands squeezing: Low volatility, potential breakout coming
  - Bands expanding: High volatility, trend in motion
  - %B > 1: Price above upper band (very overbought)
  - %B < 0: Price below lower band (very oversold)
  - %B = 0.5: Price at middle band

  Common strategies:
  - Mean reversion: Sell at upper band, buy at lower band
  - Breakout: Buy when price breaks above upper band with expanding bands
  - Squeeze: Enter when bands squeeze then expand (volatility breakout)
  - Walking the bands: Strong trends "walk" along one band

  Features:
  - Pandas optimized (rolling window operations)
  - Configurable window and standard deviation multiplier
  - Returns all components (bands, bandwidth, %B)
  - Works with any numeric column

  Args:
    data: Input Series.
    window: Rolling window size (default: 20)
    num_std: Number of standard deviations for bands (default: 2.0)

  Returns:
    DataFrame with 'bb_middle', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'bb_percent' columns

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120])
    >>> result = bollinger_bands(prices, window=10, num_std=2.0)
    >>> # Returns DataFrame with all Bollinger Band components

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
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NonEmpty],
    window: int = ...,
    num_std: float = ...,
  ) -> pd.DataFrame: ...
