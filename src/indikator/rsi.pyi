"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import RSIResult

class _rsi_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def window(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> RSIResult: ...

class _rsi_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for rsi.

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float

class _rsi_Config(_NCMakeableModel[_rsi_Bound]):
  """Configuration class for rsi.

  Calculate Relative Strength Index (RSI).

  RSI is a momentum oscillator that measures the speed and magnitude of
  price changes. It oscillates between 0 and 100, with readings above 70
  typically considered overbought and below 30 considered oversold.

  Formula:
  RSI = 100 - (100 / (1 + RS))
  where RS = Average Gain / Average Loss over N periods

  Uses Wilder's smoothing method for averaging:
  - First average: simple average of gains/losses over 'window' periods
  - Subsequent averages: (previous avg * (window-1) + current value) / window

  Interpretation:
  - RSI > 70: Overbought (potential reversal down)
  - RSI < 30: Oversold (potential reversal up)
  - RSI = 50: Neutral (no clear momentum)
  - RSI crossing 50: Momentum shift (bullish if crossing up, bearish if down)
  - Divergence: RSI making higher lows while price makes lower lows = bullish

  Common strategies:
  - Mean reversion: Sell when RSI > 70, buy when RSI < 30
  - Trend following: Buy when RSI crosses above 50 in uptrend
  - Divergence trading: Look for price/RSI divergences

  Features:
  - Numba-optimized for performance
  - Wilder's smoothing (original method)
  - Handles edge cases (no losses, no gains)
  - Works with any numeric column

  Args:
    data: Input Series.
    window: Lookback period (default: 14, Wilder's original)
    epsilon: Small value to prevent division by zero

  Returns:
    Series with RSI values (0-100 range)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> result = rsi(prices, window=5)
    >>> # Returns RSI values (typically 30-70 range)

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float
  def __init__(self, *, window: int = ..., epsilon: float = ...) -> None: ...
  """Initialize configuration for rsi.

    Configuration:
        window (int)
        epsilon (float)
    """

  @override
  def make(self) -> _rsi_Bound: ...

class rsi:
  Type = _rsi_Bound
  Config = _rsi_Config
  ConfigDict = _rsi_ConfigDict
  window: ClassVar[int]
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    window: int = ...,
    epsilon: float = ...,
  ) -> RSIResult: ...
