"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NonEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _macd_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def fast_period(self) -> int: ...
  @property
  def slow_period(self) -> int: ...
  @property
  def signal_period(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NonEmpty]) -> pd.DataFrame: ...

class _macd_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for macd.

  Configuration:
      fast_period (int)
      slow_period (int)
      signal_period (int)
  """

  fast_period: int
  slow_period: int
  signal_period: int

class _macd_Config(_NCMakeableModel[_macd_Bound]):
  """Configuration class for macd.

  Calculate MACD (Moving Average Convergence Divergence).

  MACD is a trend-following momentum indicator that shows the relationship
  between two exponential moving averages (EMAs) of a price series.

  Components:
  1. MACD Line = EMA(fast_period) - EMA(slow_period)
  2. Signal Line = EMA(MACD Line, signal_period)
  3. Histogram = MACD Line - Signal Line

  Interpretation:
  - MACD > 0: Price is above the slow EMA (bullish)
  - MACD < 0: Price is below the slow EMA (bearish)
  - MACD crossing above signal: Bullish signal (buy)
  - MACD crossing below signal: Bearish signal (sell)
  - Histogram growing: Trend strengthening
  - Histogram shrinking: Trend weakening
  - Divergence: Price makes new high but MACD doesn't = bearish

  Common strategies:
  - Signal crossovers: Buy on MACD cross above signal, sell on cross below
  - Zero crossovers: Buy on MACD cross above 0, sell on cross below 0
  - Divergence: Look for price/MACD divergences for reversal signals
  - Histogram: Use histogram for early trend change detection

  Features:
  - Numba-optimized for performance
  - Standard parameters (12, 26, 9) by default
  - Returns all three components
  - Works with any numeric column

  Args:
    data: Input Series.
    fast_period: Fast EMA period (default: 12)
    slow_period: Slow EMA period (default: 26)
    signal_period: Signal line EMA period (default: 9)

  Returns:
    DataFrame with 'macd', 'macd_signal', 'macd_histogram' columns

  Raises:
    ValueError: If fast_period >= slow_period

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    >>> result = macd(prices)
    >>> # Returns DataFrame with MACD components

  Configuration:
      fast_period (int)
      slow_period (int)
      signal_period (int)
  """

  fast_period: int
  slow_period: int
  signal_period: int
  def __init__(
    self, *, fast_period: int = ..., slow_period: int = ..., signal_period: int = ...
  ) -> None: ...
  """Initialize configuration for macd.

    Configuration:
        fast_period (int)
        slow_period (int)
        signal_period (int)
    """

  @override
  def make(self) -> _macd_Bound: ...

class macd:
  Type = _macd_Bound
  Config = _macd_Config
  ConfigDict = _macd_ConfigDict
  fast_period: ClassVar[int]
  slow_period: ClassVar[int]
  signal_period: ClassVar[int]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NonEmpty],
    fast_period: int = ...,
    slow_period: int = ...,
    signal_period: int = ...,
  ) -> pd.DataFrame: ...
