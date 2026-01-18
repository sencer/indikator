"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _trix_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> pd.Series: ...

class _trix_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for trix.

  Configuration:
      period (int)
  """

  period: int

class _trix_Config(_NCMakeableModel[_trix_Bound]):
  """Configuration class for trix.

  Calculate TRIX (Triple Exponential Average).

  TRIX is a momentum oscillator that displays the percent rate of change
  of a triple exponentially smoothed moving average. It filters out minor
  price movements, making it useful for identifying overbought/oversold
  conditions and divergences.

  Formula:
  1. EMA1 = EMA(Price, period)
  2. EMA2 = EMA(EMA1, period)
  3. EMA3 = EMA(EMA2, period)
  4. TRIX = ((EMA3[today] - EMA3[yesterday]) / EMA3[yesterday]) * 100

  Interpretation:
  - TRIX > 0: Bullish momentum (triple EMA rising)
  - TRIX < 0: Bearish momentum (triple EMA falling)
  - Zero line crossovers: Trend change signals
  - Divergence: TRIX direction differs from price (reversal signal)

  Common strategies:
  - Signal line: Use 9-period EMA of TRIX as signal line
  - Zero line crossovers: Buy when TRIX crosses above 0
  - Divergence trading: Look for price/TRIX divergences

  Features:
  - Numba-optimized for performance
  - Triple smoothing eliminates short-term noise
  - Shows rate of change, not absolute values
  - Good for identifying trend changes

  Args:
    data: Input Series (typically closing prices)
    period: EMA period (default: 14)

  Returns:
    Series with TRIX values (percentage)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 5)
    >>> result = trix(prices, period=5)
    >>> # Returns TRIX values (typically small percentages near 0)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for trix.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _trix_Bound: ...

class trix:
  Type = _trix_Bound
  Config = _trix_Config
  ConfigDict = _trix_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...
  ) -> pd.Series: ...
