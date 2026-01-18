"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ADXResult

class _adx_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> ADXResult: ...

class _adx_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for adx.

  Configuration:
      period (int)
  """

  period: int

class _adx_Config(_NCMakeableModel[_adx_Bound]):
  """Configuration class for adx.

  Calculate Average Directional Index (ADX).

  ADX measures trend strength regardless of direction. It's derived from
  the Directional Movement System developed by Welles Wilder.

  Components:
  - ADX: Average Directional Index (trend strength)
  - +DI: Plus Directional Indicator (bullish pressure)
  - -DI: Minus Directional Indicator (bearish pressure)

  Interpretation:
  - ADX < 20: Weak trend / ranging market
  - ADX 20-25: Trend emerging
  - ADX 25-50: Strong trend
  - ADX 50-75: Very strong trend
  - ADX > 75: Extremely strong trend

  Directional Indicators:
  - +DI > -DI: Bullish
  - -DI > +DI: Bearish
  - +DI crossing above -DI: Buy signal
  - -DI crossing above +DI: Sell signal

  Common strategies:
  - Trade only when ADX > 25 (confirms trend)
  - Use DI crossovers for entry signals
  - Exit when ADX starts declining

  Features:
  - Numba-optimized for performance
  - Wilder's smoothing method
  - Returns ADX and both directional indicators

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    DataFrame with 'adx', 'plus_di', 'minus_di' columns

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> high = pd.Series([105, 107, 106, 108, 110])
    >>> low = pd.Series([100, 102, 101, 103, 105])
    >>> close = pd.Series([102, 105, 104, 106, 108])
    >>> result = adx(high, low, close)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for adx.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _adx_Bound: ...

class adx:
  Type = _adx_Bound
  Config = _adx_Config
  ConfigDict = _adx_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> ADXResult: ...
