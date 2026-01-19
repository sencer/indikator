"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ADXResult, ADXSingleResult

class _adx_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> ADXSingleResult: ...

class _adx_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for adx.

  Configuration:
      period (int)
  """

  period: int

class _adx_Config(_NCMakeableModel[_adx_Bound]):
  """Configuration class for adx.

  Calculate Average Directional Index (ADX).

  ADX measures trend strength regardless of direction. This function returns
  only the ADX series for maximum performance (matching TA-Lib).

  For Directional Indicators (+DI, -DI), use `adx_with_di()`.

  Interpretation:
  - ADX < 20: Weak trend / ranging market
  - ADX 25-50: Strong trend
  - ADX > 75: Extremely strong trend

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ADXSingleResult(index, adx)

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
  ) -> ADXSingleResult: ...

class _adx_with_di_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> ADXResult: ...

class _adx_with_di_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for adx_with_di.

  Configuration:
      period (int)
  """

  period: int

class _adx_with_di_Config(_NCMakeableModel[_adx_with_di_Bound]):
  """Configuration class for adx_with_di.

  Calculate Average Directional Index (ADX) with DI components.

  Extended calculation that returns +DI and -DI alongside ADX.

  Components:
  - ADX: Average Directional Index (trend strength)
  - +DI: Plus Directional Indicator (bullish pressure)
  - -DI: Minus Directional Indicator (bearish pressure)

  Directional Indicators:
  - +DI > -DI: Bullish
  - -DI > +DI: Bearish
  - +DI crossing above -DI: Buy signal
  - -DI crossing above +DI: Sell signal

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ADXResult object with adx, plus_di, minus_di series.

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for adx_with_di.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _adx_with_di_Bound: ...

class adx_with_di:
  Type = _adx_with_di_Bound
  Config = _adx_with_di_Config
  ConfigDict = _adx_with_di_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> ADXResult: ...
