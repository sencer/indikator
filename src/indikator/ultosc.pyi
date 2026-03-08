"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _ultosc_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period1(self) -> int: ...
  @property
  def period2(self) -> int: ...
  @property
  def period3(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> IndicatorResult: ...

class _ultosc_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for ultosc.

  Configuration:
      period1 (int)
      period2 (int)
      period3 (int)
  """

  period1: int
  period2: int
  period3: int

class _ultosc_Config(_NCMakeableModel[_ultosc_Bound]):
  """Configuration class for ultosc.

  Calculate Ultimate Oscillator (ULTOSC).

  ULTOSC combines momentum from three timeframes to reduce false
  signals common in single-period oscillators.

  Formula:
  BP = Close - min(Low, Prior Close)
  TR = max(High, Prior Close) - min(Low, Prior Close)
  ULTOSC = 100 * (4*Avg1 + 2*Avg2 + 1*Avg3) / 7

  Interpretation:
  - ULTOSC > 70: Overbought
  - ULTOSC < 30: Oversold
  - Divergence with price: Potential reversal

  Features:
  - O(1) rolling sums using circular buffers
  - Three weighted timeframes reduce noise

  Args:
    high: High prices
    low: Low prices
    close: Close prices
    period1: Short period (default: 7, weight 4)
    period2: Medium period (default: 14, weight 2)
    period3: Long period (default: 28, weight 1)

  Returns:
    IndicatorResult with oscillator values (0-100)

  Example:
    >>> result = ultosc(high, low, close, period1=7, period2=14, period3=28)

  Configuration:
      period1 (int)
      period2 (int)
      period3 (int)
  """

  period1: int
  period2: int
  period3: int
  def __init__(
    self, *, period1: int = ..., period2: int = ..., period3: int = ...
  ) -> None: ...
  """Initialize configuration for ultosc.

    Configuration:
        period1 (int)
        period2 (int)
        period3 (int)
    """

  @override
  def make(self) -> _ultosc_Bound: ...

class ultosc:
  Type = _ultosc_Bound
  Config = _ultosc_Config
  ConfigDict = _ultosc_ConfigDict
  period1: ClassVar[int]
  period2: ClassVar[int]
  period3: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
    period1: int = ...,
    period2: int = ...,
    period3: int = ...,
  ) -> IndicatorResult: ...
