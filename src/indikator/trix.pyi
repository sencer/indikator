"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _trix_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

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
  - TRIX > 0: Momentum is positive (uptrend)
  - TRIX < 0: Momentum is negative (downtrend)
  - Signal line crossover can be used for entries/exits
  - Divergences indicate potential reversals

  Features:
  - Numba-optimized for performance
  - Filters out insignificant price movements (due to triple smoothing)
  - Standard 30 period default (often 15 or 30)

  Args:
    data: Input Series.
    period: Lookback period (default: 30)

  Returns:
    IndicatorResult(index, trix)

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
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...
