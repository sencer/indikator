"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ROCResult

class _roc_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(self, data: Validated[pd.Series, Finite, NotEmpty]) -> ROCResult: ...

class _roc_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for roc.

  Configuration:
      period (int)
  """

  period: int

class _roc_Config(_NCMakeableModel[_roc_Bound]):
  """Configuration class for roc.

  Calculate Rate of Change (ROC).

  ROC is a momentum oscillator that measures the percentage change between
  the current price and the price n periods ago.

  Formula:
  ROC = ((Price - Price_n_periods_ago) / Price_n_periods_ago) * 100

  Interpretation:
  - ROC > 0: Price is higher than n periods ago (bullish)
  - ROC < 0: Price is lower than n periods ago (bearish)
  - ROC crossing 0: Momentum shift
  - Extreme readings: Potential reversal

  Common strategies:
  - Buy on positive ROC (momentum confirmation)
  - Sell on negative ROC
  - Look for divergences between price and ROC

  Features:
  - Numba-optimized for performance
  - Standard 10-period default
  - Simple percentage-based calculation

  Args:
    data: Input Series.
    period: Lookback period (default: 10)

  Returns:
    ROCResult(index, roc)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120])
    >>> result = roc(prices, period=5).to_pandas()

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for roc.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _roc_Bound: ...

class roc:
  Type = _roc_Bound
  Config = _roc_Config
  ConfigDict = _roc_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series, Finite, NotEmpty], period: int = ...
  ) -> ROCResult: ...
