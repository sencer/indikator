"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _wma_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _wma_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for wma.

  Configuration:
      period (int)
  """

  period: int

class _wma_Config(_NCMakeableModel[_wma_Bound]):
  """Configuration class for wma.

  Calculate Weighted Moving Average (WMA).

  WMA assigns linearly increasing weights to prices, with the most
  recent price having the highest weight.

  Formula:
  WMA = (P1*1 + P2*2 + ... + Pn*n) / (1 + 2 + ... + n)

  Interpretation:
  - More responsive than SMA, less than EMA
  - Smooth trend following
  - Good for identifying trend direction

  Features:
  - O(1) rolling update per step (not O(period))
  - Uses weighted/unweighted sum trick for efficiency

  Args:
    data: Input price Series (typically close prices)
    period: Lookback period (default: 20)

  Returns:
    IndicatorResult with weighted moving average values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = wma(prices, period=5)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for wma.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _wma_Bound: ...

class wma:
  Type = _wma_Bound
  Config = _wma_Config
  ConfigDict = _wma_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> IndicatorResult: ...
