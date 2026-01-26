"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import IndicatorResult

class _kama_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  @property
  def fast_period(self) -> int: ...
  @property
  def slow_period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> IndicatorResult: ...

class _kama_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for kama.

  Configuration:
      period (int)
      fast_period (int)
      slow_period (int)
  """

  period: int
  fast_period: int
  slow_period: int

class _kama_Config(_NCMakeableModel[_kama_Bound]):
  """Configuration class for kama.

  Calculate Kaufman Adaptive Moving Average (KAMA).

  KAMA adjusts its smoothing constant based on market efficiency:
  - In trending markets (high efficiency): responds quickly
  - In choppy markets (low efficiency): smooths more

  Formula:
  ER = |Price - Price_n_ago| / sum(|daily changes|)
  SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
  KAMA = KAMA_prev + SC * (Price - KAMA_prev)

  Interpretation:
  - KAMA slope indicates trend direction
  - Flat KAMA suggests ranging market
  - Price crossing KAMA can signal trend changes

  Features:
  - O(1) rolling volatility calculation
  - Adaptive smoothing responds to market conditions

  Args:
    data: Input price Series (typically close prices)
    period: Efficiency ratio lookback (default: 10)
    fast_period: Fast smoothing period (default: 2)
    slow_period: Slow smoothing period (default: 30)

  Returns:
    IndicatorResult with adaptive moving average values

  Example:
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> result = kama(prices, period=10)

  Configuration:
      period (int)
      fast_period (int)
      slow_period (int)
  """

  period: int
  fast_period: int
  slow_period: int
  def __init__(
    self, *, period: int = ..., fast_period: int = ..., slow_period: int = ...
  ) -> None: ...
  """Initialize configuration for kama.

    Configuration:
        period (int)
        fast_period (int)
        slow_period (int)
    """

  @override
  def make(self) -> _kama_Bound: ...

class kama:
  Type = _kama_Bound
  Config = _kama_Config
  ConfigDict = _kama_ConfigDict
  period: ClassVar[int]
  fast_period: ClassVar[int]
  slow_period: ClassVar[int]
  def __new__(
    cls,
    data: Validated[pd.Series[float], Finite, NotEmpty],
    period: int = ...,
    fast_period: int = ...,
    slow_period: int = ...,
  ) -> IndicatorResult: ...
