"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import SMAResult

class _sma_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> SMAResult: ...

class _sma_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for sma.

  Configuration:
      period (int)
  """

  period: int

class _sma_Config(_NCMakeableModel[_sma_Bound]):
  """Configuration class for sma.

  Calculate Simple Moving Average (SMA).

    SMA is a trend-following indicator that averages prices over a specified
    period. All prices are weighted equally.

    Formula:
    SMA = (P1 + P2 + ... + Pn) / n

    Interpretation:
    - Price above SMA: Bullish
    - Price below SMA: Bearish
    - SMA crossovers: Trend change signals
    - Multiple SMAs: Short crossing long = golden/death cross

    Common periods:
    - 10/20: Short-term trend
    - 50: Medium-term trend
    - 200: Long-term trend

    Features:
    - Numba-optimized for performance
    - Rolling sum algorithm for O(n) efficiency
    - Works with any numeric column

    Args:
      data: Input Series.
      period: Lookback period (default: 20)

    Returns:
      SMAResult(index, sma)

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
  from indikator.utils import to_numpy
      >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
      >>> result = sma(prices, period=5).to_pandas()

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for sma.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _sma_Bound: ...

class sma:
  Type = _sma_Bound
  Config = _sma_Config
  ConfigDict = _sma_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> SMAResult: ...
