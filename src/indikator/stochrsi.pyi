"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import StochRSIResult

class _stochrsi_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def rsi_period(self) -> int: ...
  @property
  def stoch_period(self) -> int: ...
  @property
  def k_period(self) -> int: ...
  @property
  def d_period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series, Finite, NotEmpty]
  ) -> StochRSIResult: ...

class _stochrsi_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for stochrsi.

  Configuration:
      rsi_period (int)
      stoch_period (int)
      k_period (int)
      d_period (int)
  """

  rsi_period: int
  stoch_period: int
  k_period: int
  d_period: int

class _stochrsi_Config(_NCMakeableModel[_stochrsi_Bound]):
  """Configuration class for stochrsi.

  Calculate Stochastic RSI (STOCHRSI).

  StochRSI applies the Stochastic oscillator formula to RSI values,
  creating a more sensitive indicator that oscillates between 0-100.

  Formula:
  StochRSI = (RSI - min(RSI, n)) / (max(RSI, n) - min(RSI, n)) * 100
  %K = SMA(StochRSI, k_period)
  %D = SMA(%K, d_period)

  Interpretation:
  - StochRSI > 80: Overbought
  - StochRSI < 20: Oversold
  - More sensitive than RSI alone
  - K/D crossovers for signals

  Features:
  - Fused RSI + Stochastic computation
  - Lazy rescan for min/max (amortized O(N))

  Args:
    data: Input price Series (typically close prices)
    rsi_period: RSI lookback period (default: 14)
    stoch_period: Stochastic lookback on RSI (default: 14)
    k_period: %K SMA smoothing period (default: 3)
    d_period: %D SMA smoothing period (default: 3)

  Returns:
    StochRSIResult with %K and %D values

  Example:
    >>> prices = pd.Series([...])
    >>> result = stochrsi(prices, rsi_period=14, stoch_period=14)

  Configuration:
      rsi_period (int)
      stoch_period (int)
      k_period (int)
      d_period (int)
  """

  rsi_period: int
  stoch_period: int
  k_period: int
  d_period: int
  def __init__(
    self,
    *,
    rsi_period: int = ...,
    stoch_period: int = ...,
    k_period: int = ...,
    d_period: int = ...,
  ) -> None: ...
  """Initialize configuration for stochrsi.

    Configuration:
        rsi_period (int)
        stoch_period (int)
        k_period (int)
        d_period (int)
    """

  @override
  def make(self) -> _stochrsi_Bound: ...

class stochrsi:
  Type = _stochrsi_Bound
  Config = _stochrsi_Config
  ConfigDict = _stochrsi_ConfigDict
  rsi_period: ClassVar[int]
  stoch_period: ClassVar[int]
  k_period: ClassVar[int]
  d_period: ClassVar[int]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NotEmpty],
    rsi_period: int = ...,
    stoch_period: int = ...,
    k_period: int = ...,
    d_period: int = ...,
  ) -> StochRSIResult: ...
