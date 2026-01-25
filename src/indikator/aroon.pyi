"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import AROONOSCResult, AROONResult

class _aroon_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> AROONResult: ...

class _aroon_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for aroon.

  Configuration:
      period (int)
  """

  period: int

class _aroon_Config(_NCMakeableModel[_aroon_Bound]):
  """Configuration class for aroon.

  Calculate AROON indicator.

    AROON measures trend strength by tracking how many periods since
    the highest high and lowest low occurred. It helps identify trend
    changes and strength.

    Formulas:
    Aroon Up = 100 * (period - periods_since_high) / period
    Aroon Down = 100 * (period - periods_since_low) / period
    Aroon Oscillator = Aroon Up - Aroon Down

    Interpretation:
    - Aroon Up > 70 and Aroon Down < 30: Strong uptrend
    - Aroon Down > 70 and Aroon Up < 30: Strong downtrend
    - Both low: Consolidation
    - Crossovers: Trend change signals
    - Aroon Osc > 0: Bullish, < 0: Bearish

    Features:
    - Numba-optimized for performance
    - Returns Up, Down, and Oscillator values
    - Range: 0 to 100 for Up/Down, -100 to +100 for Oscillator

    Args:
      high: High prices Series
      low: Low prices Series
      period: Lookback period (default: 25)

    Returns:
      DataFrame with columns: aroon_up, aroon_down, aroon_osc

    Raises:
      ValueError: If data contains NaN/Inf

    Example:
      >>> import pandas as pd
  from indikator.utils import to_numpy
      >>> high = pd.Series([105, 106, 104, 108, 107, 109, 108, 110] * 5)
      >>> low = pd.Series([100, 101, 99, 103, 102, 104, 103, 105] * 5)
      >>> result = aroon(high, low, period=5)
      >>> # Returns DataFrame with aroon_up, aroon_down, aroon_osc

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for aroon.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _aroon_Bound: ...

class aroon:
  Type = _aroon_Bound
  Config = _aroon_Config
  ConfigDict = _aroon_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    period: int = ...,
  ) -> AROONResult: ...

class _aroonosc_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> AROONOSCResult: ...

class _aroonosc_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for aroonosc.

  Configuration:
      period (int)
  """

  period: int

class _aroonosc_Config(_NCMakeableModel[_aroonosc_Bound]):
  """Configuration class for aroonosc.

  Calculate Aroon Oscillator.

  AROONOSC = Aroon Up - Aroon Down

  Range: -100 to +100
  - Positive: Bullish (uptrend)
  - Negative: Bearish (downtrend)

  Args:
    high: High prices Series
    low: Low prices Series
    period: Lookback period (default: 25)

  Returns:
    AROONOSCResult

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for aroonosc.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _aroonosc_Bound: ...

class aroonosc:
  Type = _aroonosc_Bound
  Config = _aroonosc_Config
  ConfigDict = _aroonosc_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    period: int = ...,
  ) -> AROONOSCResult: ...
