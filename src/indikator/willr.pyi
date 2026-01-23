"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar
from typing import Protocol
from typing import TypedDict
from typing import override

from nonfig import MakeableModel as _NCMakeableModel

from datawarden import Finite, NotEmpty, Validated
import pandas as pd
from indikator._results import WillRResult

class _willr_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> WillRResult: ...

class _willr_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for willr.

  Configuration:
      period (int)
  """

  period: int

class _willr_Config(_NCMakeableModel[_willr_Bound]):
  """Configuration class for willr.

  Calculate Williams %R.

  Williams %R is a momentum indicator that measures overbought and oversold levels.
  It moves between 0 and -100.

  Formula:
  %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

  Interpretation:
  - %R > -20: Overbought
  - %R < -80: Oversold
  - Similar to Stochastic Oscillator Fast %K, but scaled -100 to 0

  Features:
  - Numba-optimized for performance
  - Standard 14 period default

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    WillRResult(index, willr)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for willr.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _willr_Bound: ...

class willr:
  Type = _willr_Bound
  Config = _willr_Config
  ConfigDict = _willr_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    period: int = ...,
  ) -> WillRResult: ...
