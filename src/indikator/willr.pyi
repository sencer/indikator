"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Finite, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _willr_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
  ) -> pd.Series: ...

class _willr_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for willr.

  Configuration:
      period (int)
  """

  period: int

class _willr_Config(_NCMakeableModel[_willr_Bound]):
  """Configuration class for willr.

  Calculate Williams %R.

  Williams %R is a momentum indicator that measures overbought/oversold levels.
  It's similar to the Stochastic Oscillator but inverted and on a negative scale.

  Formula:
  %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)

  Range: -100 to 0

  Interpretation:
  - %R between -20 and 0: Overbought
  - %R between -100 and -80: Oversold
  - %R crossing -50: Momentum shift

  Common strategies:
  - Buy when %R moves from below -80 to above -80
  - Sell when %R moves from above -20 to below -20
  - Divergence: Price making new high but %R doesn't = bearish

  Features:
  - Numba-optimized for performance
  - Standard 14-period default
  - Works with OHLC data

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    Series with Williams %R values (-100 to 0 range)

  Raises:
    ValueError: If data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> high = pd.Series([105, 107, 106, 108, 110])
    >>> low = pd.Series([100, 102, 101, 103, 105])
    >>> close = pd.Series([102, 105, 104, 106, 108])
    >>> result = willr(high, low, close)

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
  ) -> pd.Series: ...
