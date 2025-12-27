"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Datetime, Finite, Index, NonEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _zscore_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def window(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(self, data: Validated[pd.Series, Finite, NonEmpty]) -> pd.Series: ...

class _zscore_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for zscore.

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float

class _zscore_Config(_NCMakeableModel[_zscore_Bound]):
  """Configuration class for zscore.

  Calculate Z-Score (Standard Score) over a rolling window.

  Z-Score measures how many standard deviations a data point is from the mean.
  - Z > 2.0: Significantly above average (potential overbought/outlier)
  - Z < -2.0: Significantly below average (potential oversold/outlier)
  - Z ~ 0: Near average

  Args:
    data: Series of values (e.g., close prices, volume, etc.)
    window: Rolling window size for mean and std dev calculation
    epsilon: Small value to prevent division by zero.

  Returns:
    Series with Z-Score values

  Raises:
    ValueError: If validation fails

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float
  def __init__(self, *, window: int = ..., epsilon: float = ...) -> None: ...
  """Initialize configuration for zscore.

    Configuration:
        window (int)
        epsilon (float)
    """

  @override
  def make(self) -> _zscore_Bound: ...

class zscore:
  Type = _zscore_Bound
  Config = _zscore_Config
  ConfigDict = _zscore_ConfigDict
  window: ClassVar[int]
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, NonEmpty],
    window: int = ...,
    epsilon: float = ...,
  ) -> pd.Series: ...

class _zscore_intraday_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def min_samples(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(
    self,
    data: Validated[pd.Series, Finite, Index(Datetime), NonEmpty],
    lookback_days: int | None = ...,
  ) -> pd.Series: ...

class _zscore_intraday_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for zscore_intraday.

  Configuration:
      min_samples (int)
      epsilon (float)
  """

  min_samples: int
  epsilon: float

class _zscore_intraday_Config(_NCMakeableModel[_zscore_intraday_Bound]):
  """Configuration class for zscore_intraday.

  Calculate time-of-day adjusted Z-Score.

  Compares current value to the historical mean and std dev for that specific
  time of day (e.g., 10:30 AM today vs. all previous 10:30 AM bars).

  This accounts for intraday patterns:
  - Price tends to be volatile at market open
  - Volume tends to be high at open/close, low at lunch
  - Spread/volatility patterns vary by time of day

  Regular Z-score might show "high volatility" during market open even when
  it's normal for that time. Intraday Z-score correctly identifies "high for
  this time of day".

  Features:
  - Accounts for natural intraday patterns
  - Works with any column (price, volume, spread, etc.)
  - Configurable lookback period
  - Requires minimum samples per time slot for reliability

  Args:
    data: Series with DatetimeIndex
    lookback_days: Number of days to look back (None = use all history)
    min_samples: Minimum observations required before calculating aggregate (NaN until met)
    epsilon: Small value to prevent division by zero

  Returns:
    Series with intraday Z-Score values

  Raises:
    ValueError: If index is not DatetimeIndex

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01 09:30', periods=10, freq='1D')
    >>> data = pd.Series([100]*9 + [150], index=dates)
    >>> # Same time each day, last day has spike
    >>> result = zscore_intraday(data)
    >>> # Will show high z-score on last day

  Configuration:
      min_samples (int)
      epsilon (float)
  """

  min_samples: int
  epsilon: float
  def __init__(self, *, min_samples: int = ..., epsilon: float = ...) -> None: ...
  """Initialize configuration for zscore_intraday.

    Configuration:
        min_samples (int)
        epsilon (float)
    """

  @override
  def make(self) -> _zscore_intraday_Bound: ...

class zscore_intraday:
  Type = _zscore_intraday_Bound
  Config = _zscore_intraday_Config
  ConfigDict = _zscore_intraday_ConfigDict
  min_samples: ClassVar[int]
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, Finite, Index(Datetime), NonEmpty],
    lookback_days: int | None = ...,
    min_samples: int = ...,
    epsilon: float = ...,
  ) -> pd.Series: ...
