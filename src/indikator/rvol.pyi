"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd
from validated import Datetime, Index, NonEmpty, NonNegative, Validated

class _rvol_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def window(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(
    self, data: Validated[pd.Series, NonNegative, NonEmpty]
  ) -> pd.Series: ...

class _rvol_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for rvol.

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float

class _rvol_Config(_NCMakeableModel[_rvol_Bound]):
  """Configuration class for rvol.

  Calculate Relative Volume (RVOL).

  Measures current volume relative to average volume over a lookback window.
  RVOL is a key indicator for identifying unusual trading activity:
  - RVOL > 1.0: Above-average volume (more activity than usual)
  - RVOL = 1.0: Average volume (normal activity)
  - RVOL < 1.0: Below-average volume (less activity than usual)

  Values significantly above 1.0 (e.g., > 2.0 or > 3.0) indicate
  exceptional volume that may signal:
  - Breakouts or breakdowns
  - News events or earnings
  - Institutional accumulation/distribution
  - Potential trend changes

  Features:
  - Division by zero protection with epsilon
  - Handles insufficient data gracefully (returns 1.0 as neutral)
  - Simple moving average for stable baseline

  Args:
    data: Volume Series
    window: Rolling window size for average volume calculation
    epsilon: Small value to prevent division by zero

  Returns:
    Series with RVOL values

  Raises:
    ValueError: If validation fails

  Example:
    >>> import pandas as pd
    >>> data = pd.Series([1000, 1000, 1000, 2000, 3000])
    >>> result = rvol(data, window=3)
    >>> # RVOL will be ~1.0 for first bars, then spike to 2.0, 3.0

  Configuration:
      window (int)
      epsilon (float)
  """

  window: int
  epsilon: float
  def __init__(self, *, window: int = ..., epsilon: float = ...) -> None: ...
  """Initialize configuration for rvol.

    Configuration:
        window (int)
        epsilon (float)
    """

  @override
  def make(self) -> _rvol_Bound: ...

class rvol:
  Type = _rvol_Bound
  Config = _rvol_Config
  ConfigDict = _rvol_ConfigDict
  window: ClassVar[int]
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, NonNegative, NonEmpty],
    window: int = ...,
    epsilon: float = ...,
  ) -> pd.Series: ...

class _rvol_intraday_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def min_samples(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(
    self,
    data: Validated[pd.Series, NonNegative, Index(Datetime), NonEmpty],
    lookback_days: int | None = ...,
  ) -> pd.Series: ...

class _rvol_intraday_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for rvol_intraday.

  Configuration:
      min_samples (int)
      epsilon (float)
  """

  min_samples: int
  epsilon: float

class _rvol_intraday_Config(_NCMakeableModel[_rvol_intraday_Bound]):
  """Configuration class for rvol_intraday.

  Calculate intraday RVOL based on time-of-day historical averages.

  Compares current volume to the historical average volume for that specific
  time of day (e.g., 10:30 AM today vs. all previous 10:30 AM bars).

  This captures intraday seasonality patterns:
  - Market open (9:30-10:00) typically has high volume
  - Lunch (12:00-13:00) typically has low volume
  - Market close (15:30-16:00) typically has high volume

  Regular RVOL might show "high" during market open even when it's normal
  for that time. Intraday RVOL correctly identifies "high for this time of day".

  Features:
  - Accounts for natural intraday volume patterns
  - Configurable lookback period (None = use all history)
  - Requires minimum samples per time slot for reliability
  - Division by zero protection

  Args:
    data: Series (e.g., volume) with DatetimeIndex
    lookback_days: Number of days to look back (None = use all history)
    min_samples: Minimum observations required before calculating aggregate (NaN until met)
    epsilon: Small value to prevent division by zero

  Returns:
    Series with intraday RVOL values

  Raises:
    ValueError: If index is not DatetimeIndex

  Configuration:
      min_samples (int)
      epsilon (float)
  """

  min_samples: int
  epsilon: float
  def __init__(self, *, min_samples: int = ..., epsilon: float = ...) -> None: ...
  """Initialize configuration for rvol_intraday.

    Configuration:
        min_samples (int)
        epsilon (float)
    """

  @override
  def make(self) -> _rvol_intraday_Bound: ...

class rvol_intraday:
  Type = _rvol_intraday_Bound
  Config = _rvol_intraday_Config
  ConfigDict = _rvol_intraday_ConfigDict
  min_samples: ClassVar[int]
  epsilon: ClassVar[float]
  def __new__(
    cls,
    data: Validated[pd.Series, NonNegative, Index(Datetime), NonEmpty],
    lookback_days: int | None = ...,
    min_samples: int = ...,
    epsilon: float = ...,
  ) -> pd.Series: ...
