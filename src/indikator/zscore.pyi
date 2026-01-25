"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Datetime, Finite, Index, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ZScoreIntradayResult, ZScoreResult

class _zscore_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self, data: Validated[pd.Series[float], Finite, NotEmpty]
  ) -> ZScoreResult: ...

class _zscore_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for zscore.

  Configuration:
      period (int)
  """

  period: int

class _zscore_Config(_NCMakeableModel[_zscore_Bound]):
  """Configuration class for zscore.

  Calculate Z-Score (Standard Score).

  Z-Score measures how many standard deviations a price is from the mean.
  It is a mean-reversion indicator.

  Formula:
  Z = (Price - One_Year_Mean) / Standard_Deviation

  Interpretation:
  - Z > 2: Price is 2 standard deviations above mean (statistically rare/expensive)
  - Z < -2: Price is 2 standard deviations below mean (cheap)
  - Extreme values indicate potential reversal (mean reversion)
  - Z = 0: Price is exactly at the mean

  Features:
  - Numba-optimized for performance
  - Rolling window calculation
  - Standard 20-period default (similar to Bollinger Bands)

  Args:
    data: Input Series.
    period: Lookback period (default: 20)

  Returns:
    ZScoreResult(index, zscore)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for zscore.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _zscore_Bound: ...

class zscore:
  Type = _zscore_Bound
  Config = _zscore_Config
  ConfigDict = _zscore_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls, data: Validated[pd.Series[float], Finite, NotEmpty], period: int = ...
  ) -> ZScoreResult: ...

class _zscore_intraday_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def min_samples(self) -> int: ...
  @property
  def epsilon(self) -> float: ...
  def __call__(
    self,
    data: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
    lookback_days: int | None = ...,
  ) -> ZScoreIntradayResult: ...

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
  from indikator.utils import to_numpy
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
    data: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
    lookback_days: int | None = ...,
    min_samples: int = ...,
    epsilon: float = ...,
  ) -> ZScoreIntradayResult: ...
