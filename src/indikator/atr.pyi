"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd
from validated import Datetime, HasColumns, Index, NonEmpty, Validated

class _atr_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def window(self) -> int: ...
  def __call__(
    self, data: Validated[pd.DataFrame, HasColumns(["high", "low", "close"]), NonEmpty]
  ) -> pd.DataFrame: ...

class _atr_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for atr.

  Configuration:
      window (int)
  """

  window: int

class _atr_Config(_NCMakeableModel[_atr_Bound]):
  """Configuration class for atr.

  Calculate Average True Range (ATR).

  ATR measures market volatility by calculating the average of true ranges
  over a specified period. Uses Wilder's smoothing method for a smoother output.

  The True Range is the greatest of:
  - Current high - current low
  - |Current high - previous close|
  - |Current low - previous close|

  ATR is essential for:
  - Position sizing (risk-adjusted position sizes)
  - Stop-loss placement (volatility-based stops)
  - Identifying breakout potential (volatility expansion)
  - Trend strength assessment (higher ATR = stronger trend)

  Uses Wilder's smoothing (similar to EMA):
  ATR[i] = (ATR[i-1] * (window-1) + TR[i]) / window

  Features:
  - Numba-optimized for performance
  - Handles edge cases (first bar, insufficient data)
  - Returns both ATR and True Range values
  - Standard 14-period default (Wilder's original)

  Args:
    data: OHLCV DataFrame with 'high', 'low', 'close' columns
    window: Smoothing period (default: 14, Wilder's original)

  Returns:
    DataFrame with 'atr' and 'true_range' columns added

  Raises:
    ValueError: If required columns are missing or data contains NaN/Inf

  Example:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'high': [102, 104, 103, 106, 108],
    ...     'low': [100, 101, 100, 103, 105],
    ...     'close': [101, 103, 102, 105, 107]
    ... })
    >>> result = atr(data, window=3)
    >>> # Returns DataFrame with 'atr' and 'true_range' columns

  Configuration:
      window (int)
  """

  window: int
  def __init__(self, *, window: int = ...) -> None: ...
  """Initialize configuration for atr.

    Configuration:
        window (int)
    """

  @override
  def make(self) -> _atr_Bound: ...

class atr:
  Type = _atr_Bound
  Config = _atr_Config
  ConfigDict = _atr_ConfigDict
  window: ClassVar[int]
  def __new__(
    cls,
    data: Validated[pd.DataFrame, HasColumns(["high", "low", "close"]), NonEmpty],
    window: int = ...,
  ) -> pd.DataFrame: ...

class _atr_intraday_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def min_samples(self) -> int: ...
  def __call__(
    self,
    data: Validated[
      pd.DataFrame, HasColumns(["high", "low", "close"]), Index(Datetime), NonEmpty
    ],
    lookback_days: int | None = ...,
  ) -> pd.DataFrame: ...

class _atr_intraday_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for atr_intraday.

  Configuration:
      min_samples (int)
  """

  min_samples: int

class _atr_intraday_Config(_NCMakeableModel[_atr_intraday_Bound]):
  """Configuration class for atr_intraday.

  Calculate time-of-day adjusted ATR (intraday volatility).

  Compares current volatility to the historical average volatility for that
  specific time of day. This accounts for intraday volatility patterns:
  - Market open (9:30-10:00) typically has high volatility
  - Lunch (12:00-13:00) typically has low volatility
  - Market close (15:30-16:00) typically has high volatility

  Regular ATR might show "high volatility" during market open even when it's
  normal for that time. Intraday ATR correctly identifies "high for this time
  of day".

  Features:
  - Accounts for natural intraday volatility patterns
  - Configurable lookback period (None = use all history)
  - Requires minimum samples per time slot for reliability
  - Returns both intraday ATR and True Range

  Args:
    data: OHLCV DataFrame with DatetimeIndex and 'high', 'low', 'close' columns
    lookback_days: Number of days to look back (None = use all history)
    min_samples: Minimum historical samples required per time slot

  Returns:
    DataFrame with 'atr_intraday' and 'true_range' columns added

  Raises:
    ValueError: If required columns missing or index is not DatetimeIndex

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01 09:30', periods=100, freq='5min')
    >>> data = pd.DataFrame({
    ...     'high': [102]*100,
    ...     'low': [100]*100,
    ...     'close': [101]*100
    ... }, index=dates)
    >>> result = atr_intraday(data)
    >>> # Returns DataFrame with time-of-day adjusted ATR

  Configuration:
      min_samples (int)
  """

  min_samples: int
  def __init__(self, *, min_samples: int = ...) -> None: ...
  """Initialize configuration for atr_intraday.

    Configuration:
        min_samples (int)
    """

  @override
  def make(self) -> _atr_intraday_Bound: ...

class atr_intraday:
  Type = _atr_intraday_Bound
  Config = _atr_intraday_Config
  ConfigDict = _atr_intraday_ConfigDict
  min_samples: ClassVar[int]
  def __new__(
    cls,
    data: Validated[
      pd.DataFrame, HasColumns(["high", "low", "close"]), Index(Datetime), NonEmpty
    ],
    lookback_days: int | None = ...,
    min_samples: int = ...,
  ) -> pd.DataFrame: ...
