"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import Columns, Datetime, Finite, Index, NotEmpty, Validated
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

from indikator._results import ATRIntradayResult, ATRResult, TRANGEResult

class _atr_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def period(self) -> int: ...
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> ATRResult: ...

class _atr_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for atr.

  Configuration:
      period (int)
  """

  period: int

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
  ATR measures market volatility. It decomposes the entire range of an asset
  for that period.

  Formula:
  TR = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
  ATR = EMA(TR, period)  (Wilder's Smoothing)

  Interpretation:
  - High ATR: High volatility (big moves)
  - Low ATR: Low volatility (consolidation)
  - Rising ATR: Volatility increasing (possible trend reversal/breakout)
  - ATR is NOT directional

  Features:
  - Numba-optimized for performance
  - Standard 14-period default (Wilder's original)

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    period: Lookback period (default: 14)

  Returns:
    ATRResult(index, atr)

  Configuration:
      period (int)
  """

  period: int
  def __init__(self, *, period: int = ...) -> None: ...
  """Initialize configuration for atr.

    Configuration:
        period (int)
    """

  @override
  def make(self) -> _atr_Bound: ...

class atr:
  Type = _atr_Bound
  Config = _atr_Config
  ConfigDict = _atr_ConfigDict
  period: ClassVar[int]
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
    period: int = ...,
  ) -> ATRResult: ...

class _atr_intraday_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def min_samples(self) -> int: ...
  def __call__(
    self,
    data: Validated[
      pd.DataFrame, Columns(["high", "low", "close"]), Finite, Index(Datetime), NotEmpty
    ],
    lookback_days: int | None = ...,
  ) -> ATRIntradayResult: ...

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
      Series with time-of-day adjusted ATR values (NaN until min_samples met per time slot)

    Raises:
      ValueError: If required columns missing or index is not DatetimeIndex

    Example:
      >>> import pandas as pd
  from indikator.utils import to_numpy
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
      pd.DataFrame, Columns(["high", "low", "close"]), Finite, Index(Datetime), NotEmpty
    ],
    lookback_days: int | None = ...,
    min_samples: int = ...,
  ) -> ATRIntradayResult: ...

class _trange_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> TRANGEResult: ...

class _trange_ConfigDict(TypedDict, total=False):
  pass

class _trange_Config(_NCMakeableModel[_trange_Bound]):
  """Configuration class for trange.

  Calculate True Range (TRANGE).

  The True Range is the greatest of:
  - Current high - current low
  - |Current high - previous close|
  - |Current low - previous close|

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.

  Returns:
    TRangeResult(index, trange)
  """

  pass

class trange:
  Type = _trange_Bound
  Config = _trange_Config
  ConfigDict = _trange_ConfigDict
  def __new__(
    cls,
    high: Validated[pd.Series[float], Finite, NotEmpty],
    low: Validated[pd.Series[float], Finite, NotEmpty],
    close: Validated[pd.Series[float], Finite, NotEmpty],
  ) -> TRANGEResult: ...
