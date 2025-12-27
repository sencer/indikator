"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Literal, Protocol, TypedDict

from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd
from validated import (
  Datetime,
  Ge as GeValidator,
  HasColumns,
  Index,
  NonEmpty,
  Validated,
)

class _vwap_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close", "volume"]),
      Index(Datetime),
      GeValidator("high", "low"),
      NonEmpty,
    ],
    session_freq: Literal["D", "W", "ME"] = ...,
  ) -> pd.Series: ...

class _vwap_ConfigDict(TypedDict, total=False):
  pass

class _vwap_Config(_NCMakeableModel[_vwap_Bound]):
  """Configuration class for vwap.

  Calculate Volume-Weighted Average Price (VWAP).

  VWAP is the ratio of cumulative (price * volume) to cumulative volume,
  typically reset at the beginning of each trading session. It represents
  the average price weighted by volume.

  VWAP = Sum(Typical Price * Volume) / Sum(Volume)
  where Typical Price = (High + Low + Close) / 3

  Institutional traders use VWAP as:
  - Execution benchmark (am I getting better/worse than VWAP?)
  - Support/resistance level (price tends to revert to VWAP)
  - Trend indicator (price above VWAP = bullish, below = bearish)
  - Entry/exit signal (crossing VWAP can indicate trend changes)

  Features:
  - Numba-optimized for performance
  - Configurable session period (daily, weekly, monthly)
  - Handles missing volume gracefully
  - Returns both VWAP and typical price

  Args:
    data: OHLCV DataFrame with DatetimeIndex
    session_freq: Session reset frequency ('D'=daily, 'W'=weekly, 'ME'=month-end)

  Returns:
    DataFrame with 'vwap' and 'typical_price' columns added

  Raises:
    ValueError: If required columns missing or index not DatetimeIndex

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01 09:30', periods=10, freq='5min')
    >>> data = pd.DataFrame({
    ...     'high': [102, 104, 103, 106, 108, 107, 109, 108, 110, 112],
    ...     'low': [100, 101, 100, 103, 105, 104, 106, 105, 107, 109],
    ...     'close': [101, 103, 102, 105, 107, 106, 108, 107, 109, 111],
    ...     'volume': [1000]*10
    ... }, index=dates)
    >>> result = vwap(data)
    >>> # Returns DataFrame with VWAP column
  """

  pass

class vwap:
  Type = _vwap_Bound
  Config = _vwap_Config
  ConfigDict = _vwap_ConfigDict
  def __new__(
    cls,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close", "volume"]),
      Index(Datetime),
      GeValidator("high", "low"),
      NonEmpty,
    ],
    session_freq: Literal["D", "W", "ME"] = ...,
  ) -> pd.Series: ...

class _vwap_anchored_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    data: Validated[
      pd.DataFrame, HasColumns(["high", "low", "close", "volume"]), NonEmpty
    ],
    anchor_index: int | None = ...,
    anchor_datetime: pd.Timestamp | str | None = ...,
  ) -> pd.Series: ...

class _vwap_anchored_ConfigDict(TypedDict, total=False):
  pass

class _vwap_anchored_Config(_NCMakeableModel[_vwap_anchored_Bound]):
  """Configuration class for vwap_anchored.

  Calculate Anchored VWAP from a specific point in time.

  Anchored VWAP calculates VWAP starting from a specific bar forward,
  rather than resetting at session boundaries. This is useful for:
  - Anchoring to significant events (earnings, news, pivots)
  - Tracking institutional positioning from specific entry points
  - Measuring average fill price from a particular time
  - Swing trading support/resistance from key levels

  Common anchor points:
  - Earnings announcements
  - Market structure breaks (new high/low)
  - Major news events
  - Session open/close
  - Previous day high/low

  Features:
  - Anchor by index position or datetime
  - No session resets (continuous from anchor)
  - Returns NaN for all bars before anchor
  - Numba-optimized for performance

  Args:
    data: OHLCV DataFrame
    anchor_index: Bar index to start VWAP calculation (0-based)
    anchor_datetime: Datetime to start VWAP (alternative to anchor_index)

  Returns:
    DataFrame with 'vwap_anchored' and 'typical_price' columns added

  Raises:
    ValueError: If neither or both anchor parameters provided, or anchor not found

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
    >>> data = pd.DataFrame({
    ...     'high': [102, 104, 103, 106, 108, 107, 109, 108, 110, 112],
    ...     'low': [100, 101, 100, 103, 105, 104, 106, 105, 107, 109],
    ...     'close': [101, 103, 102, 105, 107, 106, 108, 107, 109, 111],
    ...     'volume': [1000]*10
    ... }, index=dates)
    >>> # Anchor VWAP from bar 3 (representing a breakout)
    >>> result = vwap_anchored(data, anchor_index=3)
    >>> # Or anchor from specific date
    >>> result = vwap_anchored(data, anchor_datetime='2024-01-04')
  """

  pass

class vwap_anchored:
  Type = _vwap_anchored_Bound
  Config = _vwap_anchored_Config
  ConfigDict = _vwap_anchored_ConfigDict
  def __new__(
    cls,
    data: Validated[
      pd.DataFrame, HasColumns(["high", "low", "close", "volume"]), NonEmpty
    ],
    anchor_index: int | None = ...,
    anchor_datetime: pd.Timestamp | str | None = ...,
  ) -> pd.Series: ...
