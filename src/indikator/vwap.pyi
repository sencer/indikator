"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar
from typing import Protocol
from typing import TypedDict
from typing import override

from nonfig import MakeableModel as _NCMakeableModel

from datawarden import Columns, Finite, NotEmpty, Validated
import pandas as pd
from indikator._results import VWAPAnchoredResult, VWAPResult

class _vwap_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def anchor(self) -> str | pd.Timedelta | int: ...
  def __call__(
    self,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    volume: Validated[pd.Series, Finite, NotEmpty],
  ) -> VWAPResult: ...

class _vwap_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for vwap.

  Configuration:
      anchor (str | pd.Timedelta | int.Config | str | pd.Timedelta | int.ConfigDict)
  """

  anchor: str | pd.Timedelta | int.Config | str | pd.Timedelta | int.ConfigDict

class _vwap_Config(_NCMakeableModel[_vwap_Bound]):
  """Configuration class for vwap.

  Calculate Volume Weighted Average Price (VWAP).

  VWAP is a trading benchmark that gives the average price a security has
  traded at throughout the day, based on both volume and price.

  Formula:
  Typical Price = (High + Low + Close) / 3
  VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)

  Reset:
  The cumulative sums reset based on the anchor period (e.g., daily 'D').

  Interpretation:
  - Price > VWAP: Bullish sentiment (buyers are in control)
  - Price < VWAP: Bearish sentiment (sellers are in control)
  - VWAP acts as dynamic support/resistance
  - Institutions use VWAP to execute large orders without moving market

  Features:
  - Numba-optimized for performance
  - Flexible anchoring (Time-based or Bar-count based)
  - Standard 'D' (daily) anchor default

  Args:
    high: High prices Series.
    low: Low prices Series.
    close: Close prices Series.
    volume: Volume Series.
    anchor: Reset anchor (e.g. 'D', 'W', '1h') or int (bars). Default 'D'.

  Returns:
    VWAPResult(index, vwap)

  Configuration:
      anchor (str | pd.Timedelta | int.Config | str | pd.Timedelta | int.ConfigDict)
  """

  anchor: str | pd.Timedelta | int.Config | str | pd.Timedelta | int.ConfigDict
  def __init__(
    self,
    *,
    anchor: str | pd.Timedelta | int.Config | str | pd.Timedelta | int.ConfigDict = ...,
  ) -> None: ...
  """Initialize configuration for vwap.

    Configuration:
        anchor (str | pd.Timedelta | int.Config | str | pd.Timedelta | int.ConfigDict)
    """

  @override
  def make(self) -> _vwap_Bound: ...

class vwap:
  Type = _vwap_Bound
  Config = _vwap_Config
  ConfigDict = _vwap_ConfigDict
  anchor: ClassVar[str | pd.Timedelta | int]
  def __new__(
    cls,
    high: Validated[pd.Series, Finite, NotEmpty],
    low: Validated[pd.Series, Finite, NotEmpty],
    close: Validated[pd.Series, Finite, NotEmpty],
    volume: Validated[pd.Series, Finite, NotEmpty],
    anchor: str | pd.Timedelta | int = ...,
  ) -> VWAPResult: ...

class _vwap_anchored_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    data: Validated[
      pd.DataFrame, Columns(["high", "low", "close", "volume"]), Finite, NotEmpty
    ],
    anchor_index: int | None = ...,
    anchor_datetime: pd.Timestamp | str | None = ...,
  ) -> VWAPAnchoredResult: ...

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
      pd.DataFrame, Columns(["high", "low", "close", "volume"]), Finite, NotEmpty
    ],
    anchor_index: int | None = ...,
    anchor_datetime: pd.Timestamp | str | None = ...,
  ) -> VWAPAnchoredResult: ...
