"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import ClassVar, Protocol, TypedDict, override

from datawarden import (
  Datetime,
  Finite,
  Ge as GeValidator,
  HasColumns,
  Index,
  NotEmpty,
  Validated,
)
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _opening_range_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  @property
  def minutes(self) -> int: ...
  def __call__(
    self,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close"]),
      Index(Datetime),
      Finite,
      GeValidator("high", "low"),
      NotEmpty,
    ],
    session_start: str = ...,
  ) -> pd.DataFrame: ...

class _opening_range_ConfigDict(TypedDict, total=False):
  """Configuration dictionary for opening_range.

  Configuration:
      minutes (int)
  """

  minutes: int

class _opening_range_Config(_NCMakeableModel[_opening_range_Bound]):
  """Configuration class for opening_range.

  Calculate Opening Range (OR) for intraday trading.

  The Opening Range is the high and low of the first N minutes of the
  trading session. Breakouts above/below this range are considered
  significant trading signals.

  Calculations:
  - OR High = highest high during first N minutes of session
  - OR Low = lowest low during first N minutes of session
  - OR Mid = (OR High + OR Low) / 2
  - OR Range = OR High - OR Low
  - Breakout Status: 1 (above OR), 0 (inside OR), -1 (below OR)

  Theory:
  - Opening range represents initial supply/demand equilibrium
  - Breakouts show strong directional conviction
  - Failed breakouts can signal reversals
  - Wider OR = higher volatility day expected

  Common strategies:
  - Classic ORB: Buy breakout above OR high, sell breakout below OR low
  - First pullback: Wait for breakout, then buy first pullback to OR
  - Failed break: Fade false breakouts (enter opposite direction)
  - Range contraction: Small OR after wide OR = potential big move

  Features:
  - Configurable opening range period (default: 30 minutes)
  - Configurable session start time
  - Returns OR levels for each bar
  - Breakout status indicator

  Args:
    data: OHLC DataFrame with DatetimeIndex
    minutes: Number of minutes for opening range (default: 30)
    session_start: Session start time in "HH:MM" format (default: "09:30")

  Returns:
    DataFrame with 'or_high', 'or_low', 'or_mid', 'or_range', 'or_breakout' columns

  Raises:
    ValueError: If required columns missing or index not DatetimeIndex

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01 09:30', periods=100, freq='5min')
    >>> data = pd.DataFrame({
    ...     'high': np.random.uniform(100, 110, 100),
    ...     'low': np.random.uniform(95, 105, 100),
    ...     'close': np.random.uniform(97, 107, 100)
    ... }, index=dates)
    >>> result = opening_range(data, minutes=30)
    >>> # Returns DataFrame with OR levels

  Configuration:
      minutes (int)
  """

  minutes: int
  def __init__(self, *, minutes: int = ...) -> None: ...
  """Initialize configuration for opening_range.

    Configuration:
        minutes (int)
    """

  @override
  def make(self) -> _opening_range_Bound: ...

class opening_range:
  Type = _opening_range_Bound
  Config = _opening_range_Config
  ConfigDict = _opening_range_ConfigDict
  minutes: ClassVar[int]
  def __new__(
    cls,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close"]),
      Index(Datetime),
      Finite,
      GeValidator("high", "low"),
      NotEmpty,
    ],
    session_start: str = ...,
    minutes: int = ...,
  ) -> pd.DataFrame: ...
