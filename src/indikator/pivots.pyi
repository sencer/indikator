"""Auto-generated type stubs for @configurable decorators.

Do not edit manually - regenerate with: nonfig-stubgen <path>
"""

from typing import Literal, Protocol, TypedDict

from datawarden import (
  Datetime,
  Finite,
  Ge as GeValidator,
  HasColumns,
  Index,
  NonEmpty,
  Validated,
)
from nonfig import MakeableModel as _NCMakeableModel
import pandas as pd

class _pivot_points_Bound(Protocol):
  """Bound function with hyperparameters as attributes."""
  def __call__(
    self,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close"]),
      Index(Datetime),
      GeValidator("high", "low"),
      Finite,
      NonEmpty,
    ],
    method: Literal["standard", "fibonacci", "woodie", "camarilla"] = ...,
    period: Literal["D", "W", "ME"] = ...,
  ) -> pd.DataFrame: ...

class _pivot_points_ConfigDict(TypedDict, total=False):
  pass

class _pivot_points_Config(_NCMakeableModel[_pivot_points_Bound]):
  """Configuration class for pivot_points.

  Calculate Pivot Points for support and resistance levels.

  Pivot Points are price levels calculated using the previous period's high,
  low, and close. They're used to identify potential support/resistance and
  intraday trading ranges.

  methods:
  1. Standard (Floor Trader's Pivots):
     - PP = (High + Low + Close) / 3
     - R1 = 2xPP - Low, S1 = 2xPP - High
     - R2 = PP + (High - Low), S2 = PP - (High - Low)
     - R3 = High + 2x(PP - Low), S3 = Low - 2x(High - PP)

  2. Fibonacci:
     - PP = (High + Low + Close) / 3
     - R1 = PP + 0.382x(High - Low), S1 = PP - 0.382x(High - Low)
     - R2 = PP + 0.618x(High - Low), S2 = PP - 0.618x(High - Low)
     - R3 = PP + 1.000x(High - Low), S3 = PP - 1.000x(High - Low)

  3. Woodie:
     - PP = (High + Low + 2xClose) / 4
     - R1 = 2xPP - Low, S1 = 2xPP - High
     - R2 = PP + (High - Low), S2 = PP - (High - Low)

  4. Camarilla:
     - PP = (High + Low + Close) / 3
     - R1 = Close + 1.1/12x(High - Low)
     - R2 = Close + 1.1/6x(High - Low)
     - R3 = Close + 1.1/4x(High - Low)
     - R4 = Close + 1.1/2x(High - Low)
     - (Similar formula for S1-S4 with subtraction)

  Theory:
  - Pivot point is the primary support/resistance level
  - R1/R2/R3 are resistance levels above pivot
  - S1/S2/S3 are support levels below pivot
  - Price tends to range between S1 and R1 (80% of the time)
  - Breakouts beyond R1/S1 signal strong moves

  Common strategies:
  - Range trading: Buy at support, sell at resistance
  - Breakout: Enter when price breaks R1/S1 with volume
  - Pivot bounce: Enter on pullback to pivot point
  - Multiple timeframe: Use daily pivots for intraday trading

  Features:
  - Multiple calculation methods (standard, fibonacci, woodie, camarilla)
  - Configurable period (daily, weekly, monthly)
  - Returns all pivot levels
  - Forward-fills pivots for current period

  Args:
    data: OHLC DataFrame with DatetimeIndex
    method: Calculation method (default: "standard")
    period: Period for pivot calculation ('D'=daily, 'W'=weekly, 'ME'=month-end)

  Returns:
    DataFrame with pivot point columns (pp, r1, r2, r3, s1, s2, s3, etc.)

  Raises:
    ValueError: If required columns missing or index not DatetimeIndex

  Example:
    >>> import pandas as pd
    >>> dates = pd.date_range('2024-01-01', periods=20, freq='D')
    >>> data = pd.DataFrame({
    ...     'high': [102, 104, 103, 106, 108, 107, 109, 108, 110, 112] * 2,
    ...     'low': [100, 101, 100, 103, 105, 104, 106, 105, 107, 109] * 2,
    ...     'close': [101, 103, 102, 105, 107, 106, 108, 107, 109, 111] * 2
    ... }, index=dates)
    >>> result = pivot_points(data, method="standard", period="D")
    >>> # Returns DataFrame with PP, R1-R3, S1-R3 levels
  """

  pass

class pivot_points:
  Type = _pivot_points_Bound
  Config = _pivot_points_Config
  ConfigDict = _pivot_points_ConfigDict
  def __new__(
    cls,
    data: Validated[
      pd.DataFrame,
      HasColumns(["high", "low", "close"]),
      Index(Datetime),
      GeValidator("high", "low"),
      Finite,
      NonEmpty,
    ],
    method: Literal["standard", "fibonacci", "woodie", "camarilla"] = ...,
    period: Literal["D", "W", "ME"] = ...,
  ) -> pd.DataFrame: ...
