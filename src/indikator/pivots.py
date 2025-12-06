"""Pivot Points indicator module.

This module provides pivot point calculation, a popular indicator for
identifying potential support and resistance levels based on previous periods.
"""

from typing import Literal

from hipr import configurable
import pandas as pd
from pdval import (
  Datetime,
  Ge as GeValidator,
  HasColumns,
  Index,
  Validated,
  validated,
)


@configurable
@validated
def pivot_points(  # noqa: PLR0915, pyright: ignore[reportRedeclaration, reportReturnType]
  data: Validated[
    pd.DataFrame,
    HasColumns[Literal["high", "low", "close"]],
    Index[Datetime],
    GeValidator[Literal["high", "low"]],
  ],
  method: Literal["standard", "fibonacci", "woodie", "camarilla"] = "standard",
  period: Literal["D", "W", "ME"] = "D",
) -> pd.DataFrame:
  """Calculate Pivot Points for support and resistance levels.

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

  # Calculate previous period's high, low, close
  if period == "D":
    prev_period = (
      data.resample("D")
      .agg(  # pyright: ignore[reportUnknownMemberType]
        {"high": "max", "low": "min", "close": "last"}
      )
      .shift(1)
    )
  elif period == "W":
    prev_period = (
      data.resample("W")
      .agg(  # pyright: ignore[reportUnknownMemberType]
        {"high": "max", "low": "min", "close": "last"}
      )
      .shift(1)
    )
  elif period == "ME":
    prev_period = (
      data.resample("ME")
      .agg(  # pyright: ignore[reportUnknownMemberType]
        {"high": "max", "low": "min", "close": "last"}
      )
      .shift(1)
    )
  else:
    raise ValueError(f"Invalid period: {period}")

  # Reindex to match original data (forward fill within period)
  prev_high = prev_period["high"].reindex(data.index, method="ffill")
  prev_low = prev_period["low"].reindex(data.index, method="ffill")
  prev_close = prev_period["close"].reindex(data.index, method="ffill")

  # Calculate pivot points based on method
  if method == "standard":
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    r2 = pp + (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s1 = 2 * pp - prev_high
    s2 = pp - (prev_high - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)

    result = pd.DataFrame(
      {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3},
      index=data.index,
    )

  elif method == "fibonacci":
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pp + 0.382 * range_hl
    r2 = pp + 0.618 * range_hl
    r3 = pp + 1.000 * range_hl
    s1 = pp - 0.382 * range_hl
    s2 = pp - 0.618 * range_hl
    s3 = pp - 1.000 * range_hl

    result = pd.DataFrame(
      {"pp": pp, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3},
      index=data.index,
    )

  elif method == "woodie":
    pp = (prev_high + prev_low + 2 * prev_close) / 4
    r1 = 2 * pp - prev_low
    r2 = pp + (prev_high - prev_low)
    s1 = 2 * pp - prev_high
    s2 = pp - (prev_high - prev_low)

    result = pd.DataFrame(
      {"pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2},
      index=data.index,
    )

  elif method == "camarilla":
    pp = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = prev_close + 1.1 / 12 * range_hl
    r2 = prev_close + 1.1 / 6 * range_hl
    r3 = prev_close + 1.1 / 4 * range_hl
    r4 = prev_close + 1.1 / 2 * range_hl
    s1 = prev_close - 1.1 / 12 * range_hl
    s2 = prev_close - 1.1 / 6 * range_hl
    s3 = prev_close - 1.1 / 4 * range_hl
    s4 = prev_close - 1.1 / 2 * range_hl

    result = pd.DataFrame(
      {
        "pp": pp,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
      },
      index=data.index,
    )

  else:
    raise ValueError(f"Invalid method: {method}")

  return result
