"""Opening Range Breakout indicator module.

This module provides opening range calculation, a popular day trading
strategy that identifies the high/low range of the first N minutes.
"""

from typing import Literal, cast

from hipr import Ge, Hyper, configurable
import numpy as np
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
def opening_range(
  data: Validated[
    pd.DataFrame,
    HasColumns[Literal["high", "low", "close"]],
    Index[Datetime],
    GeValidator[Literal["high", "low"]],
  ],
  minutes: Hyper[int, Ge[1]] = 30,
  session_start: str = "09:30",
) -> pd.DataFrame:
  """Calculate Opening Range (OR) for intraday trading.

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
  """
  # Create session date column for grouping
  dt_index = cast("pd.DatetimeIndex", data.index)
  session_date = dt_index.normalize()

  # Create time-based filter for opening range period
  time_of_day = dt_index.time
  or_start_time = pd.Timestamp(f"1900-01-01 {session_start}").time()
  or_end_time = (
    pd.Timestamp(f"1900-01-01 {session_start}") + pd.Timedelta(minutes=minutes)
  ).time()

  # Vectorized: Identify bars within opening range period
  is_or_period = (time_of_day >= or_start_time) & (time_of_day < or_end_time)

  # Calculate OR high/low per session using groupby (vectorized)
  df_work = data.copy()
  df_work["_session"] = session_date
  df_work["_is_or"] = is_or_period

  # Get OR stats only from bars within the opening range
  or_data = df_work[df_work["_is_or"]]
  or_stats = or_data.groupby("_session").agg(  # pyright: ignore[reportUnknownMemberType]
    _or_high=("high", "max"),
    _or_low=("low", "min"),
  )

  # Map OR stats back to original data by session (preserves index order)
  df_work["or_high"] = df_work["_session"].map(or_stats["_or_high"])
  df_work["or_low"] = df_work["_session"].map(or_stats["_or_low"])
  df_work["or_mid"] = (df_work["or_high"] + df_work["or_low"]) / 2.0
  df_work["or_range"] = df_work["or_high"] - df_work["or_low"]

  # Calculate breakout status (vectorized)
  df_work["or_breakout"] = np.select(
    [
      df_work["close"] > df_work["or_high"],
      df_work["close"] < df_work["or_low"],
    ],
    [1, -1],
    default=0,
  ).astype(np.int8)

  # Create result dataframe with only the columns we want
  data_copy = data.copy()
  data_copy["or_high"] = df_work["or_high"].values
  data_copy["or_low"] = df_work["or_low"].values
  data_copy["or_mid"] = df_work["or_mid"].values
  data_copy["or_range"] = df_work["or_range"].values
  data_copy["or_breakout"] = df_work["or_breakout"].values

  return data_copy
