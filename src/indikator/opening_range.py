"""Opening Range Breakout indicator module.

This module provides opening range calculation, a popular day trading
strategy that identifies the high/low range of the first N minutes.
"""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from typing import Literal

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
def opening_range(  # noqa: PLR0914
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

    # Parse session start time
    start_hour, start_minute = map(int, session_start.split(":"))

    # Create date column for grouping by session
    dates = pd.Series(data.index, index=data.index)
    session_date = dates.dt.normalize()

    # Initialize result arrays
    n = len(data)
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    or_mid = np.full(n, np.nan)
    or_range = np.full(n, np.nan)
    or_breakout = np.zeros(n, dtype=np.int8)

    # Process each session
    for session in session_date.unique():
        # Get bars for this session
        session_mask = session_date == session
        session_data = data[session_mask]

        if len(session_data) == 0:
            continue

        # Find opening range period
        session_start_time = pd.Timestamp(
            year=session.year,
            month=session.month,
            day=session.day,
            hour=start_hour,
            minute=start_minute,
        )
        or_end_time = session_start_time + pd.Timedelta(minutes=minutes)

        # Get bars within opening range
        or_mask = (session_data.index >= session_start_time) & (
            session_data.index < or_end_time
        )
        or_bars = session_data[or_mask]

        if len(or_bars) == 0:
            continue

        # Calculate OR high and low
        session_or_high = or_bars["high"].max()
        session_or_low = or_bars["low"].min()
        session_or_mid = (session_or_high + session_or_low) / 2
        session_or_range = session_or_high - session_or_low

        # Fill OR levels for entire session
        session_indices = data.index[session_mask]
        for idx in session_indices:
            pos = data.index.get_loc(idx)
            or_high[pos] = session_or_high
            or_low[pos] = session_or_low
            or_mid[pos] = session_or_mid
            or_range[pos] = session_or_range

            # Determine breakout status
            close_price = data.loc[idx, "close"]
            if close_price > session_or_high:
                or_breakout[pos] = 1  # Above OR
            elif close_price < session_or_low:
                or_breakout[pos] = -1  # Below OR
            else:
                or_breakout[pos] = 0  # Inside OR

    # Create result dataframe
    data_copy = data.copy()
    data_copy["or_high"] = or_high
    data_copy["or_low"] = or_low
    data_copy["or_mid"] = or_mid
    data_copy["or_range"] = or_range
    data_copy["or_breakout"] = or_breakout

    return data_copy
