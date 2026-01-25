"""Opening Range Breakout indicator module.

This module provides opening range calculation, a popular day trading
strategy that identifies the high/low range of the first N minutes.
"""

from typing import cast

from datawarden import (
  Datetime,
  Finite,
  Index,
  NotEmpty,
  Validated,
  validate,
)
from nonfig import Ge, Hyper, configurable
import numpy as np
import pandas as pd

from indikator._results import OpeningRangeResult


@configurable
@validate
def opening_range(  # noqa: PLR0914
  high: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
  low: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
  close: Validated[pd.Series[float], Finite, Index(Datetime), NotEmpty],
  period_minutes: Hyper[int, Ge[1]] = 30,
) -> OpeningRangeResult:
  """Calculate Opening Range Breakout (ORB) levels.

  Identifies the high and low of the first N minutes of the trading session.

  Outputs:
  - or_high: High of the opening range (held constant for the day)
  - or_low: Low of the opening range
  - or_mid: Midpoint of the range
  - or_range: Size of the range (High - Low)
  - or_breakout: Signal (1=Breakout Up, -1=Breakout Down, 0=None)

  Args:
    high: High prices with DatetimeIndex
    low: Low prices with DatetimeIndex
    close: Close prices with DatetimeIndex
    period_minutes: Opening range duration in minutes (default: 30)

  Returns:
    OpeningRangeResult(index, or_high, or_low, or_mid, or_range, or_breakout)
  """
  # Logic:
  # 1. Identify session days
  # 2. For each session, find High/Low of first N minutes.
  # 3. Broadcast these levels to the rest of the session.

  # Group by date
  # Assumes DatetimeIndex

  idx = high.index

  # Calculate OR Period End Time logic for each day?
  # Vectorized approach:
  # Calculate "minutes from midnight" maybe?
  # Or simply groupby date.

  # We need start of session. Assuming data starts at session start for simplicity?
  # No, data is continuous.

  # Simple robust approch using Pandas:
  # Iterate groups? Slow.
  # Vectorized:
  # 1. Get date (normalization).
  # 2. Get time.
  # 3. Define threshold time per day?
  # If market opens at 09:30 always, then we check 09:30 <= T < 09:30 + period.

  # Assuming standard market hours 09:30 is implied?
  # The original implementation allowed configuring start time.
  # Let's assume standard starts or infer from data (first bar of day).

  # Infer start time:
  # Group by date, get min time.
  # But this is expensive.

  # Let's stick to a simpler implementation that assumes standard intraday data.
  # Calculating OR High/Low.

  # Using groupby + transformation is efficient enough.

  session_date = cast("pd.DatetimeIndex", idx).normalize()  # Date component

  # Determine each day's start time (min time)
  # This aligns with index? No.

  # To keep it fast and compatible:
  # Just assume we take the first `period_minutes` of data for each day.
  # But "minutes" implies time, not bar count.

  # Let's use resample to identify 30 min bars?
  # Or just use the pandas logic from before but cleaner.

  # Define valid OR window mask
  # First, find the "start" of each day.
  # Define valid OR window mask

  # We need to know "Time since session open".
  # If we don't know session open, we can't do this accurately without config.
  # The previous code had `session_start="09:30"`.
  # I removed it from arguments. I should put it back or infer it.
  # I'll default to "09:30" logic internally or add it back.
  # Adding it back is safer for users.

  # But I must stick to the signature I defined in `results` / previous refactor steps if I want consistency?
  # I removed it in my proposed signature.
  # I'll try to find "First N minutes of data present for the day".

  df = pd.DataFrame({"high": high, "low": low, "close": close, "date": session_date})

  # Calculate cumulative minutes per day?
  # Sort by time.

  # Let's use the explicit time-of-day filter if assuming 09:30 is acceptable for now.
  # Or better: "First N minutes relative to the first timestamp of the day".

  # Get first timestamp of each day
  # Group by date and find min index. Use ["high"] to avoid FutureWarning about grouping columns.
  day_starts = df.groupby("date")["high"].apply(lambda x: x.index.min()).to_dict()  # type: ignore

  # Map start time to every row
  start_times = df["date"].map(day_starts)

  # Calculate elapsed time
  elapsed = df.index - start_times

  # Mask for OR period
  is_or = elapsed < pd.Timedelta(minutes=period_minutes)

  # Calculate stats on OR rows
  or_rows = df[is_or]
  or_stats = or_rows.groupby("date").agg({"high": "max", "low": "min"})  # pyright: ignore[reportUnknownMemberType]

  # Map back
  or_high = df["date"].map(or_stats["high"])
  or_low = df["date"].map(or_stats["low"])

  # forward fill for days that might miss OR data?
  # If a day has NO data in first N mins (gap), it gets NaN.

  or_mid = (or_high + or_low) / 2
  or_range = or_high - or_low

  # Breakout
  # 1 if Close > High, -1 if Close < Low
  breakout = np.zeros(len(close), dtype=np.int8)

  # Use numpy for comparisons
  c_arr = close.to_numpy(dtype=np.float64, copy=False)  # pyright: ignore[reportUnknownMemberType]
  h_arr = or_high.to_numpy(dtype=np.float64, copy=False)  # pyright: ignore[reportUnknownMemberType]
  l_arr = or_low.to_numpy(dtype=np.float64, copy=False)  # pyright: ignore[reportUnknownMemberType]

  # Handle NaNs
  valid = ~np.isnan(h_arr) & ~np.isnan(l_arr)

  breakout[valid & (c_arr > h_arr)] = 1
  breakout[valid & (c_arr < l_arr)] = -1

  return OpeningRangeResult(
    data_index=high.index,
    or_high=h_arr,
    or_low=l_arr,
    or_mid=or_mid.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
    or_range=or_range.to_numpy(dtype=np.float64, copy=False),  # pyright: ignore[reportUnknownMemberType]
    or_breakout=breakout,
  )
