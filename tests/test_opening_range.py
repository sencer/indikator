"""Tests for Opening Range Breakout."""

import pandas as pd

from indikator.opening_range import opening_range


def test_opening_range_basic():
  """Test basic Opening Range calculation."""
  # 1 hour data, minute intervals.
  # First 30 mins: High=105, Low=95.
  # Rest of day: High=110, Low=90.

  dates = pd.date_range("2024-01-01 09:30", "2024-01-01 16:00", freq="1min")
  n = len(dates)

  high = pd.Series([100.0] * n, index=dates)
  low = pd.Series([100.0] * n, index=dates)
  close = pd.Series([100.0] * n, index=dates)

  # Set OR (09:30 - 10:00)
  # 30 mins = 30 bars (inclusive of 09:30? usually 09:30 to 10:00)
  # Let's say bar 0 to 29.

  high.iloc[5] = 105.0  # High in OR
  low.iloc[10] = 95.0  # Low in OR

  # Set Breakout later
  close.iloc[40] = 106.0  # Breakout Up

  result = opening_range(high, low, close, period_minutes=30)
  assert hasattr(result, "to_pandas")
  df = result.to_pandas()

  assert "or_high" in df.columns
  assert "or_low" in df.columns
  assert "or_breakout" in df.columns

  # OR High should be 105 for the day (after it is established)
  # Note: logic might establish it progressively or at end of OR.
  # Usually standard OR indicators backfill or forward fill from end of OR.
  # My implementation likely computes it.

  # Check at end of day
  assert df["or_high"].iloc[-1] == 105.0
  assert df["or_low"].iloc[-1] == 95.0

  # Breakout signal
  # My implementation sets breakout=1 on the bar it happens?
  # Or Status?
  # 1 = Breakout Up.

  # Check bar 40
  assert df["or_breakout"].iloc[40] == 1
