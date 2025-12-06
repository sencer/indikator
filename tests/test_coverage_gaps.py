import pandas as pd
import pytest

from indikator import opening_range, vwap, vwap_anchored


def test_opening_range_empty_period():
  """Test opening_range when a session has data but none within the opening range window."""
  # Create data where the first bar of the day is at 10:00, but OR is 9:30-9:45 (15 mins)
  dates = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:05"])
  data = pd.DataFrame(
    {"high": [100.0, 101.0], "low": [99.0, 100.0], "close": [99.5, 100.5]}, index=dates
  )

  # Run with default start 09:30 and 15 mins duration -> end 09:45
  # Data starts at 10:00, so no bars in OR.
  res = opening_range(data, minutes=15, session_start="09:30")

  # Should result in NaN for OR columns because no bars were found
  assert res["or_high"].isna().all()
  assert res["or_low"].isna().all()


def test_vwap_invalid_freq():
  """Test vwap with invalid session frequency."""
  dates = pd.date_range("2024-01-01", periods=5)
  data = pd.DataFrame(
    {"high": [10] * 5, "low": [9] * 5, "close": [9.5] * 5, "volume": [100] * 5},
    index=dates,
  )

  with pytest.raises(ValueError, match="Invalid session_freq"):
    vwap(data, session_freq="INVALID")  # type: ignore


def test_vwap_anchored_non_datetime_index():
  """Test vwap_anchored with anchor_datetime but index is not DatetimeIndex."""
  data = pd.DataFrame({
    "high": [10] * 5,
    "low": [9] * 5,
    "close": [9.5] * 5,
    "volume": [100] * 5,
  })  # Default range index (int)

  with pytest.raises(ValueError, match="anchor_datetime requires DatetimeIndex"):
    vwap_anchored(data, anchor_datetime="2024-01-01")
