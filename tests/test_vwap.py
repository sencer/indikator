import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from indikator.vwap import vwap, vwap_anchored


def test_vwap_basic():
  """Test basic VWAP calculation."""
  # Const price, volume -> VWAP should equal price
  dates = pd.date_range("2024-01-01", periods=10, freq="1min")
  high = pd.Series([10.0] * 10, index=dates)
  low = pd.Series([10.0] * 10, index=dates)
  close = pd.Series([10.0] * 10, index=dates)
  volume = pd.Series([100.0] * 10, index=dates)

  result = vwap(high, low, close, volume, anchor="D")
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "vwap"
  assert np.allclose(res, 10.0)


def test_vwap_reset():
  """Test VWAP reset."""
  # Two days.
  # Day 1: Price 10, Vol 100. VWAP=10.
  # Day 2: Price 20, Vol 100. VWAP should be 20 (reset).

  dates = pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00"])
  high = pd.Series([10.0, 20.0], index=dates)
  low = pd.Series([10.0, 20.0], index=dates)
  close = pd.Series([10.0, 20.0], index=dates)
  volume = pd.Series([100.0, 100.0], index=dates)

  result = vwap(high, low, close, volume, anchor="D")
  res = result.to_pandas()

  assert np.isclose(res.iloc[0], 10.0)
  assert np.isclose(res.iloc[1], 20.0)

  # If we computed continuous VWAP (no reset or huge anchor):
  # (10*100 + 20*100) / 200 = 15.
  # But with Daily anchor, second day resets.


def test_vwap_int_anchor():
  """Test bar-count based anchor."""
  dates = pd.date_range("2024-01-01", periods=6, freq="1min")
  # 3 bars per set
  # Set 1 (0-2): P=10
  # Set 2 (3-5): P=20

  high = pd.Series([10] * 3 + [20] * 3, index=dates)
  low = pd.Series([10] * 3 + [20] * 3, index=dates)
  close = pd.Series([10] * 3 + [20] * 3, index=dates)
  volume = pd.Series([100] * 6, index=dates)

  result = vwap(high, low, close, volume, anchor=3)
  res = result.to_pandas()

  assert np.allclose(res.iloc[0:3], 10.0)
  assert np.allclose(res.iloc[3:6], 20.0)


def test_vwap_invalid_freq():
  """Test vwap with invalid session frequency."""
  dates = pd.date_range("2024-01-01", periods=5)
  data = pd.DataFrame(
    {"high": [10] * 5, "low": [9] * 5, "close": [9.5] * 5, "volume": [100] * 5},
    index=dates,
  )

  # Validation should catch this
  with pytest.raises((ValueError, KeyError, AssertionError, ValidationError)):
    vwap(
      pd.Series([10] * 5, index=dates),
      pd.Series([9] * 5, index=dates),
      pd.Series([9.5] * 5, index=dates),
      pd.Series([100] * 5, index=dates),
      anchor="INVALID",  # type: ignore
    )


def test_vwap_anchored_non_datetime_index():
  """Test vwap_anchored with anchor_datetime but index is not DatetimeIndex."""
  data = pd.DataFrame({
    "high": [10] * 5,
    "low": [9] * 5,
    "close": [9.5] * 5,
    "volume": [100] * 5,
  })  # Default range index (int)

  # Validation or function logic should catch this
  with pytest.raises(
    (ValueError, ValidationError), match="anchor_datetime requires DatetimeIndex"
  ):
    vwap_anchored(data, anchor_datetime="2024-01-01")


def test_vwap_anchored_datetime_not_found():
  """Test vwap_anchored when datetime is not found but close matches exist."""
  dates = pd.date_range("2024-01-01", periods=10, freq="1D")
  data = pd.DataFrame(
    {
      "high": [10.0] * 10,
      "low": [9.0] * 10,
      "close": [9.5] * 10,
      "volume": [100.0] * 10,
    },
    index=dates,
  )

  # Use a datetime that doesn't exist exactly but has a close match
  result_obj = vwap_anchored(data, anchor_datetime="2024-01-03 12:00")
  result = result_obj.to_pandas()

  # Should find the nearest datetime and work - returns Series
  assert isinstance(result, pd.Series)
  assert result.name == "vwap_anchored"
  # First 2 bars should be NaN (before anchor)
  assert result.iloc[:2].isna().all()
