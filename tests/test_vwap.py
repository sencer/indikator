"""Tests for VWAP indicator."""

import numpy as np
import pandas as pd

from indikator.vwap import vwap


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
