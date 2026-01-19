"""Tests for Pivot Points."""

import numpy as np
import pandas as pd

from indikator.pivots import pivots


def test_pivots_basic():
  """Test basic Standard Pivots."""
  # Day 1: High=110, Low=90, Close=100.
  # Day 2: Should have pivots based on Day 1.
  # P = (110+90+100)/3 = 100.
  # R1 = 2*100 - 90 = 110.
  # S1 = 2*100 - 110 = 90.

  dates = pd.to_datetime(["2024-01-01 10:00", "2024-01-02 10:00"])
  high = pd.Series([110.0, 105.0], index=dates)
  low = pd.Series([90.0, 95.0], index=dates)
  close = pd.Series([100.0, 100.0], index=dates)

  result = pivots(high, low, close, method="standard", anchor="D")
  assert hasattr(result, "to_pandas")
  df = result.to_pandas()

  assert "pivot" in df.columns
  assert "r1" in df.columns
  assert "s1" in df.columns

  # Day 2 (index 1) should have Day 1's pivots
  assert np.isclose(df["pivot"].iloc[1], 100.0)
  assert np.isclose(df["r1"].iloc[1], 110.0)
  assert np.isclose(df["s1"].iloc[1], 90.0)

  # Day 1 (index 0) usually has NaN (no prior day)
  assert np.isnan(df["pivot"].iloc[0])


def test_pivots_methods():
  """Test different calculation methods return correct columns."""
  dates = pd.date_range("2024-01-01", periods=10, freq="D")
  high = pd.Series(np.random.randn(10) + 100, index=dates)
  low = pd.Series(np.random.randn(10) + 90, index=dates)
  close = pd.Series(np.random.randn(10) + 95, index=dates)

  # Fibonacci
  res_fib = pivots(high, low, close, method="fibonacci").to_pandas()
  assert "r3" in res_fib.columns

  # Woodie
  res_woodie = pivots(high, low, close, method="woodie").to_pandas()
  assert "pivot" in res_woodie.columns
  assert "r1" in res_woodie.columns
  # Woodie has fewer levels?

  # Camarilla
  res_cam = pivots(high, low, close, method="camarilla").to_pandas()
  assert "r4" in res_cam.columns
