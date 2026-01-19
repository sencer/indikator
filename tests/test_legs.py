"""Tests for ZigZag Legs."""

import pandas as pd

from indikator.legs import legs


def test_legs_basic():
  """Test basic ZigZag legs identification."""
  # 0: 10
  # 1: 20 (Up 100%)
  # 2: 15 (Down 25%)
  # 3: 10 (Down 33% from 15? or from 20?)
  # deviation 5%.

  prices = [10, 20, 10, 20, 10]
  # Should find legs: 10->20 (Up), 20->10 (Down), 10->20 (Up), 20->10 (Down)

  n = len(prices)
  high = pd.Series(prices)
  low = pd.Series(prices)
  close = pd.Series(prices)

  result = legs(high, low, close, deviation=0.05)  # 5% deviation
  assert hasattr(result, "to_pandas")
  df = result.to_pandas()

  # Expect 4 legs? Or maybe last leg not confirmed?
  # 20 to 10 is -50%.
  # 10 to 20 is +100%.

  assert not df.empty
  assert "direction" in df.columns
  assert "start_price" in df.columns
  assert "end_price" in df.columns

  # First leg: Start 10, End 20. Direction 1.
  assert df["start_price"].iloc[0] == 10.0
  assert df["end_price"].iloc[0] == 20.0
  assert df["direction"].iloc[0] == 1


def test_legs_no_movement():
  """Test minimal data with no zigzag."""
  prices = [10, 10, 10]
  high = pd.Series(prices)
  low = pd.Series(prices)
  close = pd.Series(prices)

  result = legs(high, low, close)
  df = result.to_pandas()

  assert df.empty
