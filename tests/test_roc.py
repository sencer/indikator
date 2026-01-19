"""Tests for ROC (Rate of Change) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.roc import roc

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_roc_basic():
  """Test basic ROC calculation."""
  # Simple uptrend
  data = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])

  # Period 1 ROC
  # (102-100)/100 = 2%
  result = roc(data, period=1)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "roc"
  assert len(res) == 5
  assert np.isclose(res.iloc[1], 2.0)
  assert np.isnan(res.iloc[0])


def test_roc_downtrend():
  """Test ROC in a downtrend."""
  prices = pd.Series([100.0, 95.0, 90.0, 85.0, 80.0])

  result = roc(prices, period=1)
  res = result.to_pandas()

  # All valid ROC values should be negative
  valid = res.dropna()
  assert (valid < 0).all()
  assert np.isclose(res.iloc[1], -5.0)  # (95-100)/100 = -5%


def test_roc_insufficient_data():
  """Test ROC behavior with insufficient data."""
  data = pd.Series([10.0, 11.0, 12.0])  # Only 3 points

  # With period=14 (default)
  result = roc(data)
  res = result.to_pandas()

  assert len(res) == 3
  assert res.isna().all()


def test_roc_period_parameter():
  """Test ROC with different periods."""
  prices = pd.Series(np.random.randn(20) + 100)

  result1 = roc(prices, period=2)
  res1 = result1.to_pandas()

  result2 = roc(prices, period=5)
  res2 = result2.to_pandas()

  # Check NaN counts
  assert res1.isna().sum() == 2
  assert res2.isna().sum() == 5


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_roc_matches_talib():
  """Test ROC values match TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 14

  result = roc(data, period=period)
  res = result.to_pandas()

  expected = talib.ROC(data.values, timeperiod=period)

  # TA-Lib returns numpy array, we return Series
  pd.testing.assert_series_equal(res, pd.Series(expected, index=data.index, name="roc"))
