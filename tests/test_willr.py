"""Tests for Williams %R."""

import numpy as np
import pandas as pd
import pytest

from indikator.willr import willr

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_willr_basic():
  """Test basic WillR calculation."""
  # High=10, Low=0. Close=0 -> %R = -100
  # Close=10 -> %R = 0
  # Close=5 -> %R = -50

  high = pd.Series([10.0] * 20)
  low = pd.Series([0.0] * 20)

  # Case 1: Close at Low
  close_low = pd.Series([0.0] * 20)
  res_low = willr(high, low, close_low, period=5).to_pandas()
  assert np.allclose(res_low.dropna(), -100.0)

  # Case 2: Close at High
  close_high = pd.Series([10.0] * 20)
  res_high = willr(high, low, close_high, period=5).to_pandas()
  assert np.allclose(res_high.dropna(), -0.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_willr_matches_talib():
  """Test WillR against TA-Lib."""
  np.random.seed(42)
  high = pd.Series(np.random.uniform(105, 110, 100))
  low = pd.Series(np.random.uniform(95, 100, 100))
  close = pd.Series(np.random.uniform(95, 110, 100))

  period = 14
  result = willr(high, low, close, period=period)
  res = result.to_pandas()

  expected = talib.WILLR(high.values, low.values, close.values, timeperiod=period)

  pd.testing.assert_series_equal(
    res, pd.Series(expected, index=high.index, name="willr")
  )
