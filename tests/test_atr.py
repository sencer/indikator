"""Tests for ATR (Average True Range)."""

import numpy as np
import pandas as pd
import pytest

from indikator.atr import atr

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_atr_basic():
  """Test basic ATR calculation."""
  # High=12, Low=10, Close=11. Range=2.
  # If constant range of 2. ATR should converge to 2.

  high = pd.Series([12.0] * 50)
  low = pd.Series([10.0] * 50)
  close = pd.Series([11.0] * 50)

  result = atr(high, low, close, period=14)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "atr"

  # After warmup, ATR should be 2.0
  valid = res.iloc[20:]
  assert np.allclose(valid, 2.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_atr_matches_talib():
  """Test ATR against TA-Lib."""
  np.random.seed(42)
  high = pd.Series(np.random.uniform(105, 110, 100))
  low = pd.Series(np.random.uniform(95, 100, 100))
  close = pd.Series(np.random.uniform(95, 110, 100))

  period = 14
  result = atr(high, low, close, period=period)
  res = result.to_pandas()

  expected = talib.ATR(high.values, low.values, close.values, timeperiod=period)

  pd.testing.assert_series_equal(
    res, pd.Series(expected, index=high.index, name="atr"), check_exact=False, rtol=0.05
  )
