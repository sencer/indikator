"""Tests for CCI (Commodity Channel Index)."""

import numpy as np
import pandas as pd
import pytest

from indikator.cci import cci

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_cci_basic():
  """Test basic CCI calculation."""
  # CCI measures deviation from mean.
  # If price is constantly increasing linearly, CCI should be positive and high.

  high = pd.Series(np.linspace(10, 20, 50))
  low = pd.Series(np.linspace(10, 20, 50))
  close = pd.Series(np.linspace(10, 20, 50))

  result = cci(high, low, close, period=10)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "cci"
  valid = res.dropna()

  # Trend is up -> CCI > 0
  assert (valid > 0).all()


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_cci_matches_talib():
  """Test CCI against TA-Lib."""
  np.random.seed(42)
  high = pd.Series(np.random.uniform(105, 110, 100))
  low = pd.Series(np.random.uniform(95, 100, 100))
  close = pd.Series(np.random.uniform(95, 110, 100))

  period = 14
  result = cci(high, low, close, period=period)
  res = result.to_pandas()

  # Note: logic handles default constant=0.015 same as TA-Lib
  expected = talib.CCI(high.values, low.values, close.values, timeperiod=period)

  pd.testing.assert_series_equal(res, pd.Series(expected, index=high.index, name="cci"))
