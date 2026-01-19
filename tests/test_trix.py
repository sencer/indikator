"""Tests for TRIX indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.trix import trix

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_trix_basic():
  """Test basic TRIX calculation."""
  data = pd.Series(np.linspace(10, 100, 100))  # Strong uptrend

  result = trix(data, period=10)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "trix"
  assert len(res) == 100

  # Should be positive in uptrend
  valid = res.dropna()
  assert (valid > 0).all()


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_trix_matches_talib():
  """Test TRIX against TA-Lib."""
  np.random.seed(42)
  # Needs enough data for triple smoothing
  data = pd.Series(np.random.randn(200) + 100)
  period = 30

  result = trix(data, period=period)
  res = result.to_pandas()

  expected = talib.TRIX(data.values, timeperiod=period)

  # TA-Lib TRIX output matches?
  pd.testing.assert_series_equal(
    res, pd.Series(expected, index=data.index, name="trix")
  )
