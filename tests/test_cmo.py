"""Tests for Chande Momentum Oscillator (CMO)."""

import numpy as np
import pandas as pd
import pytest

from indikator.cmo import cmo

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_cmo_basic():
  """Test basic CMO calculation."""
  # Strong uptrend
  data = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

  # Period 2
  # Diff: 1, 1, 1, 1, 1 (all up)
  # SumUp = 2, SumDown = 0 (for window 2)
  # CMO = 100 * (2-0)/(2+0) = 100

  result = cmo(data, period=2)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "cmo"
  assert len(res) == 6

  # Third value (index 2) should be 100
  assert np.isclose(res.iloc[2], 100.0)


def test_cmo_downtrend():
  """Test CMO in a downtrend."""
  data = pd.Series([20.0, 19.0, 18.0, 17.0, 16.0])

  result = cmo(data, period=2)
  res = result.to_pandas()

  # Should be -100
  assert np.isclose(res.iloc[2], -100.0)


def test_cmo_oscillation():
  """Test CMO mixed movement."""
  data = pd.Series([10, 12, 11, 13, 12])

  result = cmo(data, period=2)
  res = result.to_pandas()

  valid = res.dropna()
  assert (valid <= 100).all()
  assert (valid >= -100).all()


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_cmo_matches_talib():
  """Test CMO values match TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 14

  result = cmo(data, period=period)
  res = result.to_pandas()

  expected = talib.CMO(data.values, timeperiod=period)

  pd.testing.assert_series_equal(res, pd.Series(expected, index=data.index, name="cmo"))
