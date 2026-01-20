"""Tests for LINEARREG family indicators."""

import numpy as np
import pandas as pd
import pytest

from indikator.linearreg import (
  linearreg,
  linearreg_angle,
  linearreg_intercept,
  linearreg_slope,
  tsf,
)

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_linearreg_basic():
  """Test basic LINEARREG calculation."""
  # Simple uptrend
  data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

  result = linearreg(data, period=5)
  res = result.to_pandas()

  assert res.name == "linearreg"
  assert len(res) == 10
  # With perfect trend, last point should equal actual value
  assert not np.isnan(res.iloc[-1])
  # First 4 should be NaN (period=5)
  assert res[:4].isna().all()


def test_linearreg_intercept_basic():
  """Test basic LINEARREG_INTERCEPT calculation."""
  data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

  result = linearreg_intercept(data, period=5)
  res = result.to_pandas()

  assert res.name == "linearreg_intercept"
  assert len(res) == 10
  assert not np.isnan(res.iloc[-1])


def test_linearreg_angle_basic():
  """Test basic LINEARREG_ANGLE calculation."""
  data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

  result = linearreg_angle(data, period=5)
  res = result.to_pandas()

  assert res.name == "linearreg_angle"
  assert len(res) == 10
  # Perfect uptrend should have positive angle
  assert res.iloc[-1] > 0


def test_linearreg_slope_basic():
  """Test basic LINEARREG_SLOPE calculation."""
  data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

  result = linearreg_slope(data, period=5)
  res = result.to_pandas()

  assert res.name == "linearreg_slope"
  assert len(res) == 10
  # Perfect uptrend: slope should be 1 (y increases by 1 per x)
  assert np.isclose(res.iloc[-1], 1.0)


def test_tsf_basic():
  """Test basic TSF (Time Series Forecast) calculation."""
  data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

  result = tsf(data, period=5)
  res = result.to_pandas()

  assert res.name == "tsf"
  assert len(res) == 10
  # With perfect trend, TSF should project 1 ahead
  # At index 9, values 6-10 form the window. TSF should predict 11.
  assert np.isclose(res.iloc[-1], 11.0)


def test_linearreg_insufficient_data():
  """Test LINEARREG with insufficient data."""
  data = pd.Series([1.0, 2.0, 3.0])
  result = linearreg(data, period=14)
  res = result.to_pandas()
  assert res.isna().all()


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_linearreg_matches_talib():
  """Test LINEARREG matches TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 14

  result = linearreg(data, period=period).to_pandas()
  expected = talib.LINEARREG(data.values, timeperiod=period)
  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="linearreg")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_linearreg_intercept_matches_talib():
  """Test LINEARREG_INTERCEPT matches TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 14

  result = linearreg_intercept(data, period=period).to_pandas()
  expected = talib.LINEARREG_INTERCEPT(data.values, timeperiod=period)
  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="linearreg_intercept")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_linearreg_angle_matches_talib():
  """Test LINEARREG_ANGLE matches TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 14

  result = linearreg_angle(data, period=period).to_pandas()
  expected = talib.LINEARREG_ANGLE(data.values, timeperiod=period)
  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="linearreg_angle")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_linearreg_slope_matches_talib():
  """Test LINEARREG_SLOPE matches TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 14

  result = linearreg_slope(data, period=period).to_pandas()
  expected = talib.LINEARREG_SLOPE(data.values, timeperiod=period)
  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="linearreg_slope")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_tsf_matches_talib():
  """Test TSF matches TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 14

  result = tsf(data, period=period).to_pandas()
  expected = talib.TSF(data.values, timeperiod=period)
  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="tsf")
  )
