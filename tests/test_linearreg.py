import numpy as np
import pandas as pd
import pytest
import talib

from indikator.linearreg import (
  linearreg,
  linearreg_angle,
  linearreg_intercept,
  linearreg_slope,
  tsf,
)


@pytest.mark.parametrize("period", [5, 14, 30])
def test_linearreg_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = linearreg(data, period=period)
  expected = talib.LINEARREG(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(),
    pd.Series(expected, index=data.index, name="linearreg"),
    atol=1e-10,
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_linearreg_intercept_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = linearreg_intercept(data, period=period)
  expected = talib.LINEARREG_INTERCEPT(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(),
    pd.Series(expected, index=data.index, name="linearreg_intercept"),
    atol=1e-10,
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_linearreg_angle_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = linearreg_angle(data, period=period)
  expected = talib.LINEARREG_ANGLE(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(),
    pd.Series(expected, index=data.index, name="linearreg_angle"),
    atol=1e-10,
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_linearreg_slope_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = linearreg_slope(data, period=period)
  expected = talib.LINEARREG_SLOPE(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(),
    pd.Series(expected, index=data.index, name="linearreg_slope"),
    atol=1e-10,
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_tsf_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = tsf(data, period=period)
  expected = talib.TSF(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.to_pandas(), pd.Series(expected, index=data.index, name="tsf"), atol=1e-10
  )
