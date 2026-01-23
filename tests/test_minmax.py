import numpy as np
import pandas as pd
import pytest
import talib
from indikator.minmax import min_val, max_val, min_index, max_index, sum_val


def test_min_matches_talib():
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")
  period = 10

  result = min_val(data, period=period)
  expected = talib.MIN(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.iloc[period:],
    pd.Series(expected, index=data.index, name="min").iloc[period:],
    atol=1e-10,
  )


def test_max_matches_talib():
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")
  period = 10

  result = max_val(data, period=period)
  expected = talib.MAX(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.iloc[period:],
    pd.Series(expected, index=data.index, name="max").iloc[period:],
    atol=1e-10,
  )


def test_minindex_matches_talib():
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")
  period = 10

  result = min_index(data, period=period)
  expected = talib.MININDEX(data.values, timeperiod=period)

  # Note: TA-Lib MININDEX returns index relative to start (0-based)
  # Our implementation matches this behavior.

  pd.testing.assert_series_equal(
    result.iloc[period:],
    pd.Series(expected, index=data.index, name="min_index").iloc[period:],
    check_dtype=False,  # TA-Lib might return int or float
  )


def test_maxindex_matches_talib():
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")
  period = 10

  result = max_index(data, period=period)
  expected = talib.MAXINDEX(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.iloc[period:],
    pd.Series(expected, index=data.index, name="max_index").iloc[period:],
    check_dtype=False,
  )


def test_sum_matches_talib():
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")
  period = 10

  result = sum_val(data, period=period)
  expected = talib.SUM(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result.iloc[period:],
    pd.Series(expected, index=data.index, name="sum").iloc[period:],
    atol=1e-10,
  )
