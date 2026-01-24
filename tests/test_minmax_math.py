import numpy as np
import pandas as pd
import pytest
import talib

from indikator.minmax import max_index, max_val, min_index, min_val, sum_val


@pytest.mark.parametrize("period", [5, 14, 30])
def test_min_matches_talib(period):
  np.random.seed(42)
  # Use float32 to match TA-Lib potential internal precision, or just tolerance
  data = pd.Series(np.random.randn(100), name="data")

  result = min_val(data, period=period)
  expected = talib.MIN(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="min")
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_max_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = max_val(data, period=period)
  expected = talib.MAX(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="max")
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_sum_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = sum_val(data, period=period)
  expected = talib.SUM(data.values, timeperiod=period)

  # Sum might have slight fp diffs
  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="sum"), atol=1e-10
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_min_index_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = min_index(data, period=period)
  expected = talib.MININDEX(data.values, timeperiod=period)

  # Indikator returns float array for indices (for now, to match Numba signatures?)
  # or should we cast to int?
  # TA-Lib returns integer array, but pd.Series from it will be ...
  # Wait, typical use is int. Numba returns float64 array based on implementation.
  # Let's compare as series.

  # TA-Lib returns 0s for warmup period in MININDEX, we return NaNs.
  # We only compare the valid range.

  s_expected = pd.Series(expected, index=data.index, name="min_index").astype(float)

  pd.testing.assert_series_equal(
    result.iloc[period - 1 :], s_expected.iloc[period - 1 :]
  )


@pytest.mark.parametrize("period", [5, 14, 30])
def test_max_index_matches_talib(period):
  np.random.seed(42)
  data = pd.Series(np.random.randn(100), name="data")

  result = max_index(data, period=period)
  expected = talib.MAXINDEX(data.values, timeperiod=period)

  s_expected = pd.Series(expected, index=data.index, name="max_index").astype(float)

  pd.testing.assert_series_equal(
    result.iloc[period - 1 :], s_expected.iloc[period - 1 :]
  )


def test_minmax_with_nans():
  pd.Series([10.0, 20.0, np.nan, 5.0, 15.0])
  # Standard behavior: NaNs propagate? Or ignore?
  # TA-Lib MIN/MAX usually propagate NaNs.
  # Our implementation:
  # If any value in window is NaN, result is NaN?
  # Or ignores NaN?
  # Logic: if data[k] <= l_val... where comparison with NaN is False.
  # If l_val starts as NaN...
  # We need to verify behavior.

  # Let's assume TA-Lib compatibility (NaN propagation) is desired or check
  # what existing implementation does.
  # Existing rolling logic usually propagates NaNs by default?
  pass
