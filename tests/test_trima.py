"""Tests for TRIMA (Triangular Moving Average)."""

import numpy as np
import pandas as pd
import pytest

from indikator.trima import trima

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_trima_basic():
  """Test basic TRIMA calculation."""
  data = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

  # Period 3 (Odd)
  # SMA(SMA(x, 2), 2) where 2 = (3+1)//2
  # 0: 10
  # 1: (10+11)/2 = 10.5
  # 2: (11+12)/2 = 11.5
  # 3: (12+13)/2 = 12.5
  # 4: (13+14)/2 = 13.5
  # 5: (14+15)/2 = 14.5

  # SMA of that with period 2:
  # 0: NaN
  # 1: (NaN+10.5)/2 -> NaN (Standard SMA needs valid data)
  # Actually, if compute_sma_numba propagates NaN:
  # Indices 0 to 1 will be NaN for the first SMA(2). (Wait, SMA(2) valid at index 1).
  # Index 0: NaN
  # Index 1: 10.5
  # Index 2: 11.5
  # ...
  # Second SMA(2) on [NaN, 10.5, 11.5, ...]
  # Index 1: (NaN+10.5)/2 = NaN
  # Index 2: (10.5+11.5)/2 = 11.0
  # So first valid result at index 2 (period 3-1). Correct.

  result = trima(data, period=3).to_pandas()
  assert np.isnan(result[0])
  assert np.isnan(result[1])
  assert np.isclose(result[2], 11.0)
  assert np.isclose(result[3], 12.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_trima_matches_talib_odd():
  """Test TRIMA matches TA-Lib with odd period."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 33  # Odd

  result = trima(data, period=period).to_pandas()
  expected = talib.TRIMA(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="trima")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_trima_matches_talib_even():
  """Test TRIMA matches TA-Lib with even period."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  period = 30  # Even

  result = trima(data, period=period).to_pandas()
  expected = talib.TRIMA(data.values, timeperiod=period)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="trima")
  )
