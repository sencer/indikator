"""Tests for Aroon indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.aroon import aroon

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_aroon_basic():
  """Test basic Aroon calculation."""
  # Price strictly increasing -> Highs are recent, Lows are old
  # Aroon Up should be 100, Aroon Down should be 0
  high = pd.Series(np.arange(100.0))
  low = pd.Series(np.arange(100.0) - 10)

  result = aroon(high, low, period=10)
  assert hasattr(result, "to_pandas")
  df = result.to_pandas()

  assert "aroon_up" in df.columns
  assert "aroon_down" in df.columns
  assert "aroon_osc" in df.columns

  # Check values after warmup
  # Latest high is at index i (0 bars ago) -> Up = 100
  # Latest low is at index i-10 (10 bars ago) -> Down = 0
  valid = df.iloc[20:]
  assert np.allclose(valid["aroon_up"], 100.0)
  assert np.allclose(valid["aroon_down"], 0.0)
  assert np.allclose(valid["aroon_osc"], 100.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_aroon_matches_talib():
  """Test Aroon against TA-Lib."""
  np.random.seed(42)
  high = pd.Series(np.random.uniform(100, 200, 100))
  low = pd.Series(np.random.uniform(50, 90, 100))

  period = 14
  result = aroon(high, low, period=period)
  df = result.to_pandas()

  exp_down, exp_up = talib.AROON(high.values, low.values, timeperiod=period)
  exp_osc = talib.AROONOSC(high.values, low.values, timeperiod=period)

  pd.testing.assert_series_equal(
    df["aroon_up"], pd.Series(exp_up, index=high.index, name="aroon_up")
  )
  pd.testing.assert_series_equal(
    df["aroon_down"], pd.Series(exp_down, index=high.index, name="aroon_down")
  )
  pd.testing.assert_series_equal(
    df["aroon_osc"], pd.Series(exp_osc, index=high.index, name="aroon_osc")
  )
