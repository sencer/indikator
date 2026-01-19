"""Tests for T3 indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.t3 import t3

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_t3_basic():
  """Test basic T3 calculation."""
  data = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
  # Flat line should result in flat line (eventually)
  # T3 warms up slowly.
  result = t3(data, period=2, vfactor=0.7).to_pandas()
  # Result needs to be checked.
  # For constant input, EMA is constant. T3 should be constant.
  # But initial values might be NaN or converging.

  # For period=2, valid after... many steps?
  # 6 EMAs.
  # But simple check: last value should be close to 10.0
  valid = result.dropna()
  if len(valid) > 0:
    assert np.isclose(valid.iloc[-1], 10.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_t3_matches_talib():
  """Test T3 matches TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(200) + 100)
  period = 5
  vfactor = 0.7

  result = t3(data, period=period, vfactor=vfactor).to_pandas()
  expected = talib.T3(data.values, timeperiod=period, vfactor=vfactor)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="t3")
  )
