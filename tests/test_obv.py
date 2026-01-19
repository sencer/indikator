"""Tests for On Balance Volume (OBV)."""

import numpy as np
import pandas as pd
import pytest

from indikator.obv import obv

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_obv_basic():
  """Test basic OBV calculation."""
  # Close: 10, 11 (Up), 10 (Down), 10 (Flat)
  # Vol:   100, 100,    100,       100
  # OBV:   Nan, +100,  0,         0 (Flat maintains prev)

  # Note: First bar usually initialized to Volume? Or 0? Or NaN?
  # Talib initializes OBV[0] = Volume[0].
  # Some init at 0.

  close = pd.Series([10.0, 11.0, 10.0, 10.0])
  volume = pd.Series([100.0, 100.0, 100.0, 100.0])

  result = obv(close, volume)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  # My implementation (Numba) usually starts at 0 or vol?
  # Let's check logic:
  # obv = np.zeros
  # obv[0] = volume[0] ?

  # Assuming standard implementation:
  # Up -> Add vol
  # Down -> Sub vol

  # If implementation is:
  # OBV[0] = 0 (or Vol[0])

  diff = res.diff()
  assert diff.iloc[1] == 100.0  # Up
  assert diff.iloc[2] == -100.0  # Down
  assert diff.iloc[3] == 0.0  # Flat


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_obv_matches_talib():
  """Test OBV against TA-Lib."""
  np.random.seed(42)
  close = pd.Series(np.random.randn(100) + 100)
  volume = pd.Series(np.abs(np.random.randn(100)) * 1000)

  result = obv(close, volume)
  res = result.to_pandas()

  expected = talib.OBV(close.values, volume.values)

  pd.testing.assert_series_equal(
    res, pd.Series(expected, index=close.index, name="obv")
  )
