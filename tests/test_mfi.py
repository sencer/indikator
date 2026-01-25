"""Tests for Money Flow Index (MFI)."""

import numpy as np
import pandas as pd
import pytest

from indikator.mfi import mfi

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_mfi_basic():
  """Test basic MFI calculation."""
  # All prices up, high volume -> MFI should be 100
  # Typical Price = (H+L+C)/3
  # If TP increases every bar, all flow is positive.

  high = pd.Series(np.arange(100, 200, dtype=float))
  low = pd.Series(np.arange(90, 190, dtype=float))
  close = pd.Series(np.arange(95, 195, dtype=float))
  volume = pd.Series([1000.0] * 100)

  result = mfi(high, low, close, volume, period=10)
  assert hasattr(result, "to_pandas")
  res = result.to_pandas()

  assert res.name == "mfi"
  # After warmup, all flow positive => MFI = 100
  valid = res.iloc[15:]
  assert np.allclose(valid, 100.0)


def test_mfi_down():
  """Test MFI all down."""
  high = pd.Series(np.arange(200, 100, -1, dtype=float))
  low = pd.Series(np.arange(190, 90, -1, dtype=float))
  close = pd.Series(np.arange(195, 95, -1, dtype=float))
  volume = pd.Series([1000.0] * 100)

  result = mfi(high, low, close, volume, period=10)
  res = result.to_pandas()

  # All flow negative => MFI = 0
  valid = res.iloc[15:]
  # Wait, MFI calc handles division by zero?
  # If Neg Flow is huge and Pos Flow is 0, Ratio is 0. MFI = 100 - 100/(1+0) = 0.
  assert np.allclose(valid, 0.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_mfi_matches_talib():
  """Test MFI against TA-Lib."""
  np.random.seed(42)
  high = pd.Series(np.random.uniform(105, 110, 100))
  low = pd.Series(np.random.uniform(95, 100, 100))
  close = pd.Series(np.random.uniform(95, 110, 100))
  volume = pd.Series(np.random.uniform(100, 1000, 100))

  period = 14
  result = mfi(high, low, close, volume, period=period)
  res = result.to_pandas()

  expected = talib.MFI(
    high.values, low.values, close.values, volume.values, timeperiod=period
  )

  pd.testing.assert_series_equal(res, pd.Series(expected, index=high.index, name="mfi"))
