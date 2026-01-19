"""Tests for MACD indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.macd import macd

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_macd_basic():
  """Test basic MACD calculation."""
  data = pd.Series(np.random.randn(100) + 100)

  result = macd(data, fast_period=12, slow_period=26, signal_period=9)

  assert hasattr(result, "to_pandas")
  assert hasattr(result, "macd")
  assert hasattr(result, "signal")
  assert hasattr(result, "histogram")

  df = result.to_pandas()
  assert isinstance(df, pd.DataFrame)
  assert "macd" in df.columns
  assert "signal" in df.columns
  assert "histogram" in df.columns

  assert len(df) == 100


def test_macd_values():
  """Test MACD calculation correctness."""
  # Use simple data where we can verify directions
  # Rising prices -> MACD should eventually be positive
  data = pd.Series(np.linspace(10, 20, 50))

  result = macd(data, fast_period=5, slow_period=10, signal_period=5)
  df = result.to_pandas()

  # Check late values (after warmup)
  valid_macd = df["macd"].iloc[20:]
  assert (valid_macd > 0).all()


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_macd_matches_talib():
  """Test MACD against TA-Lib."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)

  fast = 12
  slow = 26
  signal = 9

  result = macd(data, fast_period=fast, slow_period=slow, signal_period=signal)
  df = result.to_pandas()

  expected_macd, expected_signal, expected_hist = talib.MACD(
    data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
  )

  pd.testing.assert_series_equal(
    df["macd"], pd.Series(expected_macd, index=data.index, name="macd")
  )
  pd.testing.assert_series_equal(
    df["signal"], pd.Series(expected_signal, index=data.index, name="signal")
  )
  pd.testing.assert_series_equal(
    df["histogram"], pd.Series(expected_hist, index=data.index, name="histogram")
  )
