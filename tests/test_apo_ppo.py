"""Tests for APO and PPO indicators."""

import numpy as np
import pandas as pd
import pytest

from indikator.apo import apo
from indikator.ppo import ppo

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


def test_apo_basic():
  """Test basic APO calculation."""
  data = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
  # Fast(2), Slow(4)
  # SMA(2): -, 11, 13, 15, 17
  # SMA(4): -, -, -, 13, 15
  # APO = 15-13=2, 17-15=2

  result = apo(data, fast_period=2, slow_period=4).to_pandas()
  assert np.isnan(result[0])
  assert np.isnan(result[1])
  assert np.isnan(result[2])  # Slow needs 4 points (index 3)
  assert np.isclose(result[3], 2.0)
  assert np.isclose(result[4], 2.0)


def test_ppo_basic():
  """Test basic PPO calculation."""
  data = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
  # Fast(2), Slow(4)
  # SMA(2): 11, 13, 15, 17
  # SMA(4): 13, 15
  # PPO = (15-13)/13*100 = 2/13*100 = 15.38...
  #       (17-15)/15*100 = 2/15*100 = 13.33...

  result = ppo(data, fast_period=2, slow_period=4).to_pandas()
  assert np.isclose(result[3], (2.0 / 13.0) * 100.0)
  assert np.isclose(result[4], (2.0 / 15.0) * 100.0)


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_apo_matches_talib_sma():
  """Test APO matches TA-Lib (SMA default)."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  fast = 12
  slow = 26

  result = apo(data, fast_period=fast, slow_period=slow).to_pandas()
  expected = talib.APO(data.values, fastperiod=fast, slowperiod=slow, matype=0)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="apo")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_ppo_matches_talib_sma():
  """Test PPO matches TA-Lib (SMA default)."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  fast = 12
  slow = 26

  result = ppo(data, fast_period=fast, slow_period=slow).to_pandas()
  expected = talib.PPO(data.values, fastperiod=fast, slowperiod=slow, matype=0)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="ppo")
  )


@pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
def test_apo_matches_talib_ema():
  """Test APO matches TA-Lib with EMA."""
  np.random.seed(42)
  data = pd.Series(np.random.randn(100) + 100)
  fast = 12
  slow = 26

  result = apo(data, fast_period=fast, slow_period=slow, matype=1).to_pandas()
  expected = talib.APO(data.values, fastperiod=fast, slowperiod=slow, matype=1)

  pd.testing.assert_series_equal(
    result, pd.Series(expected, index=data.index, name="apo")
  )
