"""Tests for STDDEV indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.stddev import stddev

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


@pytest.fixture
def price_data():
  """Generate price test data."""
  np.random.seed(42)
  return pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))


class TestStddev:
  """Tests for STDDEV indicator."""

  def test_stddev_basic(self, price_data):
    """Test STDDEV calculation."""
    result = stddev(price_data, period=5).to_pandas()

    assert len(result) == len(price_data)
    assert result.isna().sum() == 4  # period - 1

  def test_stddev_formula(self):
    """Verify STDDEV formula."""
    data = pd.Series([100.0, 102.0, 104.0, 103.0, 105.0])

    result = stddev(data, period=3, nbdev=1.0).to_pandas()

    # Manual calc at index 2 (100, 102, 104)
    expected_std = data[:3].std(ddof=0)
    assert result.iloc[2] == pytest.approx(expected_std)

  def test_stddev_nbdev(self):
    """Test STDDEV with nbdev multiplier."""
    data = pd.Series([100.0, 102.0, 104.0, 103.0, 105.0])

    result1 = stddev(data, period=3, nbdev=1.0).to_pandas()
    result2 = stddev(data, period=3, nbdev=2.0).to_pandas()

    # result2 should be 2x result1
    valid = result1.notna()
    np.testing.assert_allclose(result2[valid].values, result1[valid].values * 2)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_stddev_matches_talib(self, price_data):
    """Test STDDEV matches TA-Lib."""
    period = 5
    result = stddev(price_data, period=period, nbdev=1.0).to_pandas()
    expected = talib.STDDEV(price_data.values, timeperiod=period, nbdev=1.0)

    valid = ~np.isnan(expected)
    np.testing.assert_allclose(result.values[valid], expected[valid], rtol=1e-9)
