"""Tests for MIDPOINT indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.midpoint import midpoint

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


class TestMidpoint:
  """Tests for MIDPOINT indicator."""

  def test_midpoint_basic(self, price_data):
    """Test MIDPOINT calculation."""
    result = midpoint(price_data, period=14).to_pandas()

    assert len(result) == len(price_data)
    assert result.isna().sum() == 13  # period - 1

  def test_midpoint_formula(self):
    """Verify MIDPOINT formula."""
    data = pd.Series([100.0, 105.0, 110.0, 108.0, 115.0])

    result = midpoint(data, period=3).to_pandas()

    # At index 2: highest = 110, lowest = 100 -> (110 + 100) / 2 = 105
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx((110 + 100) / 2)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_midpoint_matches_talib(self, price_data):
    """Test MIDPOINT matches TA-Lib."""
    period = 14
    result = midpoint(price_data, period=period).to_pandas()
    expected = talib.MIDPOINT(price_data.values, timeperiod=period)

    valid = ~np.isnan(expected)
    np.testing.assert_allclose(result.values[valid], expected[valid], rtol=1e-10)
