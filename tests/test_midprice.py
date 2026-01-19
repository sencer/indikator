"""Tests for MIDPRICE indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.midprice import midprice

try:
  import talib

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


@pytest.fixture
def hl_data():
  """Generate high/low test data."""
  np.random.seed(42)
  n = 100
  close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
  high = close + np.abs(np.random.randn(n) * 0.3)
  low = close - np.abs(np.random.randn(n) * 0.3)
  return {"high": pd.Series(high), "low": pd.Series(low)}


class TestMidprice:
  """Tests for MIDPRICE indicator."""

  def test_midprice_basic(self, hl_data):
    """Test MIDPRICE calculation."""
    result = midprice(hl_data["high"], hl_data["low"], period=14).to_pandas()

    assert len(result) == len(hl_data["high"])
    assert result.isna().sum() == 13  # period - 1

  def test_midprice_formula(self):
    """Verify MIDPRICE formula."""
    high = pd.Series([100.0, 105.0, 110.0, 108.0, 115.0])
    low = pd.Series([90.0, 92.0, 95.0, 93.0, 98.0])

    result = midprice(high, low, period=3).to_pandas()

    # At index 2: highest high = 110, lowest low = 90 -> (110 + 90) / 2 = 100
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx((110 + 90) / 2)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_midprice_matches_talib(self, hl_data):
    """Test MIDPRICE matches TA-Lib."""
    period = 14
    result = midprice(hl_data["high"], hl_data["low"], period=period).to_pandas()
    expected = talib.MIDPRICE(
      hl_data["high"].values, hl_data["low"].values, timeperiod=period
    )

    valid = ~np.isnan(expected)
    np.testing.assert_allclose(result.values[valid], expected[valid], rtol=1e-10)
