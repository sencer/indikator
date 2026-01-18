"""Tests for Williams %R indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.willr import willr

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestWillR:
  """Tests for Williams %R indicator."""

  def test_willr_basic(self):
    """Test Williams %R basic calculation."""
    np.random.seed(42)
    n = 50
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = willr(high, low, close, period=14)

    # Check shape
    assert len(result) == len(close)

    # Check %R is in range [-100, 0] where valid
    valid = result.dropna()
    assert (valid >= -100).all()
    assert (valid <= 0).all()

  def test_willr_overbought_oversold(self):
    """Test Williams %R with extreme price levels."""
    # Price at high of range - should show overbought (near 0)
    high = pd.Series([110.0] * 20)
    low = pd.Series([100.0] * 20)
    close = pd.Series([110.0] * 20)  # Close at high

    result = willr(high, low, close, period=5)

    # %R should be near 0 (overbought)
    valid = result.dropna()
    assert (valid > -20).all()

    # Price at low of range - should show oversold (near -100)
    close_low = pd.Series([100.0] * 20)  # Close at low
    result_low = willr(high, low, close_low, period=5)

    valid_low = result_low.dropna()
    assert (valid_low < -80).all()

  def test_willr_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="not empty"):
      willr(empty, empty, empty)

  def test_willr_insufficient_data(self):
    """Test Williams %R with insufficient data."""
    high = pd.Series([105.0, 106.0])
    low = pd.Series([100.0, 101.0])
    close = pd.Series([102.0, 103.0])

    result = willr(high, low, close, period=14)

    # Should return all NaN
    assert result.isna().all()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_willr_matches_talib(self):
    """Test Williams %R matches TA-Lib output."""
    np.random.seed(42)
    n = 100
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2

    result = willr(high, low, close, period=14)
    expected = pd.Series(
      talib.WILLR(high.values, low.values, close.values, timeperiod=14),
    )

    # Compare non-NaN values
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
