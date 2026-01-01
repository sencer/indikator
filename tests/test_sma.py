"""Tests for SMA (Simple Moving Average) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.sma import sma

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestSMA:
  """Tests for SMA indicator."""

  def test_sma_basic(self):
    """Test SMA basic calculation."""
    prices = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])

    result = sma(prices, period=3)

    # Check shape
    assert len(result) == len(prices)

    # First 2 should be NaN (need 3 values)
    assert result.isna().iloc[:2].all()

    # Check actual SMA values
    assert result.iloc[2] == pytest.approx((100 + 102 + 104) / 3)
    assert result.iloc[3] == pytest.approx((102 + 104 + 106) / 3)
    assert result.iloc[4] == pytest.approx((104 + 106 + 108) / 3)

  def test_sma_equal_weights(self):
    """Test that SMA uses equal weights."""
    prices = pd.Series([100.0, 100.0, 100.0, 100.0, 200.0])

    result = sma(prices, period=5)

    # SMA should be exact average
    assert result.iloc[-1] == pytest.approx((100 + 100 + 100 + 100 + 200) / 5)

  def test_sma_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty_data = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="non-empty"):
      sma(empty_data)

  def test_sma_insufficient_data(self):
    """Test SMA with insufficient data."""
    prices = pd.Series([100.0, 101.0])
    result = sma(prices, period=5)

    # Should return all NaN
    assert result.isna().all()

  def test_sma_period_parameter(self):
    """Test SMA with different period sizes."""
    prices = pd.Series(
      [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0] * 3
    )

    result_short = sma(prices, period=3)
    result_long = sma(prices, period=10)

    # Short period should have values earlier
    assert result_short.notna().sum() > result_long.notna().sum()

  def test_sma_rolling_consistency(self):
    """Test SMA matches pandas rolling mean."""
    prices = pd.Series([
      100.0,
      102.0,
      101.0,
      103.0,
      105.0,
      104.0,
      106.0,
      108.0,
      107.0,
      109.0,
    ])

    result = sma(prices, period=5)
    expected = prices.rolling(window=5).mean()

    pd.testing.assert_series_equal(
      result,
      expected.rename("sma"),
      check_exact=False,
      atol=1e-10,
    )

  def test_sma_with_inf(self):
    """Test SMA with Inf values."""
    prices = pd.Series([100.0, 102.0, np.inf, 103.0, 105.0] * 5)

    with pytest.raises(ValueError, match="must be finite"):
      sma(prices)

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_sma_matches_talib(self):
    """Test SMA matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = sma(prices, period=14)
    expected = pd.Series(talib.SMA(prices.values, timeperiod=14))

    # Compare non-NaN values
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
