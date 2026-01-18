"""Tests for ROC (Rate of Change) indicator."""

import numpy as np
import pandas as pd
import pytest

from indikator.roc import roc

# Try to import talib for comparison tests
try:
  import talib  # type: ignore[import-untyped]

  HAS_TALIB = True
except ImportError:
  HAS_TALIB = False


class TestROC:
  """Tests for ROC indicator."""

  def test_roc_basic(self):
    """Test ROC basic calculation."""
    prices = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])

    result = roc(prices, period=3)

    # Check shape
    assert len(result) == len(prices)

    # First 3 should be NaN
    assert result.isna().iloc[:3].all()

    # Check specific values
    # ROC at index 3: (115 - 100) / 100 * 100 = 15%
    assert result.iloc[3] == pytest.approx(15.0)
    # ROC at index 4: (120 - 105) / 105 * 100 ≈ 14.29%
    assert result.iloc[4] == pytest.approx(14.285714, rel=1e-5)

  def test_roc_negative_change(self):
    """Test ROC with declining prices."""
    prices = pd.Series([100.0, 95.0, 90.0, 85.0, 80.0])

    result = roc(prices, period=2)

    # All valid ROC values should be negative
    valid = result.dropna()
    assert (valid < 0).all()

  def test_roc_empty_data(self):
    """Should raise ValueError when data is empty."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="not empty"):
      roc(empty)

  def test_roc_insufficient_data(self):
    """Test ROC with insufficient data."""
    prices = pd.Series([100.0, 101.0])

    result = roc(prices, period=10)

    # Should return all NaN
    assert result.isna().all()

  def test_roc_period_parameter(self):
    """Test ROC with different periods."""
    prices = pd.Series([
      100.0,
      102.0,
      104.0,
      106.0,
      108.0,
      110.0,
      112.0,
      114.0,
      116.0,
      118.0,
    ])

    result_short = roc(prices, period=1)
    result_long = roc(prices, period=5)

    # Short period should have values earlier
    assert result_short.notna().sum() > result_long.notna().sum()

  @pytest.mark.skipif(not HAS_TALIB, reason="TA-Lib not installed")
  def test_roc_matches_talib(self):
    """Test ROC matches TA-Lib output."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.randn(100) * 0.5))

    result = roc(prices, period=10)
    expected = pd.Series(talib.ROC(prices.values, timeperiod=10))

    # Compare non-NaN values
    valid_mask = result.notna() & expected.notna()
    np.testing.assert_allclose(
      result[valid_mask].values,
      expected[valid_mask].values,
      rtol=1e-10,
    )
